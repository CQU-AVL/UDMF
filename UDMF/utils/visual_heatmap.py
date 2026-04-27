import torch
import cv2
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import numpy as np
from PIL import Image
import gc
import argparse
import torchvision.transforms as T
from collections import OrderedDict
import torch
from pytorch_grad_cam import GradCAM, EigenCAM

# 假设这些是你项目中的自定义模块
from jaad_data import JAAD
from jaad_preprocessing import *
from my_dataset2 import RandomResizeSeq, padding
from networks.double_model import MyModel

# ================= 配置区域 =================
OUTPUT_FOLDER = "/media/avl/disk1/lr/PedCMT/visual_heatmap"  # 结果保存路径
ENABLE_VISUALIZATION = False         # 是否开启可视化
TEST_FRAMES = 10                     # 仅测试前 N 帧 (设为 None 则跑完全部)
# ===========================================

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def label_transforms(y):
    return y.float()

def vel_norm(v):
    vmax = torch.max(v)
    if vmax == 0:
        return v
    else:
        return v / vmax

def main(args):
    # 1. 环境与设备
    if not torch.cuda.is_available():
        print("错误: 需要 GPU 环境")
        return
    device = torch.device("cuda")
    ensure_dir(OUTPUT_FOLDER)
    
    # 2. 数据准备 (JAAD处理部分)
    print("正在加载数据...")
    data_opts = {
        'fstride': 1, 'sample_type': args.bh, 'subset': 'default',
        'height_rng': [0, float('inf')], 'squarify_ratio': 0,
        'data_split_type': 'default', 'seq_type': 'crossing',
        'min_track_size': 15,
        'random_params': {'ratios': None, 'val_data': True, 'regen_data': False},
        'kfold_params': {'num_folds': 5, 'fold': 1},
    }
    tte = [30, 60]
    imdb = JAAD(data_path=args.set_path)
    seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)
    tte_seq_test, traj_seq_test = tte_dataset(seq_test, tte, 0, args.times_num)
    
    # 提取数据
    raw_bbox_test = tte_seq_test['bbox'] 
    vel_test = tte_seq_test['vehicle_act']
    action_test = tte_seq_test['activities']
    
    # 图片路径处理
    img_path_test = sorted(
        [f"{args.cache_dir}_all/test_cache/{i}.jpg" for i in range(len(raw_bbox_test)) if
            os.path.exists(f"{args.cache_dir}_all/test_cache/{i}.jpg")],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    # 数据预处理
    normalized_bbox_test = normalize_bbox(raw_bbox_test) # 归一化用于模型输入
    label_action_test = prepare_label(action_test)
    
    # 转为 Tensor
    bbox_tensor_all = torch.Tensor(np.array(normalized_bbox_test))
    label_tensor_all = torch.Tensor(np.array(label_action_test))
    vel_tensor_all = torch.Tensor(np.array(vel_test))
    
    # 清理内存
    del seq_test, tte_seq_test, traj_seq_test, normalized_bbox_test
    gc.collect()

    # 3. 加载模型
    print("正在加载模型...")
    model = MyModel(args).to(device)
    print(model)
    checkpoint = torch.load("/media/avl/disk1/lr/PedCMT/checkpoints/JAAD_checkpoint/checkpoint0000.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

        # 1. 定义存储容器
    activations = {
    "backbone": None,         # C5 feature map  (B, 2048, Hf, Wf)
    "encoder_attn": []        # list of 12 attention maps
    }

    # 2. 定义 Hook 函数
    def hook_backbone(module, input, output):
        # output: (B, 2048, Hf, Wf)
        activations["backbone"] = output.detach().cpu()

    def register_backbone_hook(model):
        model.backbone_img.body.layer4.register_forward_hook(hook_backbone)


    def make_encoder_hook(idx):
        """
        Hook: 捕获 self_attn 的 output，尽量拿到 attn_weights（原始）
        不对形状做不可逆的假设，直接保存原始张量以便后面处理。
        """
        def hook(module, input, output):
            # Many implementations: output = (attn_output, attn_weights)
            # But some implementations may return attn_weights only or other structure
            if isinstance(output, (tuple, list)) and len(output) >= 2:
                attn_weights = output[1]
            elif isinstance(output, (tuple, list)) and len(output) == 1:
                attn_weights = output[0]
            else:
                # 有些自定义 MultiheadAttention 直接返回 attn_weights
                attn_weights = output

            # detach -> cpu
            try:
                aw = attn_weights.detach().cpu()
            except Exception as e:
                print(f"[EncoderHook {idx}] cannot detach attn_weights, skipping. type={type(attn_weights)} err={e}")
                return

            # 保存原始 attn（后续我们会统一处理）
            activations["encoder_attn"].append(aw)
        return hook
    
    def hook_encoder_self_attn(module, input, output):
        # output[1] = attention weights of MultiheadAttention
        attn = output[1]                 # shape: (B, heads, HW, HW)
        attn = attn.mean(dim=1)          # average over heads → (B, HW, HW)
        activations["encoder_attn"].append(attn.detach().cpu())

    def register_encoder_hooks(model):
        encoder_layers = model.encoder.layers

        def make_hook(idx):
            def hook(module, input, output):
                attn_output, attn_weights = output  # (B, heads, L, L)
                activations["encoder_attn"].append(attn_weights.detach().cpu())
            return hook

        for i, layer in enumerate(encoder_layers):
            layer.self_attn.register_forward_hook(make_hook(i))


    # 4. 可视化函数 (修复版)
    class BackboneWrapper(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, x):
            # x expected: plain tensor (B,3,H,W)
            # but backbone may accept (tensor, mask) - wrapper.forward only receives tensor
            # so call backbone with only tensor and handle whatever it returns
            try:
                out = self.backbone(x)   # may raise if backbone expects mask
            except TypeError:
                # try with a simple mask of False (no padding)
                B, C, H, W = x.shape
                mask = torch.zeros((B, H, W), dtype=torch.bool, device=x.device)
                out = self.backbone(x, mask)

            # Normalize outputs to a single tensor
            if isinstance(out, (dict, OrderedDict)):
                vals = list(out.values())
                feat = vals[-1]
            elif isinstance(out, (list, tuple)):
                feat = out[-1]
            else:
                feat = out

            # feat expected shape: (B, C, Hf, Wf)
            gap = feat.mean(dim=(2,3))   # (B, C)
            return gap

    # ---------- main Grad-CAM function ----------
    def visualize_backbone_gradcam(model, img_tensor, img_pil, save_path, use_eigencam=True):
        """
        Robust Grad-CAM for your backbone.
        - model: full MyModel instance
        - img_tensor: torch.Tensor (1,3,H,W) OR NestedTensor-like (has .tensors)
        - img_pil: PIL.Image
        """

        # --- ensure plain tensor ---
        if not isinstance(img_tensor, torch.Tensor):
            # try NestedTensor -> extract .tensors
            if hasattr(img_tensor, 'tensors'):
                img_tensor = img_tensor.tensors
            else:
                raise ValueError("img_tensor must be torch.Tensor or NestedTensor-like with .tensors")

        device = next(model.parameters()).device
        img_tensor = img_tensor.to(device)

        # --- get backbone module (your ResNet body) ---
        backbone_module = model.backbone_img.body

        # --- pick target layer inside backbone ---
        # for ResNet: last Bottleneck block is layer4[-1]
        try:
            target_layer = backbone_module.layer4[-1]
        except Exception:
            # fallback: use backbone_module itself
            target_layer = backbone_module

        # --- build wrapper (so CAM libs see a simple forward returning Tensor) ---
        wrapper = BackboneWrapper(backbone_module).to(device)

        # --- select CAM impl ---
        cam_impl = EigenCAM if use_eigencam else __import__('pytorch_grad_cam').base_cam.GradCAM
        cam = cam_impl(model=wrapper, target_layers=[target_layer], use_cuda=img_tensor.is_cuda)

        # --- run CAM ---
        # wrapper expects plain tensor; if backbone expects mask internally our wrapper tries both ways
        grayscale_cam = cam(input_tensor=img_tensor, targets=None)[0]  # normalized Hf x Wf

        # --- resize to original image size and overlay ---
        cam_resized = cv2.resize(grayscale_cam, img_pil.size, interpolation=cv2.INTER_LINEAR)
        heat = np.uint8(255 * cam_resized)
        heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(img_bgr, 0.5, heat_color, 0.5, 0)

        cv2.imwrite(save_path, overlay)
        print("[Saved Backbone CAM]:", save_path)
        
    
    def normalize_attn_to_B_LL(attn_tensor, L):
        """
        将多种可能的 attn 格式转换为 (B, L, L) 或返回 None。
        支持: (B, heads, L, L), (heads, L, L), (B, L, L), (L, L), (L,), (1,)
        """
        import torch
        aw = attn_tensor
        # ensure torch tensor on cpu
        if not torch.is_tensor(aw):
            try:
                aw = torch.tensor(aw)
            except Exception:
                return None

        aw = aw.cpu()
        d = aw.dim()
        if d == 4:
            # (B, heads, L, L)
            B, heads, l1, l2 = aw.shape
            if l1 == L and l2 == L:
                return aw.mean(dim=1)  # -> (B, L, L)
            # try reshape if flattened inside
            if l1 * l2 == L * L:
                try:
                    return aw.view(B, heads, L, L).mean(dim=1)
                except:
                    return None

        elif d == 3:
            # could be (B, L, L) or (heads, L, L)
            a, b, c = aw.shape
            if a > 1 and b == L and c == L:
                return aw  # (B, L, L)
            if b == L and c == L:
                # likely (heads, L, L) -> average heads -> (1, L, L)
                return aw.mean(dim=0).unsqueeze(0)
            # else unknown
            return None

        elif d == 2:
            # (L, L)
            l1, l2 = aw.shape
            if l1 == L and l2 == L:
                return aw.unsqueeze(0)
            else:
                return None

        elif d == 1:
            # vector maybe length L*L
            if aw.numel() == L * L:
                return aw.view(1, L, L)
            else:
                return None

        elif d == 0:
            # scalar
            return None
        else:
            return None


    # ---------- robust visualize_encoder_attn ----------
    def visualize_encoder_attn(att_raw, image_pil, feat_hw, save_path,
                           head_idx=0, query="center", alpha=0.5, debug_dir="attn_debug"):
        """
        Robust visualization for encoder self-attention.
        Supports inputs:
        - (B, heads, L, L)
        - (B, L, L)
        - (heads, L, L)
        - (L, L)
        - (B, L)
        - (L,)
        If att_raw cannot be interpreted, function will skip and save debug info.
        """

        Hf, Wf = feat_hw
        L = Hf * Wf

        # ensure torch tensor on cpu
        if not torch.is_tensor(att_raw):
            try:
                att = torch.tensor(att_raw)
            except Exception:
                print(f"[visualize_encoder_attn] att_raw not tensor and cannot convert: {type(att_raw)}")
                return
        else:
            att = att_raw

        att = att.cpu()

        # helper: save debug artifact
        def save_debug(msg):
            os.makedirs(debug_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(save_path))[0]
            dbg_path = os.path.join(debug_dir, f"{base}_debug.txt")
            with open(dbg_path, "a") as f:
                f.write(msg + "\n")
            print(msg, f"(saved to {dbg_path})")

        # Print shape quickly
        ashape = tuple(att.shape)
        # CASE A: full attention tensor (B, heads, L, L)
        if att.dim() == 4:
            B, heads, l1, l2 = att.shape
            if l1 != L or l2 != L:
                save_debug(f"[visualize_encoder_attn] bad spatial dims in (B,heads,L,L): {att.shape}, expected L={L}")
                return
            # choose head and query
            head_idx = min(max(0, head_idx), heads-1)
            att_h = att[0, head_idx]   # (L, L)

            if query == "center":
                qi = L // 2
            elif isinstance(query, int):
                qi = query % L
            else:
                qi = 0

            att_vec = att_h[qi]  # (L,)
            # reshape
            try:
                att_map = att_vec.view(Hf, Wf).numpy()
            except Exception as e:
                save_debug(f"[visualize_encoder_attn] reshape fail for (B,heads,L,L) -> {e}")
                return

        # CASE B: (B, L, L)
        elif att.dim() == 3:
            a,b,c = att.shape
            # if a==B and b==L and c==L -> good
            if b == L and c == L:
                att_mat = att[0]  # (L, L)
                if query == "center":
                    qi = L // 2
                elif isinstance(query, int):
                    qi = query % L
                else:
                    qi = 0
                att_vec = att_mat[qi]
                try:
                    att_map = att_vec.view(Hf, Wf).numpy()
                except Exception as e:
                    save_debug(f"[visualize_encoder_attn] reshape fail for (B,L,L) -> {e}")
                    return
            # else maybe it's (heads, L, L)
            elif a < 128 and b == L and c == L:
                # treat as (heads, L, L)
                att_h = att.mean(dim=0) if a > 1 else att[0]
                # choose head_idx after averaging - just take att_h as (L,L)
                if query == "center":
                    qi = L // 2
                elif isinstance(query, int):
                    qi = query % L
                else:
                    qi = 0
                att_vec = att_h[qi]
                try:
                    att_map = att_vec.view(Hf, Wf).numpy()
                except Exception as e:
                    save_debug(f"[visualize_encoder_attn] reshape fail for (heads,L,L) case -> {e}")
                    return
            else:
                save_debug(f"[visualize_encoder_attn] unexpected 3D att shape: {att.shape}")
                return

        # CASE C: (L, L)
        elif att.dim() == 2:
            a,b = att.shape
            if a == L and b == L:
                att_mat = att
                if query == "center":
                    qi = L // 2
                elif isinstance(query, int):
                    qi = query % L
                else:
                    qi = 0
                att_vec = att_mat[qi]
                try:
                    att_map = att_vec.view(Hf, Wf).numpy()
                except Exception as e:
                    save_debug(f"[visualize_encoder_attn] reshape fail for (L,L) -> {e}")
                    return
            else:
                # maybe it's (B, L) or (1, L) but represented as 2D; treat below
                if a == 1 and b == L:
                    att_vec = att[0]
                    att_map = att_vec.view(Hf, Wf).numpy()
                else:
                    save_debug(f"[visualize_encoder_attn] unexpected 2D att shape: {att.shape}")
                    return

        # CASE D: (B, L) or (L,)
        elif att.dim() == 1:
            if att.numel() == L:
                att_map = att.view(Hf, Wf).numpy()
            else:
                save_debug(f"[visualize_encoder_attn] 1D att vector length {att.numel()} != L {L}")
                return

        elif att.dim() == 2 and att.shape[0] > 1 and att.shape[1] != L:
            # unexpected 2D, try flatten fallback
            flat = att.flatten()
            if flat.numel() == L:
                att_map = flat.view(Hf, Wf).numpy()
            else:
                save_debug(f"[visualize_encoder_attn] 2D unexpected and cannot flatten->L. shape={att.shape}")
                return

        else:
            save_debug(f"[visualize_encoder_attn] unsupported att.dim={att.dim()} shape={att.shape}")
            return

        # at this point, att_map is numpy (Hf, Wf)
        # normalize and upsample
        att_map = np.array(att_map, dtype=np.float32)
        att_map = att_map - att_map.min()
        denom = att_map.max() if att_map.max() != 0 else 1.0
        att_map = att_map / (denom + 1e-6)

        # resize to original image
        orig_w, orig_h = image_pil.size
        att_up = cv2.resize(att_map, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        att_uint8 = np.uint8(255 * att_up)
        att_color = cv2.applyColorMap(att_uint8, cv2.COLORMAP_JET)
        # convert to RGB for blending with PIL image array
        att_color_rgb = cv2.cvtColor(att_color, cv2.COLOR_BGR2RGB)

        # blend with original (PIL -> numpy)
        img_np = np.array(image_pil).astype(np.uint8)
        blended = (0.45 * img_np + 0.55 * att_color_rgb).astype(np.uint8)

        Image.fromarray(blended).save(save_path)
        print(f"[Saved encoder attention] {save_path}  (att shape was {ashape})")


    # ---------- safe c5 extractor ----------
    def extract_c5_tensor_from_feats(feats):
        """
        feats: could be list/tuple (values), OrderedDict, or a single Tensor, or a NestedTensor-like
        Returns: c5_tensor (torch.Tensor on cpu) and (Hf, Wf)
        """
        import torch
        from collections import OrderedDict

        if isinstance(feats, (dict, OrderedDict)):
            vals = list(feats.values())
            last = vals[-1]
        elif isinstance(feats, (list, tuple)):
            last = feats[-1]
        else:
            last = feats

        # last might be NestedTensor-like with .tensors, or a plain tensor
        if hasattr(last, 'tensors'):
            t = last.tensors
        else:
            t = last

        if not torch.is_tensor(t):
            try:
                t = torch.tensor(t)
            except Exception as e:
                raise RuntimeError(f"extract_c5_tensor_from_feats: cannot convert last to tensor: {type(last)} error:{e}")

        # move to cpu for visualization pipeline
        t = t.detach().cpu()
        if t.dim() != 4:
            raise RuntimeError(f"extract_c5_tensor_from_feats: expected 4D tensor, got shape {t.shape}")

        Hf, Wf = t.shape[-2], t.shape[-1]
        return t, (Hf, Wf)


    def get_backbone_feats(model, img_tensor):
        """
        Robustly call model.backbone_img (handles signatures that require mask,
        NestedTensor inputs, and different output types).

        Returns:
            features: list/tuple or tensor representing backbone features.
                    If backbone returns (features, pos) or OrderedDict, function
                    returns the features part (list or OrderedDict values).
        """
        # Normalize input: if NestedTensor-like, extract .tensors
        if not isinstance(img_tensor, torch.Tensor):
            if hasattr(img_tensor, 'tensors'):
                x = img_tensor.tensors
            else:
                raise ValueError("img_tensor must be torch.Tensor or NestedTensor-like with .tensors")
        else:
            x = img_tensor

        x = x.to(next(model.parameters()).device)

        backbone = model.backbone_img

        # Try calling backbone in several ways and return the features part.
        # We'll attempt:
        # 1) backbone(x)                # works if forward accepts just tensor
        # 2) backbone(x, mask)          # for implementations that need mask
        # 3) backbone.body(x) / backbone.body(x, mask)
        last_exception = None
        out = None

        # helper to build a default mask (all False -> no padding)
        def build_mask_from_tensor(t):
            B, C, H, W = t.shape
            return torch.zeros((B, H, W), dtype=torch.bool, device=t.device)

        # candidate callables and signatures to try
        candidates = [
            (backbone, (x,)),               # backbone(x)
            (backbone, (x, build_mask_from_tensor(x))),  # backbone(x, mask)
            (backbone.body, (x,)),          # backbone.body(x)
            (backbone.body, (x, build_mask_from_tensor(x))), # backbone.body(x,mask)
        ]

        for func, args in candidates:
            try:
                out = func(*args)
                # succeeded
                break
            except TypeError as e:
                last_exception = e
                continue
            except Exception as e:
                # other runtime error; keep last_exception and continue trying
                last_exception = e
                continue

        if out is None:
            # all attempts failed; raise informative error
            raise RuntimeError(f"All attempts to call backbone failed. Last exception: {repr(last_exception)}")

        # out may be: OrderedDict / dict / list / tuple / tensor / (features, pos)
        # normalize to 'features' (list or tensor). If out is (features, pos) return features.
        if isinstance(out, (list, tuple)) and len(out) == 2 and (isinstance(out[0], (list, tuple, dict)) or torch.is_tensor(out[0])):
            # common pattern: (features, pos)
            features = out[0]
        elif isinstance(out, (dict, OrderedDict)):
            # IntermediateLayerGetter often returns OrderedDict of feature maps
            # convert to list of values to be consistent
            features = list(out.values())
        else:
            # could be list of features or a single tensor
            features = out

        return features

    # -------------------------------------------------------
    # 修改后的主循环调用
    # -------------------------------------------------------
    # 在 main 函数的循环里：

    register_backbone_hook(model)
    register_encoder_hooks(model)

    # --- 主循环 ---
    # 只测试前几张，避免弹窗太多卡死
    for i in range(3): 
        print(f"Processing image {i}...")
        
        # 每次循环前清空，防止数据污染
        activations["backbone"] = None
        activations["encoder_attn"] = []

        # 数据准备
        bbox_input = bbox_tensor_all[i].to(device)
        vel_input = vel_norm(vel_tensor_all[i]).to(device)
        img_path = img_path_test[i]
        
        # 保留原始 PIL 图片用于画图
        im_pil = Image.open(img_path).convert('RGB') 
        transforms = RandomResizeSeq([800], max_size=1333)
        normalize = T.ToTensor()
        
        # 模型输入变换
        im_data = transforms(normalize(im_pil))[None].to(device)
        im_data = padding(im_data, len(im_data))
        
        # 推理
        model.eval()
        with torch.no_grad():
            model(im_data, bbox_input[None], vel_input[None], False)

        # --- Encoder 12 层 self-attn ---
        # 获取 C5 的空间分辨率
        # 你 backbone 最终输出 (B,2048,Hf,Wf)
        feats = get_backbone_feats(model, im_data)   # as before
        try:
            c5_tensor, (Hf, Wf) = extract_c5_tensor_from_feats(feats)  # c5_tensor on CPU
        except Exception as e:
            print("Failed to extract c5:", e)
            # skip this image
            continue

        # If visualize_backbone_gradcam expects tensor on device, send the plain im_data tensor (non-nested)
        # Ensure im_data is plain Tensor (not NestedTensor) - if it's NestedTensor-like, use .tensors
        if not isinstance(im_data, torch.Tensor) and hasattr(im_data, 'tensors'):
            im_plain = im_data.tensors
        else:
            im_plain = im_data

        visualize_backbone_gradcam(
            model,
            im_plain,        # plain tensor (1,3,H,W)
            im_pil,
            f"backbone_gradcam_img{i}.jpg"
        )


        for idx, att in enumerate(activations["encoder_attn"]):
            save_path = f"encoder_layer_{idx}.jpg"
            visualize_encoder_attn(att, im_pil, (Hf, Wf), save_path)

        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Pedestrian Crossing Intention Prediction.')
    # ... (保持原本的 arguments 不变) ...
    # 建议添加一个默认参数防止报错，或者保留你原有的参数
    parser.add_argument('--set_path', type=str, default='/media/avl/disk1/lr/JAAD/JAAD-JAAD_2.0')
    parser.add_argument('--bh', type=str, default='all')
    parser.add_argument('--cache_dir', type=str, default="/media/avl/disk1/lr/JAAD/JAAD-JAAD_2.0/select_images")
    parser.add_argument('--times_num', type=int, default=15, help='sequence length') 
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--dff', type=int, default=512)
    parser.add_argument('--num_head', type=int, default=8)
    parser.add_argument('--num_points', type=int, default=16)
    parser.add_argument('--bbox_input', type=int, default=4)
    parser.add_argument('--vel_input', type=int, default=1)
    parser.add_argument('--time_crop', type=bool, default=False) 
    parser.add_argument('--bv_input', type=int, default=5)
    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--annealing_step', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--backbone', type=str, default="resnet50")
    parser.add_argument('--num_encoder', type=int, default=12, help='the number of encoder.')
    
    args = parser.parse_args()
    main(args)