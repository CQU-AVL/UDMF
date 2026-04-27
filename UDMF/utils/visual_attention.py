import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from jaad_data import JAAD
from jaad_preprocessing import *
from my_dataset2 import RandomResizeSeq, padding
from networks.double_model import MyModel
import gc
import torch.backends.cudnn as cudnn
import torchvision.transforms as T
import cv2
from torch import nn

# ================= 配置区域 =================
OUTPUT_FOLDER = "/media/avl/disk1/lr/PedCMT/visual_attention"  # 结果保存路径
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
    
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= 全局绘图配置 =================
# 统一字体和字号，确保所有图风格一致
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 或者 'SimHei' (如果需要中文)
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 16          # 全局字体大小
plt.rcParams['axes.titlesize'] = 16     # 标题字体大小
plt.rcParams['axes.labelsize'] = 16     # 轴标签字体大小
plt.rcParams['xtick.labelsize'] = 16    # X轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 16    # Y轴刻度字体大小

# 统一画布尺寸 (宽, 高) - 英寸
UNIFIED_FIGSIZE = (6, 5) 
UNIFIED_FIGSIZE2 = (24, 5) 
# ===============================================

def plot_sharp_attention(attn, save_path):
    """
    强制拉伸微小差异的 Attention 可视化
    """
    # 转 numpy
    if torch.is_tensor(attn): attn = attn.detach().cpu().numpy()
    
    # 形状检查 (1, T)
    if attn.ndim == 3: attn = attn[0] # 取 batch 0
    if attn.shape[0] != 1: attn = attn.mean(axis=0, keepdims=True) # 如果是多头，取平均

    T = attn.shape[1]
    
    # === 诊断信息 (保持不变) ===
    print(f"\n[诊断] Attention 统计信息:")
    print(f"  - Mean: {attn.mean():.6f} (理想值 1/{T} = {1/T:.6f})")
    print(f"  - Std : {attn.std():.8f}")
    print(f"  - Max - Min: {attn.max() - attn.min():.8f}")
    
    # === 强制锐化 ===
    diff = attn.max() - attn.min()
    if diff < 1e-9:
        norm_attn = attn 
        title = "Attention Collapsed"
    else:
        norm_attn = (attn - attn.min()) / diff
        title = "Sharpened Attention"
        print("  -> 已应用强制锐化")

    # === 绘图 ===
    plt.figure(figsize=UNIFIED_FIGSIZE2) # 使用统一尺寸
    
    sns.heatmap(norm_attn, 
                annot=False, 
                cmap='jet', 
                cbar=True,
                xticklabels=[f"{i}" for i in range(T)],
                yticklabels=["image_token"],
                cbar_kws={'label': 'Relative Weight'}) # 统一 colorbar 标签
    
    # 调整标题和标签
    plt.title(title) 
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000)
    plt.close()
    print(f"  -> 图表已保存: {save_path}")

def check_feature_diversity(motion_features, save_path):
    """
    检查 Motion Tokens (Key) 自身是否有区分度
    """
    # 转为 float
    feats = motion_features[0].float()
    
    # 计算归一化和相似度
    norm = feats.norm(p=2, dim=1, keepdim=True)
    feats_norm = feats / (norm + 1e-8)
    similarity = torch.mm(feats_norm, feats_norm.t())
    
    # === 诊断信息 ===
    sim_min = similarity.min().item()
    if sim_min > 0.99:
        print("  !!! 严重警告: 特征几乎一模一样 !!!")
    else:
        print("  -> Motion 特征具有区分度。")

    # === 绘图 ===
    try:
        plt.figure(figsize=UNIFIED_FIGSIZE) # 使用统一尺寸
        
        # 强制设置范围 0.8-1.0 以便看清差异，如果不需要固定范围可去掉 vmin/vmax
        sns.heatmap(similarity.detach().cpu().numpy(), 
                    cmap='viridis', 
                    vmin=0.8, vmax=1.0,
                    square=True, # 保持矩阵正方形比例
                    cbar_kws={'label': 'Cosine Similarity'})
        
        plt.title("Motion Token Self-Similarity")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=1000)
        plt.close()
        print(f"  -> 自相似度矩阵已保存: {save_path}")
    except Exception as e:
        print(f"  (绘图失败: {e})")

def plot_attention_matrix(attn_weights, x_labels, y_labels, save_path, title="Attention Matrix"):
    """
    绘制 Attention Matrix 热力图
    """
    plt.figure(figsize=UNIFIED_FIGSIZE) # 使用统一尺寸
    
    # 自动判断是否显示数值
    annot = True if (attn_weights.shape[0] < 15 and attn_weights.shape[1] < 15) else False
    
    try:
        sns.heatmap(attn_weights, 
                    annot=annot,
                    fmt=".2f", 
                    cmap='jet',       
                    xticklabels=x_labels, 
                    yticklabels=y_labels,
                    square=True, # 保持矩阵正方形比例
                    cbar_kws={'label': 'Attention Weight'})
    except ValueError as e:
        print(f"绘图错误: {e}")
        plt.close()
        return
    
    plt.title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=1000)
    plt.close()
    print(f"[已保存矩阵图] {save_path}")

def main(args):
    # 1. 环境准备
    if not torch.cuda.is_available():
        print("错误: 需要 GPU 环境")
        return
    device = torch.device("cuda")
    ensure_dir(OUTPUT_FOLDER)
    torch.backends.cudnn.enabled = False
    
    # 2. 数据与模型加载 (模拟部分，请替换为你的真实逻辑)
    print("正在加载模型与数据...")
    
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
    checkpoint = torch.load("/media/avl/disk1/lr/PedCMT/checkpoints/JAAD_checkpoint/checkpoint0000.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    # =========================================================
    # 关键步骤：定义 Hook 来捕获 Attention 权重
    # =========================================================
    
    # 存储容器
    activations = {
            "SA_b": None,
            "SA_v": None,
            "cross_atten": None,
            "fused_atten": None
        }

    def get_attn_hook(name):
            def hook(module, input, output):
                # PyTorch MultiheadAttention 输出通常是 (attn_output, attn_output_weights)
                # attn_output_weights shape: (Batch, Query_Len, Key_Len) 或 (Batch, Heads, Q, K)
                
                weights = None
                # 尝试从输出中提取权重
                if isinstance(output, tuple) and len(output) == 2:
                    weights = output[1]  # index 1 通常是 weights
                elif isinstance(output, torch.Tensor):
                    # 有些魔改实现可能直接返回权重，或者需要检查是否是 weights
                    if output.dim() >= 3: 
                        weights = output
                
                if weights is None:
                    # 注意：如果 forward 时没有设置 need_weights=True，这里可能拿到 None
                    return

                # 如果包含多头维度 (Batch, Heads, Q, K)，则平均化
                if weights.dim() == 4:
                    weights = weights.mean(dim=1)
                
                # 保存到字典
                activations[name] = weights.detach().cpu()
            return hook
    inputs_cache = {}
    def debug_hook(module, input, output):
        # input 通常是 (query, key, value)
        # 你的 fused_atten 输入顺序可能是 (img, motion) 或者在内部生成 Q, K
        # 这里假设我们能捕获到 motion_tokens (作为 Key)
        # 注意：这取决于你具体的 forward 实现，如果 hook 只能拿到 tensor，
        # 建议直接在 forward 里 print 或保存
        
        # 捕获 Attention Weights
        if isinstance(output, tuple):
            attn_weights = output[1]
        else:
            attn_weights = output
        
        inputs_cache['weights'] = attn_weights.detach()
        
        # 尝试捕获 Key (Motion Features)
        # 如果 input 是 tuple 且长度 >= 2
        if isinstance(input, tuple) and len(input) >= 2:
            # 假设第2个参数是 Key (Motion)
            inputs_cache['query'] = input[0].detach()

    # =========================================================
    # [用户修改区] 注册 Hook 到你的模型层
    # =========================================================
    print("注册 Hook...")
    
    # 1. Bbox 时序自注意力 (如果有)
    # 假设路径: model.bbox_branch.encoder.layers[-1].self_attn
    if hasattr(model, 'SA_b') and hasattr(model.SA_b, 'att'):
        model.SA_b.att.register_forward_hook(get_attn_hook("SA_b"))
    
    if hasattr(model, 'SA_v') and hasattr(model.SA_v, 'att'):
        model.SA_v.att.register_forward_hook(get_attn_hook("SA_v"))
        
    if hasattr(model, 'cross_atten') and hasattr(model.cross_atten, 'att'):
        model.cross_atten.att.register_forward_hook(get_attn_hook("cross_atten"))
        
    if hasattr(model, 'fused_atten') and hasattr(model.fused_atten, 'att'):
        model.fused_atten.att.register_forward_hook(get_attn_hook("fused_atten"))

    model.fused_atten.att.register_forward_hook(debug_hook)

    # =========================================================
    # 推理循环
    # =========================================================
    # [请替换] 这里使用你的真实数据循环
    # for i, batch in enumerate(test_loader):
    for i in range(1): # 仅演示前5个样本
        print(f"Processing sample {i}...")
        
        # 清空上一轮的缓存
        for k in activations: activations[k] = None
        
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
        model.eval()
        with torch.no_grad():
            model(im_data, bbox_input[None], vel_input[None], False)

        if 'weights' in inputs_cache:
            plot_sharp_attention(inputs_cache['weights'], f"{OUTPUT_FOLDER}/debug_attention{i}.png")
        if 'query' in inputs_cache:
            check_feature_diversity(inputs_cache['query'], f"{OUTPUT_FOLDER}/motion_self_similarity{i}.png")
        # =========================================================
        # 绘图逻辑
        # =========================================================
        base_name = f"sample_{i}"
        
        # ---------------------------------------------------------
        # 图1: Bbox 模态内时序注意力 (Time x Time)
        # ---------------------------------------------------------
        if activations["SA_b"] is not None:
            mat = activations["SA_b"][0] # 取 Batch 第一个 (T, T)
            seq_len = mat.shape[0]
            labels = [f"{t}" for t in range(seq_len)]
            
            plot_attention_matrix(mat, labels, labels, 
                                  os.path.join(OUTPUT_FOLDER, f"{base_name}_SA_b.png"),
                                  title="Intra-Modal Self-Attention (Bbox)")

        # --- 图 2: Velocity 自注意力 (SA_v) ---
        if activations["SA_v"] is not None:
            mat = activations["SA_v"][0]
            seq_len = mat.shape[0]
            labels = [f"{t}" for t in range(seq_len)]
            
            plot_attention_matrix(mat, labels, labels, 
                                  os.path.join(OUTPUT_FOLDER, f"{base_name}_SA_v.png"),
                                  title="Intra-Modal Self-Attention (Velocity)")

        # --- 图 3: Bbox 和 Vel 交叉注意力 (cross_atten) ---
        if activations["cross_atten"] is not None:
            mat = activations["cross_atten"][0] # (Query_Len, Key_Len)
            h, w = mat.shape
            
            # Q=B, K=V (或者反过来，取决于你的 forward 写法)
            # 这里统一使用通用标签，你可以根据实际情况改为 "Vel_t" 和 "Bbox_t"
            y_labels = [f"{t}" for t in range(h)]
            x_labels = [f"{t}" for t in range(w)]
            
            plot_attention_matrix(mat, x_labels, y_labels, 
                                  os.path.join(OUTPUT_FOLDER, f"{base_name}_cross_atten.png"),
                                  title="Cross-Modal Attention (Bbox & Vel)")


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