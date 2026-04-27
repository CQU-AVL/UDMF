import torch
import cv2
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import numpy as np
from PIL import Image
import gc
import argparse
import torchvision.transforms as T

# 假设这些是你项目中的自定义模块
from jaad_data import JAAD
from jaad_preprocessing import *
from my_dataset2 import RandomResizeSeq, padding
from networks.double_model import MyModel

# ================= 配置区域 =================
OUTPUT_FOLDER = "/media/avl/disk1/lr/PedCMT/visual_results"  # 结果保存路径
ENABLE_VISUALIZATION = False         # 是否开启可视化
TEST_FRAMES = None                     # 仅测试前 N 帧 (设为 None 则跑完全部)
# ===========================================

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def visualize_and_save(orig_image_np, bbox, pred_label, gt_label, conf, frame_id):
    """
    args:
        orig_image_np: 原始图像数据 (H, W, 3) BGR格式
        bbox: 原始像素坐标 [x1, y1, x2, y2]
        pred_label: 预测结果字符串 "cross" or "not cross"
        gt_label: 真实标签字符串 "cross" or "not cross"
    """
    # 复制图像
    vis_img = orig_image_np.copy()
    
    # --- 1. 绘制边界框 (仅框，无文字) ---
    # 预测框颜色：预测是 cross 为绿，否则为红
    box_color = (0, 255, 0) if pred_label == "cross" else (0, 0, 255)
    
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(vis_img, (x1, y1), (x2, y2), box_color, 3)
    
    # --- 2. 给图片底部增加黑色边框用于写字 ---
    # top, bottom, left, right (底部增加 60 像素)
    vis_img = cv2.copyMakeBorder(vis_img, 0, 60, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    h, w = vis_img.shape[:2]
    
    # --- 3. 准备文字内容 ---
    pred_text = f"Pred: {pred_label}"
    gt_text = f"GT: {gt_label}"
    
    # --- 4. 确定文字颜色 ---
    # 绿色 (0, 255, 0) / 红色 (0, 0, 255)
    pred_color = (0, 255, 0) if pred_label == "cross" else (0, 0, 255)
    gt_color = (0, 255, 0) if gt_label == "cross" else (0, 0, 255)
    
    # --- 5. 绘制文字 (在底部黑色区域) ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0 # 字体稍微大一点
    thickness = 2
    
    # 计算 Pred 文字位置 (左下角)
    # y 坐标设置为 h - 20 (离底部留一点空隙)
    cv2.putText(vis_img, pred_text, (20, h - 20), font, font_scale, pred_color, thickness)
    
    # 计算 GT 文字位置 (Pred文字右边，或者固定在图片中间偏右)
    # 这里我们把 GT 放在 Pred 右边 300 像素的位置，或者直接放图片右半部分
    gt_x_pos = 350 # 根据图片宽度调整，也可以用 getTextSize 动态计算
    cv2.putText(vis_img, gt_text, (gt_x_pos, h - 20), font, font_scale, gt_color, thickness)
    
    # --- 6. 保存 ---
    filename = os.path.join(OUTPUT_FOLDER, f"result_{frame_id:04d}.jpg")
    cv2.imwrite(filename, vis_img)
    return filename

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
    raw_bbox_test = tte_seq_test['bbox'] # !!! 保留原始像素坐标用于可视化
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
    model.eval()
    
    transforms = RandomResizeSeq([800], max_size=1333)
    normalize = T.ToTensor()

    # 4. GPU 热身 (Warmup) - 修复版
    print("正在进行 GPU 热身...")
    # 构造符合模型输入维度的 Dummy 数据
    dummy_img = torch.rand(1, 3, 640, 640).to(device)
    dummy_img = padding(dummy_img, len(dummy_img))
    dummy_bbox = torch.rand(1, 15, 4).to(device) # 假设序列长度16, 坐标4
    dummy_vel = torch.rand(1, 15, 1).to(device)  # 假设序列长度16, 速度1
    
    with torch.no_grad():
        for _ in range(10):
            # 必须传入所有参数以匹配 forward 签名
            _ = model(dummy_img, dummy_bbox, dummy_vel, False)

    # 5. 正式测试循环
    print(f"=== 开始测试 (可视化: {'开启' if ENABLE_VISUALIZATION else '关闭'}) ===")
    print(f"{'Frame':<8} | {'Inference (GPU)':<18} | {'Vis+IO (CPU)':<15} | {'Total Flow':<15}")
    print("-" * 65)
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    stats_inference = []
    stats_total = []
    
    # 确定测试长度
    num_samples = len(raw_bbox_test)
    if TEST_FRAMES is not None and TEST_FRAMES < num_samples:
        num_samples = TEST_FRAMES

    for i in range(num_samples): # 修复: 使用 range()
        # --- 数据准备 (不算在推理时间内，但算在流程内需要斟酌，通常IO单独算) ---
        # 模型输入数据
        bbox_input = bbox_tensor_all[i].to(device) # (Seq, 4)
        vel_input = vel_norm(vel_tensor_all[i]).to(device) # (Seq, 1)
        
        # 读取图像
        img_path = img_path_test[i] # 字符串路径
        im_pil = Image.open(img_path).convert('RGB')
        
        # 图像变换
        im_data = transforms(normalize(im_pil))[None].to(device) # (1, 3, H, W)
        im_data = padding(im_data, len(im_data))
        
        # A. === 模型推理 (GPU 计时) ===
        torch.cuda.synchronize()
        starter.record()
        
        with torch.no_grad():
            # 增加维度 [None] 变为 (1, Seq, Dim)
            tag, _, _, _, _, _ = model(im_data, bbox_input[None], vel_input[None], False)
            conf_score = torch.sigmoid(tag) # 得到概率

        ender.record()
        torch.cuda.synchronize()
        inf_time_ms = starter.elapsed_time(ender)
        
        # B. === 可视化与后处理 (CPU 计时) ===
        vis_time_ms = 0.0
        
        if ENABLE_VISUALIZATION:
            t0_vis = time.time()
            
            # 1. 获取预测标签字符串
            pred_conf_val = conf_score.item()
            pred_label_str = "cross" if pred_conf_val > 0.5 else "not cross"
            
            # 2. 获取真实标签字符串 [新增步骤]
            # label_test[i] 是 tensor(0.) 或 tensor(1.)
            raw_gt = label_tensor_all[i].item() 
            gt_label_str = "cross" if raw_gt == 1.0 else "not cross"
            
            # 3. 获取边界框
            raw_bbox_seq = raw_bbox_test[i]
            last_frame_bbox = raw_bbox_seq[-1]
            
            # 4. 准备图像
            vis_img_np = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)
            
            # 5. 调用新的可视化函数 [参数更新]
            visualize_and_save(
                orig_image_np=vis_img_np, 
                bbox=last_frame_bbox, 
                pred_label=pred_label_str,  # 传入预测标签
                gt_label=gt_label_str,      # 传入真实标签
                conf=pred_conf_val,         # 置信度(函数内虽然不画了，但参数可以留着备用)
                frame_id=i
            )
            
            t1_vis = time.time()
            vis_time_ms = (t1_vis - t0_vis) * 1000

        # C. === 统计 ===
        # 这里 Total Time = 模型GPU耗时 + 可视化CPU耗时 
        # (忽略了 dataloader 的 io 时间，专注于算法+后处理延迟)
        total_time_ms = inf_time_ms + vis_time_ms
        
        stats_inference.append(inf_time_ms)
        stats_total.append(total_time_ms)
        
        print(f"{i:<8} | {inf_time_ms:>10.2f} ms | {vis_time_ms:>10.2f} ms | {total_time_ms:>10.2f} ms")

    # 6. 最终报告
    avg_inf = np.mean(stats_inference)
    avg_total = np.mean(stats_total)
    
    fps_inf = 1000 / avg_inf if avg_inf > 0 else 0
    fps_total = 1000 / avg_total if avg_total > 0 else 0
    
    print("\n" + "="*40)
    print("           性能测试总结报告           ")
    print("="*40)
    print(f"测试帧数: {len(stats_inference)}")
    print("-" * 40)
    print(f"1. 纯算法性能 (Inference Only):")
    print(f"   平均耗时: {avg_inf:.2f} ms")
    print(f"   理论 FPS: {fps_inf:.2f} FPS")
    print("-" * 40)
    print(f"2. 部署性能 (含 Vis & Save):")
    print(f"   平均耗时: {avg_total:.2f} ms")
    print(f"   实际 FPS: {fps_total:.2f} FPS")
    print("="*40)

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