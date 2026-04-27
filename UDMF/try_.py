import cv2
from PIL import Image
import os
from tqdm import tqdm


def extract_specific_frames(video_paths, frame_numbers, output_dir):
    """
    从视频列表提取指定帧为 JPG。
    - video_paths: list of video paths (e.g., ['video1.mp4', 'video2.mp4'])
    - frame_numbers: list of frame indices (same length as video_paths, 0-based)
    - output_dir: 保存目录 (e.g., 'extracted_frames')
    返回: list of saved JPG paths
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    for video_path, frame_num in tqdm(zip(video_paths, frame_numbers), total=len(video_paths),
                                      desc="Extracting frames"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open {video_path}")
            saved_paths.append(None)
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_num >= total_frames or frame_num < 0:
            print(f"[ERROR] Frame {frame_num} invalid for {video_path} (total {total_frames})")
            cap.release()
            saved_paths.append(None)
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame_cv = cap.read()
        cap.release()

        if ret:
            # BGR to RGB, 保存 JPG
            frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            jpg_name = f"{frame_num:05d}_new.jpg"
            jpg_path = os.path.join(output_dir, jpg_name)
            frame_pil.save(jpg_path, 'JPEG', quality=95)  # 高质量
            saved_paths.append(jpg_path)
            print(f"Saved {jpg_path}")
        else:
            print(f"[ERROR] Failed to read frame {frame_num} from {video_path}")
            saved_paths.append(None)

    return saved_paths


# 示例用法
video_paths = ["/mnt/f/Dataset_lr/set03/video_0007.mp4"]  # 替换视频列表
frame_numbers = [7293]  # 替换指定帧号列表 (对应视频)
output_dir = "/mnt/e/PIE_dataset/images/set03/video_0007"  # 保存目录
saved = extract_specific_frames(video_paths, frame_numbers, output_dir)
print(f"Extracted {len([p for p in saved if p is not None])} frames")