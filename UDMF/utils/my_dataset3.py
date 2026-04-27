import gc
import os
import random
from typing import Optional
from torch import Tensor
import numpy as np
from tqdm import tqdm
import torch
from torchvision.transforms import transforms as T
from PIL import Image
from torch.utils.data import Dataset


def preprocess_data(args, bbox, label, vel, bbox_dec, image_path, enable_augment, mod):
    if enable_augment:
        new_paths, flip_indices = preprocess_flips(image_path, args)
        add_data = data_augment(bbox, new_paths, flip_indices, label, vel, bbox_dec)
        bbox, image_path, label, vel, bbox_dec = add_data['bbox'], add_data['path'], add_data['label'], add_data['vel'], \
        add_data['bbox_dec']

    label = label_transforms(label)
    vel = vel_norm(vel)

    preprocessed_data = {
        "img": [], "bbox": [], "window": [], "label": [], "vel": [], "traj": []
    }
    random.seed(42)

    temp_data = {"img": [], "bbox": [], "window": []}
    for idx in tqdm(range(len(image_path)), desc=f"Preprocessing {mod} data"):
        img_paths = image_path[idx]
        transform = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        images = [transform(Image.open(p).convert('RGB')) for p in img_paths]
        images_select = torch.stack(images[::5])
        diff_sum = torch.sum(torch.abs(images_select[:-1] - images_select[1:]), dim=0)

        crop_rate = [1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50]
        select_rate = random.choice(crop_rate)
        _max_bbox = max_bbox(bbox[idx])
        _expand_bbox = expand_bbox(_max_bbox, select_rate)
        crop_img, image_box = crop(diff_sum, _expand_bbox)
        bboxes = adjust_bbox_to_new_scale(bbox[idx], image_box)

        temp_data["img"].append(crop_img)
        temp_data["bbox"].append(bboxes)
        temp_data["window"].append(_expand_bbox)

    num_samples = len(temp_data["img"])
    indices = list(range(num_samples))
    random.seed(42)  # For reproducibility
    random.shuffle(indices)
    for batch_start in range(0, num_samples, args.batch_size):
        batch_end = min(batch_start + args.batch_size, num_samples)
        batch_indices = indices[batch_start:batch_end]

        if len(batch_indices) == args.batch_size:
            batch_imgs = [temp_data["img"][i] for i in batch_indices]
            batch_bboxes = [temp_data["bbox"][i] for i in batch_indices]
            batch_windows = [temp_data["window"][i] for i in batch_indices]

            # 填充图片到统一尺寸
            pad_images, h_w_ = padding(batch_imgs, len(batch_imgs))

            # 调整窗口坐标
            padded_windows = [
                torch.cat([window[:2], window[2:] + torch.flip(pad_hw, dims=[0])])
                for window, pad_hw in zip(batch_windows, h_w_)
            ]

            # 归一化边界框
            padded_bboxes = normalize_bbox(batch_bboxes, pad_images.tensors[0].shape)

            # 保存批次数据
            preprocessed_data["img"].append(pad_images)
            preprocessed_data["bbox"].extend(padded_bboxes)
            preprocessed_data["window"].extend(padded_windows)
            preprocessed_data["label"].extend([label[i] for i in batch_indices])
            preprocessed_data["vel"].extend([vel[i] for i in batch_indices])
            preprocessed_data["traj"].extend([bbox_dec[i] for i in batch_indices])

    # 保存到文件
    cache_path = f"{args.cache_dir}/{mod}_{'aug' if enable_augment else 'noaug'}_preprocessed.pt"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(preprocessed_data, cache_path)
    print(f"Preprocessed data with padding saved to {cache_path}")

    # 清理内存
    del temp_data
    gc.collect()


def normalize_bbox(bbox_list, shape):
    c, h, w = shape
    normalized_bboxes = []
    for bbox in bbox_list:
        bbox_norm = bbox.clone().to(dtype=torch.float32)  # 确保数据类型为浮点型
        # 归一化x坐标（x1和x2）
        bbox_norm[:, [0, 2]] /= w
        # 归一化y坐标（y1和y2）
        bbox_norm[:, [1, 3]] /= h

        bbox_xywh = []
        for bbox in bbox_norm:
            bbox_xywh.append(xyxy_to_cxcywh(bbox))

        normalized_bboxes.append(torch.stack(bbox_xywh, 0))
    return torch.stack(normalized_bboxes, 0)


def xyxy_to_cxcywh(bbox):
    xtl = bbox[0]
    ytl = bbox[1]
    xbr = bbox[2]
    ybr = bbox[3]
    b = [(xbr + xtl) / 2, (ybr + ytl) / 2,
         xbr - xtl, ybr - ytl]
    return torch.tensor(b)


def padding(of_data_batch, batchsize):
    img_shapes = [list(img.shape) for img in of_data_batch]
    max_size = _max_by_axis(img_shapes)
    batch_shape = [batchsize] + max_size
    b, c, h, w = batch_shape
    dtype = of_data_batch[0].dtype
    device = of_data_batch[0].device
    tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
    mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
    for img, pad_img, m in zip(of_data_batch, tensor, mask):  # img pad m 分别取三个list,循环次数为第一维-batchsize
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)  # 将img copy到pad中，占用img的维度
        m[: img.shape[1], :img.shape[2]] = False  # 在m中，将img占用的维度置为false

    hw_m = torch.tensor([h, w])
    h_w_ = [hw_m - torch.tensor([hw[1], hw[2]]) for hw in img_shapes]
    return NestedTensor(tensor, mask), h_w_


def _max_by_axis(the_list):
    maxes = list(the_list[0])
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def preprocess_flips(image_paths, args):
    random.seed(42)
    flip_number = sorted(random.sample(range(len(image_paths)), int(len(image_paths) * 0.5)))
    num = (len(flip_number) + len(image_paths)) % args.batch_size

    for i in range(-1, -num - 1, -1):
        flip_number.pop(i)

    for i in tqdm(flip_number, desc="Processing flipped images"):
        path_bath = []
        for path in image_paths[i]:
            new_img_path = path.replace('.png', '_flip.png')
            if not os.path.exists(new_img_path):
                try:
                    # 使用 with 语句确保文件正确关闭
                    with Image.open(path) as img:
                        flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                        flipped_img.save(new_img_path)
                except Exception as e:
                    tqdm.write(f"Error processing {path}: {str(e)}")
                    continue
            path_bath.append(new_img_path)
        image_paths.append(path_bath)

    return image_paths, flip_number


def label_transforms(y):
    y_mat = torch.zeros(len(y), 2, dtype=y.dtype)
    y_mat[y == 1, 1] = 1
    y_mat[y == 0, 0] = 1
    y = y_mat
    return y


def vel_norm(v):
    vmax = torch.max(v)
    if vmax == 0:
        return v
    else:
        v = v / vmax
        return v


def max_bbox(bbox_list):
    """使用 NumPy 快速计算最大边界框"""
    bbox_array = np.array(bbox_list)
    xtl_m = int(np.min(bbox_array[:, 0]))
    ytl_m = int(np.min(bbox_array[:, 1]))
    xbr_m = int(np.max(bbox_array[:, 2]))
    ybr_m = int(np.max(bbox_array[:, 3]))
    bbox = [xtl_m, ytl_m, xbr_m, ybr_m]
    return torch.tensor(bbox)


def expand_bbox(bbox, scale):
    xtl = bbox[0]
    ytl = bbox[1]
    xbr = bbox[2]
    ybr = bbox[3]
    cx = (xtl + xbr) / 2
    cy = (ytl + ybr) / 2

    # 计算原始宽度和高度
    w = xbr - xtl
    h = ybr - ytl

    # 计算新的宽度和高度
    new_w = w * scale
    new_h = h * scale

    # 计算新的坐标
    new_xtl = cx - new_w / 2
    new_ytl = cy - new_h / 2
    new_xbr = cx + new_w / 2
    new_ybr = cy + new_h / 2

    return torch.stack([new_xtl, new_ytl, new_xbr, new_ybr], 0)


def flip_bbox(bbox, image_width):
    x_min, y_min, x_max, y_max = bbox
    new_x_min = image_width - x_max
    new_x_max = image_width - x_min
    bbox_flip = torch.tensor([new_x_min, y_min, new_x_max, y_max])
    return bbox_flip


def data_augment(bboxes, img_pathes, flip_number, labels, vels, bbox_decs):
    batch = {'path': img_pathes, 'bbox': bboxes, 'label': labels, 'vel': vels, 'bbox_dec': bbox_decs}
    W = 1920
    for i in flip_number:
        flipped_bboxes = []
        flipped_vel = []
        flipped_bbox_dec = []
        for bbox, vel in zip(bboxes[i], vels[i]):
            bbox_flip = flip_bbox(bbox, W)
            flipped_bboxes.append(bbox_flip)
            flipped_vel.append(vel)

        for bbox_dec in bbox_decs[i]:
            bbox_dec_flip = flip_bbox(bbox_dec, W)
            flipped_bbox_dec.append(bbox_dec_flip)

        flipped_bboxes = torch.stack(flipped_bboxes, 0).unsqueeze(0)
        flipped_vel = torch.stack(flipped_vel, 0).unsqueeze(0)
        flipped_bbox_dec = torch.stack(flipped_bbox_dec, 0).unsqueeze(0)

        batch['bbox'] = torch.cat([batch['bbox'], flipped_bboxes], 0)
        batch['label'] = torch.cat([batch['label'], labels[i].unsqueeze(0)], 0)
        batch['vel'] = torch.cat([batch['vel'], flipped_vel], 0)
        batch['bbox_dec'] = torch.cat([batch['bbox_dec'], flipped_bbox_dec], 0)

    return batch


def crop(image, bbox):
    # image:[C H W]
    if isinstance(image, Image.Image):
        image = np.array(image)  # PIL 图像 -> NumPy 数组
    elif isinstance(image, torch.Tensor):
        image = image.numpy()  # PyTorch 张量 -> NumPy 数组

    # 获取图像尺寸
    H, W = image.shape[1:]  # 高度和宽度

    # 解析 BBox
    xtl, ytl, xbr, ybr = bbox

    # 处理裁剪区域超出图像边界的情况
    xtl = int(max(0, xtl))
    ytl = int(max(0, ytl))
    xbr = int(min(W, xbr))
    ybr = int(min(H, ybr))

    # 裁剪图像
    if len(image.shape) == 3:  # RGB 图像 (C, H ,W)
        crop_img = torch.tensor(image[:, ytl:ybr, xtl:xbr])
    else:  # 灰度图像 (H, W)
        crop_img = image[ytl:ybr, xtl:xbr]

    image_box = [xtl, ytl, xbr, ybr]
    return crop_img, image_box


def adjust_bbox_to_new_scale(bbox_list, image_box):
    bboxes = []
    xtl = image_box[0]
    ytl = image_box[1]
    xbr = image_box[2]
    ybr = image_box[3]
    width_crop = xbr - xtl
    height_crop = ybr - ytl
    for bbox in bbox_list:
        x_min, y_min, x_max, y_max = bbox
        new_x_min = max(x_min - xtl, 0)
        new_y_min = max(y_min - ytl, 0)
        new_x_max = min(x_max - xtl, width_crop)
        new_y_max = min(y_max - ytl, height_crop)
        bboxes.append(torch.tensor([new_x_min, new_y_min, new_x_max, new_y_max]))
    return torch.stack(bboxes, 0)


class CustomDataset(Dataset):
    def __init__(self, args, bbox_train, label_train, vel_train, bbox_train_dec, img_path_train, enable_augment, mod):
        self.args = args
        self.mod = mod
        self.enable_augment = enable_augment
        self.cache_path = f"{args.cache_dir}/{mod}_{'aug' if enable_augment else 'noaug'}_preprocessed.pt"
        try:
            self.preprocessed_data = torch.load(self.cache_path, map_location="cpu", mmap=True)
            print(f"缓存文件已加载: {self.cache_path}")
        except:
            preprocess_data(args, bbox_train, label_train, vel_train, bbox_train_dec, img_path_train, enable_augment, mod)
            self.preprocessed_data = torch.load(self.cache_path, map_location="cpu", mmap=True)
            print(f"缓存文件已加载: {self.cache_path}")

    def __len__(self):
        return len(self.preprocessed_data["img"])

    def __getitem__(self, idx):
        img_data = self.preprocessed_data["img"][idx]  # 类型为NestedTensor

        # 计算起始索引并循环获取8个bbox
        start = idx * self.args.batch_size
        indices = [start + i for i in range(self.args.batch_size)]
        bbox = [self.preprocessed_data["bbox"][i] for i in indices]
        label = [self.preprocessed_data["label"][i] for i in indices]
        vel = [self.preprocessed_data["vel"][i] for i in indices]
        traj = [self.preprocessed_data["traj"][i] for i in indices]
        window = [self.preprocessed_data["window"][i] for i in indices]

        # 将bbox列表堆叠为(8, 32, 4)的张量
        return {
            "img": img_data,
            "bbox": torch.stack(bbox, dim=0),
            "label": torch.stack(label, dim=0),
            "vel": torch.stack(vel, dim=0),
            "traj": torch.stack(traj, dim=0),
            "window": torch.stack(window, dim=0)
                }


def custom_collate(batch):
    # 提取所有样本的NestedTensor数据
    img_list = [item['img'] for item in batch]

    # 分解NestedTensor的tensors和mask
    tensors = [img.tensors.squeeze(0) for img in img_list]  # 假设原b=1，去除单样本的batch维度
    masks = [img.mask.squeeze(0) if img.mask is not None else None for img in img_list]

    # 堆叠tensors和mask
    batch_tensors = torch.stack(tensors, dim=0)  # 形状变为 (batch_size, c, h, w)
    if masks[0] is not None:
        batch_masks = torch.stack(masks, dim=0)  # 形状变为 (batch_size, h, w)
    else:
        batch_masks = None

    # 重新封装为NestedTensor
    batch_img = NestedTensor(batch_tensors, batch_masks)

    # 处理其他数据（bbox, label等）
    bbox = torch.stack([item['bbox'] for item in batch], dim=0)
    label = torch.stack([item['label'] for item in batch], dim=0)
    vel = torch.stack([item['vel'] for item in batch], dim=0)
    traj = torch.stack([item['traj'] for item in batch], dim=0)
    window = torch.stack([item['window'] for item in batch], dim=0)

    return {
        'img': batch_img,
        'bbox': bbox,
        'label': label,
        'vel': vel,
        'traj': traj,
        'window': window
    }


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)