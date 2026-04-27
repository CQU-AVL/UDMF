import gc
import pickle
import platform
import re
from typing import Optional
import yaml
from PIL import Image
from utils.jaad_data import JAAD
from utils.pie_data import PIE
# import cv2
import numpy as np
import torch
from torch import Tensor, nn
from torchvision.transforms import transforms as T
from torchvision.transforms import functional
from tqdm import tqdm
import os
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype



class CustomDataset(Dataset):
    def __init__(self, args, label, bbox, vel, image_path, data_augment, mod):
        self.args = args
        self.data_augment = data_augment
        self.mod = mod
        self.device = 'cuda'
        self.cache_dir = os.path.join(f"{args.cache_dir}_{args.bh}", f"{mod}_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        # 初始化数据增强后的数据或原始数据
        self.path = image_path
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 预处理输入数据 (bbox, label, vel)
        label = label.unsqueeze(-1).unsqueeze(-1).repeat(1, 15, 1)
        target = {"bbox": bbox, "label": label, "vel": vel}
        self.item = self._update_data_attributes(target)  # 全数据集 tensor/dict
        self.labels = label[:, 0, 0].tolist()  # [N] scalars

        # # 检查现有 JPG 文件
        # existing_jpgs = [f for f in os.listdir(self.cache_dir) if f.endswith(".jpg")]
        # existing_ids = [int(os.path.splitext(f)[0]) for f in existing_jpgs if os.path.splitext(f)[0].isdigit()]
        # num_existing = len(existing_ids)
        # num_total = len(self.path)
        #
        # print(f"[INFO] Found {num_existing}/{num_total} existing JPGs in {self.cache_dir}")
        #
        # if num_existing == num_total:
        #     print(f"[INFO] All JPGs exist, loading from {self.cache_dir}")
        #     self.cache_data = sorted(
        #         [os.path.join(self.cache_dir, f"{i}.jpg") for i in range(num_total) if
        #          os.path.exists(os.path.join(self.cache_dir, f"{i}.jpg"))],
        #         key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        #     )
        # else:
        #     print(f"[INFO] Preprocessing missing JPGs and saving to {self.cache_dir}")
        #     missing_count = 0
        #     for i in tqdm(range(num_total), desc="Caching missing JPGs"):
        #         jpg_path = os.path.join(self.cache_dir, f"{i}.jpg")
        #         if os.path.exists(jpg_path):
        #             continue  # 跳过已存在
        #
        #         img_path = self.path[i][-1]  # 最后一帧
        #         frame_pil = Image.open(img_path).convert("RGB")
        #
        #         # 保存最后一帧为 JPG (保持格式, quality=100 无损压缩)
        #         frame_pil.save(jpg_path, 'JPEG', quality=100)  # 100 = 无损
        #
        #         missing_count += 1
        #
        #     print(f"[INFO] Saved {missing_count} missing JPGs to {self.cache_dir}")
        #
        #     # 构建完整 cache_data (现有 + 新保存, 排序)
        #     self.cache_data = sorted(
        #         [os.path.join(self.cache_dir, f"{i}.jpg") for i in range(num_total) if
        #          os.path.exists(os.path.join(self.cache_dir, f"{i}.jpg"))],
        #         key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        #     )
        #     if len(self.cache_data) < num_total:
        #         print(f"[WARN] Still missing {num_total - len(self.cache_data)} JPGs after processing!")


    def _update_data_attributes(self, data_dict):
        """更新数据集属性"""
        data_dict['label'] = self.label_transforms(data_dict['label'])
        data_dict['vel'] = self.vel_norm(data_dict['vel'])
        return data_dict

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        jpg_path = self.path[idx]  # JPG 作为索引

        # 加载 frame 从 JPG (PIL + normalize)
        if os.path.exists(jpg_path):
            frame_pil = Image.open(jpg_path).convert("RGB")
            frame = self.normalize(frame_pil)
        else:
            raise ValueError(f"JPG not found for {jpg_path}")

        # 直接从输入数据 (self.item) 取出对应 idx 的 tar (无 npz)
        tar = {k: self.item[k][idx] for k in self.item}  # bbox/label/vel 第 idx 样本

        # 可选数据增强
        if self.data_augment:
            frame, tar = self.data_augment(frame, tar)

        return frame, tar
    def label_transforms(self, y):
        """返回 0/1 浮点标签，与 BCEWithLogitsLoss 对应"""
        return y.float()

    def vel_norm(self, v):
        vmax = torch.max(v)
        if vmax == 0:
            return v
        else:
            v = v / vmax
            return v


def of_collate_fn(batch):
    new_batch = {"image":[], "bbox":[], "label":[], "vel":[]}
    for b in batch:
        new_batch["image"].append(b[0][0])
        new_batch["bbox"].append(b[1][0]["bbox"])
        new_batch["label"].append(b[1][0]["label"])
        new_batch["vel"].append(b[1][0]["vel"])

    new_batch = {k: torch.stack(v, dim=0) if k != "image" else v for k, v in new_batch.items()}
    batch_size = len(new_batch['image'])
    pad_images = padding(new_batch['image'], batch_size)
    new_batch['image'] = pad_images

    return new_batch


def create_balanced_train_loader(trainset, batch_size, of_collate_fn, drop_last, neg_weight=4.0):
    # 获取所有样本的标签
    labels = []
    for sample in trainset:
        label = sample['label']  # [2], one-hot
        label_idx = int(label.item())  # 0 (负类) 或 1 (正类)
        labels.append(label_idx)

    labels = torch.tensor(labels)
    weights = torch.ones(len(labels))
    weights[labels == 0] = neg_weight

    # 新增：计算目标采样数量
    num_pos = (labels == 1).sum().item()
    num_neg = int(num_pos * neg_weight)
    num_samples = num_pos + num_neg

    sampler = WeightedRandomSampler(weights, num_samples, replacement=True)

    # 创建新的 DataLoader，移除 shuffle=True
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=sampler,  # 使用 WeightedRandomSampler
        collate_fn=of_collate_fn,
        drop_last=drop_last  # 根据你的系统调整
    )
    check_label_distribution(train_loader, device=torch.device('cpu'))
    gc.collect()

    return train_loader


def check_label_distribution(dataset_or_loader, device=None):
    """检查数据集或 DataLoader 的正负类分布"""
    pos_count = 0
    neg_count = 0
    total_batches = 0

    if isinstance(dataset_or_loader, DataLoader):
        for batch in dataset_or_loader:
            labels = batch['label']  # [batch_size, 2]
            if device:
                labels = labels.to(device)
            labels = labels.argmax(dim=1)  # [batch_size], 0 (负类) 或 1 (正类)
            pos_count += (labels == 1).sum().item()
            neg_count += (labels == 0).sum().item()
            total_batches += 1
    else:
        for sample in dataset_or_loader:
            label = sample['label'].argmax().item()  # 0 或 1
            if label == 1:
                pos_count += 1
            else:
                neg_count += 1
        total_batches = 1

    total = pos_count + neg_count
    print(f"正类 (类别 1): {pos_count} ({pos_count/total*100:.2f}%)")
    print(f"负类 (类别 0): {neg_count} ({neg_count/total*100:.2f}%)")
    if total_batches > 1:
        print(f"总批次: {total_batches}")
    return pos_count, neg_count

def xyxy_to_cxcywh(bbox):
    xtl = bbox[0]
    ytl = bbox[1]
    xbr = bbox[2]
    ybr = bbox[3]
    b = [(xbr + xtl) / 2, (ybr + ytl) / 2,
         xbr - xtl, ybr - ytl]
    return torch.tensor(b)


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

    return NestedTensor(tensor, mask)


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

def padding_to_patch_size(of_data_batch, batchsize, patch_size=16):
    img_shapes = [list(img.shape) for img in of_data_batch]
    max_size = _max_by_axis(img_shapes)
    batch_shape = [batchsize] + max_size

    max_size[1] = (max_size[1] + patch_size - 1) // patch_size * patch_size
    max_size[2] = (max_size[2] + patch_size - 1) // patch_size * patch_size

    b, c, h, w = batch_shape
    dtype = of_data_batch[0].dtype
    device = of_data_batch[0].device
    tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
    mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
    for img, pad_img in zip(of_data_batch, tensor):  # img pad m 分别取三个list,循环次数为第一维-batchsize
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)  # 将img copy到pad中，占用img的维度

    return tensor



def _max_by_axis(the_list):
    maxes = list(the_list[0])
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class RandomHorizontalFlipSeq(nn.Module):
    def __init__(self, prob=0.7):  # 增 prob 更多变异
        super().__init__()
        self.prob = prob

    def forward(self, images, targets):
        if torch.rand(1).item() < self.prob:
            images = torch.flip(images, dims=[2])  # 水平翻转 W

            boxes = targets["bbox"].clone()  # 假设 'bbox' key
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            new_x1 = 1 - x2
            new_x2 = 1 - x1
            targets["bbox"] = torch.stack([new_x1, y1, new_x2, y2], dim=1).clamp(0, 1)
        return images, targets

class RandomResizeSeq(nn.Module):
    def __init__(self, sizes=(400, 500, 600), max_size=1000):
        super().__init__()
        self.sizes = sizes
        self.max_size = max_size

    def forward(self, images, targets=None):
        _, h, w = images.shape  # [C,H,W]
        size = self.sizes[torch.randint(0, len(self.sizes), (1,)).item()]

        scale = min(size / max(h, w), self.max_size / max(h, w))
        new_h, new_w = int(h * scale), int(w * scale)

        # F.interpolate for [C,H,W] tensor
        images = nn.functional.interpolate(images.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
        if targets is not None:
            return images, targets
        else:
            return images

class AddSpeedNoise(nn.Module):
    def __init__(self, std=0.05, prob=0.7):  # 温和 std, 高 prob
        super().__init__()
        self.std = std
        self.prob = prob

    def forward(self, images, targets):
        speed = targets["vel"]  # [15,1] [0,1]

        if torch.rand(1).item() < self.prob:
            noise = torch.randn_like(speed) * self.std
            targets["vel"] = torch.clamp(speed + noise, min=0.0, max=1.0)
        return images, targets

class AddSpeedAugmentation(nn.Module):
    def __init__(self, noise_std=0.1, scale_prob=0.8): 
        super().__init__()
        self.noise_std = noise_std
        self.scale_prob = scale_prob

    def forward(self, images, targets):
        speed = targets["vel"]  # [15,1] [0,1]
        
        if torch.rand(1).item() < self.scale_prob :
            # 策略：直接加上一个巨大的偏移量
            # 训练集均值是 0.4，我们加 0.5，均值就变成了 0.9，直接覆盖验证集！
            # 偏移量在 0.2 ~ 0.7 之间随机
            offset = torch.rand(1).item() * 0.5 + 0.2 
            speed = speed + offset

        # 记得要做截断，物理上不能超过 1.0
        targets["vel"] = torch.clamp(speed, min=0.0, max=1.0)
        return images, targets
    
class ColorJitterSeq(nn.Module):
    def __init__(self, prob=0.2):
        super().__init__()
        self.prob = prob
        self.jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)

    def forward(self, images, targets):
        if torch.rand(1).item() < self.prob:
            images = self.jitter(images)  # 支持 tensor [C,H,W]
        return images, targets

class VideoAugmentation(nn.Module):
    def __init__(self, is_train=True):
        super().__init__()
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        if is_train:
            self.flip = RandomHorizontalFlipSeq(prob=0.5)
            self.resize = RandomResizeSeq(scales, max_size=1333)
            self.color_jitter = ColorJitterSeq(prob=0.5)  # 新加背景变异
            self.speed_noise = AddSpeedNoise(std=0.05, prob=0.5)
            # self.speed_aug = AddSpeedAugmentation(noise_std=0.1, scale_prob=0.8)
        else:
            self.flip = None
            self.resize = RandomResizeSeq([800], max_size=1333)  # 固定
            self.color_jitter = None
            self.speed_noise = AddSpeedNoise(std=0, prob=0)  # 无噪声
            # self.speed_aug = AddSpeedAugmentation(noise_std=0.0, scale_prob=0.0)

    def forward(self, images, targets):  # images [C,H,W], targets dict{3 tensors}
        tar = targets.copy()  # 浅拷贝防修改原
        if self.flip is not None:
            images, tar = self.flip(images, tar)
        images, tar = self.resize(images, tar)
        if hasattr(self, 'color_jitter') and self.color_jitter is not None:
            images, tar = self.color_jitter(images, tar)
        images, tar = self.speed_noise(images, tar)
        # images, tar = self.speed_aug(images, tar)

        return [images], [tar]  # list for batch

def dictlist_to_listdict(target):
    """
    把 {key: [list]} 转换为 [dict]
    例:
        {"boxes":[b1,b2], "labels":[l1,l2]}
    ->  [{"boxes":b1, "labels":l1}, {"boxes":b2, "labels":l2}]
    """
    keys = list(target.keys())
    T = len(next(iter(target.values())))
    list_dict = []
    for t in range(T):
        d = {k: target[k][t] for k in keys}
        list_dict.append(d)
    return list_dict

def listdict_to_dictlist(list_of_dicts):
    """
    将 list[dict] 转换为 dict[list]
    输入: [ {"k1":v11, "k2":v12}, {"k1":v21, "k2":v22}, ... ]
    输出: {"k1":[v11,v21,...], "k2":[v12,v22,...]}
    """
    if not list_of_dicts:
        return {}

    out_dict = {k: [] for k in list_of_dicts[0].keys()}
    for d in list_of_dicts:
        for k, v in d.items():
            out_dict[k].append(v)
    return out_dict


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
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
