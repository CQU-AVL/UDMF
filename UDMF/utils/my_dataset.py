import json
import pickle
import shutil
from os.path import exists, getsize
from typing import Optional
import yaml

from utils.jaad_data import JAAD
from utils.pie_data import PIE
import cv2
import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import transforms
from tqdm import tqdm
import os
from torch.utils.data import Dataset
import random
import torch.nn.functional as F


def DataGenerator(data, labels, data_sizes, process, global_pooling, input_type_list, batch_size, shuffle, to_fit):
    pass


def get_pose(param, param1, data_type, file_path, dataset):
    pass


def get_path(param, param1, data_type, file_path, dataset):
    pass

class OpticalFlowProcessor:
    def __init__(self, paths, bboxes, transform=None, exam=False):
        """
        初始化光流处理器
        Args:
            paths: 图像路径列表
            bboxes: 对应的边界框列表
            transform: 可选的图像变换
        """
        self.paths = paths  # 图像路径列表
        self.bboxes = bboxes  # 边界框列表
        self.transform = transform  # 可选变换
        if exam:
            for batch_path in self.paths:
                self.pre_exam_dataset(batch_path)

    @staticmethod
    def flow_to_motion_image(flow, img_shape):
        """将光流转换为运动图像"""
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros(img_shape, dtype=np.uint8)
        hsv[..., 0] = angle * 180 / np.pi / 2  # 方向
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # 幅度
        motion_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return motion_image

    @staticmethod
    def compute_optical_flow(imgs_np):
        """高效计算光流并累加"""
        prev_gray = cv2.cvtColor(imgs_np[0], cv2.COLOR_BGR2GRAY)
        total_flow = None

        for i in range(1, len(imgs_np)):
            curr_gray = cv2.cvtColor(imgs_np[i], cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            flow = cv2.GaussianBlur(flow, (5, 5), 0)

            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            threshold = 0
            mask = magnitude > threshold

            # 仅保留大于阈值的光流向量
            filtered_flow = np.zeros_like(flow)
            filtered_flow[..., 0] = flow[..., 0] * mask
            filtered_flow[..., 1] = flow[..., 1] * mask

            if total_flow is None:
                total_flow = filtered_flow
            else:
                total_flow += filtered_flow

            prev_gray = curr_gray

        return total_flow

    @staticmethod
    def max_bbox(bbox_list):
        """使用 NumPy 快速计算最大边界框"""
        bbox_array = np.array(bbox_list)
        xtl_m = int(np.min(bbox_array[:, 0]))
        ytl_m = int(np.max(bbox_array[:, 1]))
        xbr_m = int(np.max(bbox_array[:, 2]))
        ybr_m = int(np.min(bbox_array[:, 3]))
        return xtl_m, ytl_m, xbr_m, ybr_m

    def crop(self, image, bbox_list):
        """根据边界框裁剪图像"""
        xtl_m, ytl_m, xbr_m, ybr_m = self.max_bbox(bbox_list)
        return image[ytl_m:ybr_m, xtl_m:xbr_m]

    def process(self, of_file, mode: str, args):
        """处理指定索引的图像序列并返回裁剪后的运动图像"""
        of_file_path = os.path.join(of_file, f"optical_flow_{args.bh}_{args.times_num}_{args.balance_data}")
        of_data_path = os.path.join(of_file_path, f"optical_flow_{mode}_{len(self.paths)}.pkl")
        temp_dir = os.path.join(of_file_path, "temp_of")
        checkpoint = os.path.join(of_file_path, "processing_checkpoint.json")
        try:
            with open(of_data_path, "rb") as f:
                flow_list = pickle.load(f)
                assert len(flow_list) == len(self.paths)
        except:
            # for bath_path, batch_bbox in tqdm(zip(self.paths, self.bboxes)):
            #     imgs_np = [cv2.imread(path) for path in bath_path]
            #
            #     # 计算光流
            #     total_flow = self.compute_optical_flow(imgs_np)
            #
            #     # 生成运动图像
            #     motion_image = self.flow_to_motion_image(total_flow, imgs_np[0].shape)
            #
            #     # 裁剪图像
            #     image_crop_hw = self.crop(motion_image, batch_bbox)
            #     image_crop_hw = image_crop_hw.astype(np.float32) / 255.0
            #     image_tensor = torch.from_numpy(image_crop_hw).permute(2, 0, 1)
            #     normalize = transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])  # 这里的均值和标准差可以根据需求调整
            #     image_crop_hw = normalize(image_tensor)
            #     if self.transform:
            #         image_crop_hw = self.transform(image_crop_hw)
            #
            #     flow_list.append(image_crop_hw)

            os.makedirs(temp_dir, exist_ok=True)
            # 加载检查点
            try:
                with open(checkpoint, "r") as f:
                    checkpoint = json.load(f)
                processed_ids = set(checkpoint["processed"])
                flow_list = checkpoint["flow_list"]
            except:
                processed_ids = set()
                flow_list = []
                # 初始化最终输出文件
                open(of_data_path, 'w').close()
            # 获取需要处理的总批次
            total = len(self.paths)
            try:
                # 使用 enumerate 获取批次索引
                for batch_idx, (bath_path, batch_bbox) in \
                        enumerate(tqdm(zip(self.paths, self.bboxes), total=total, initial=len(processed_ids))):

                    if str(batch_idx) in processed_ids:
                        continue

                    imgs_np = [cv2.imread(path) for path in bath_path]
                    total_flow = self.compute_optical_flow(imgs_np)
                    motion_image = self.flow_to_motion_image(total_flow, imgs_np[0].shape)
                    image_crop_hw = self.crop(motion_image, batch_bbox)
                    image_crop_hw = image_crop_hw.astype(np.float32) / 255.0
                    image_tensor = torch.from_numpy(image_crop_hw).permute(2, 0, 1)
                    normalize = transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
                    image_crop_hw = normalize(image_tensor)
                    if self.transform:
                        image_crop_hw = self.transform(image_crop_hw)

                    # 立即保存单个结果到临时文件
                    temp_path = os.path.join(temp_dir, f"flow_{batch_idx}.pkl")
                    with open(temp_path, "wb") as f:
                        pickle.dump(image_crop_hw, f)

                    # 更新检查点
                    processed_ids.add(str(batch_idx))
                    with open(checkpoint, "w") as f:
                        json.dump({
                            "processed": list(processed_ids),
                            "flow_list": len(flow_list) + 1  # 只记录数量
                        }, f)

            finally:
                # 最终合并所有临时文件
                flow_list = []
                for batch_idx in range(total):
                    temp_path = os.path.join(temp_dir, f"flow_{batch_idx}.pkl")
                    if os.path.exists(temp_path):
                        with open(temp_path, "rb") as f:
                            flow_list.append(pickle.load(f))

                # 保存最终结果
                with open(of_data_path, "wb") as f:
                    pickle.dump(flow_list, f)

                # 清理临时文件
                shutil.rmtree(temp_dir)
                os.remove(checkpoint)

        return flow_list

    def pre_exam_dataset(self, batch_path):
        for path in tqdm(batch_path,
                         desc="检查文件中",
                         unit="file",
                         colour="green",
                         ncols=80):  # 进度条样式参数
            try:
                # 检查文件存在性
                if not os.path.exists(path):
                    raise FileNotFoundError(f"文件不存在: {path}")

                # 检查文件大小（空文件检测）
                if os.path.getsize(path) == 0:
                    tqdm.write(f"WARNING: 空文件 {path}")  # 使用tqdm.write避免破坏进度条

                # 尝试读取验证
                img = cv2.imread(path)
                if img is None:
                    tqdm.write(f"损坏文件: {path}")

            except Exception as e:
                tqdm.write(f"处理 {path} 时发生错误: {str(e)}")
                # 这里可以添加更详细的错误处理逻辑


class CustomDataset(Dataset):
    def __init__(self, of_data_train, X_train, Y_train, vel_train, X_train_dec):
        """
        Args:
            data (list): 包含不等维度的张量的列表
        """
        self.X_train = self.xyxy_to_cxcywh(X_train)
        self.Y_train = self.label_transforms(Y_train)
        self.vel_train = self.vel_norm(vel_train)
        self.X_train_dec = X_train_dec  # xtl, ytl, xbr, ybr
        self.of_data_train = of_data_train

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        # 返回指定索引的数据
        return {
            "img": self.of_data_train[idx],
            "bbox": self.X_train[idx],
            "label": self.Y_train[idx],
            "vel": self.vel_train[idx],
            "traj": self.X_train_dec[idx]
        }

    def label_transforms(self, y):
        y_mat = torch.zeros(len(y), 2, dtype=y.dtype)
        y_mat[y == 1, 1] = 1
        y_mat[y == 0, 0] = 1
        y = y_mat
        return y

    def vel_norm(self, v):
        vmax = torch.max(v)
        if vmax == 0:
            return v
        else:
            v = v / vmax
            return v

    def xyxy_to_cxcywh(self, bbox):
        xtl = bbox[:, :, 0]
        ytl = bbox[:, :, 1]
        xbr = bbox[:, :, 2]
        ybr = bbox[:, :, 3]
        b = [(xbr + xtl) / 2, (ybr + ytl) / 2,
             xbr - xtl, ybr - ytl]

        return torch.stack(b, dim=-1)


def of_collate_fn(batch):
    # 获取字典中的所有键
    keys = batch[0].keys()
    collated_batch = {key: [] for key in keys}

    # 将批量数据按照键分组
    for sample in batch:
        for key in keys:
            collated_batch[key].append(sample[key])

    # 对光流数据进行特殊处理
    if "img" in collated_batch:
        of_data_batch = collated_batch["img"]
        batchsize = len(of_data_batch)
        # random crop or pad
        # of_tensor_train = random_crop_pad(of_data_batch, batchsize)

        # pad to max size
        of_tensor_train = padding(of_data_batch, batchsize)

        collated_batch["img"] = of_tensor_train

    for key in keys:
        if key != "img":  # 排除光流数据
            collated_batch[key] = torch.stack(collated_batch[key])

    return collated_batch


def random_crop_pad(of_data_batch, batchsize):
    of_tensor_train = []
    masks = []
    dtype = of_data_batch[0].dtype
    device = of_data_batch[0].device
    random.seed(42)
    idx = random.randint(0, batchsize - 1)
    c, target_h, target_w = of_data_batch[idx].shape
    for i in range(batchsize):
        if i == idx:
            of_tensor_train.append(of_data_batch[i])
            mask = torch.zeros_like(of_data_batch[i], dtype=dtype, device=device).to(torch.bool)
            masks.append(mask)
            continue
        else:
            of_data = F.interpolate(of_data_batch[i].unsqueeze(0), (target_h, target_w), mode="bilinear"
                                    , align_corners=False).squeeze(0)
            of_tensor_train.append(of_data)

    of_tensor_train = torch.stack(of_tensor_train, dim=0)
    return of_tensor_train


def padding(of_data_batch, batchsize):
    max_size = _max_by_axis([list(img.shape) for img in of_data_batch])
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


def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


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

class prepare_data(object):

    def __init__(self, args, cache=False):
        """
        Initializes the data preparation class
        Args:
            dataset: Name of the dataset
            opts: Options for preparing data
        """

        self._global_pooling = None
        self._dataset = "JAAD"
        self._cache = cache
        self._generator = None
        configs_default = "D:/pycharm files/learn/TrEP-main/PedestrianActionBenchmark/config_files/configs_default.yaml"
        with open(configs_default, 'r') as f:
            configs = yaml.safe_load(f)
        tte = configs['model_opts']['time_to_event'] if isinstance(configs['model_opts']['time_to_event'], int) else \
            configs['model_opts']['time_to_event'][1]
        configs['model_opts']['obs_length'] = args.times_num + 1
        configs['model_opts']['obs_input_type'] = ['box', 'speed', 'image']
        configs['model_opts']['dataset'] = self._dataset.lower()
        configs['model_opts']['generator'] = False
        configs['model_opts']['overlap'] = 0.5001

        configs['data_opts']['sample_type'] = args.bh
        configs['data_opts']['min_track_size'] = 15
        # configs['data_opts']['min_track_size'] = configs['model_opts']['obs_length'] + tte
        configs['data_opts']['seq_type'] = 'crossing'
        self.configs = configs
        try:
            if self._dataset == 'JAAD':
                with open("D:/pycharm files/learn/TrEP-main/data/JAAD_data/beh_seq_train", 'rb') as f:
                    self.beh_seq_train = pickle.load(f)
                with open("D:/pycharm files/learn/TrEP-main/data/JAAD_data/beh_seq_val", 'rb') as f:
                    self.beh_seq_val = pickle.load(f)
                with open("D:/pycharm files/learn/TrEP-main/data/JAAD_data/beh_seq_test", 'rb') as f:
                    self.beh_seq_test = pickle.load(f)
            elif self._dataset == 'PIE':
                with open("D:/pycharm files/learn/TrEP-main/data/PIE_data/beh_seq_train.pkl", 'rb') as f:
                    self.beh_seq_train = pickle.load(f)
                with open("D:/pycharm files/learn/TrEP-main/data/PIE_data/beh_seq_val.pkl", 'rb') as f:
                    self.beh_seq_val = pickle.load(f)
                with open("D:/pycharm files/learn/TrEP-main/data/PIE_data/beh_seq_test.pkl", 'rb') as f:
                    self.beh_seq_test = pickle.load(f)
        except:
            if self._dataset == 'JAAD':
                data_path = "D:/datasets/JAAD"
                self._data_raw = JAAD(data_path=data_path)
                imdb = JAAD(data_path=data_path)
            elif self._dataset == 'PIE':
                data_path = "PIE"
                self._data_raw = PIE(data_path=data_path)
                imdb = PIE(data_path=data_path)

            self.beh_seq_train = imdb.generate_data_trajectory_sequence('train', **configs['data_opts'])
            self.beh_seq_val = imdb.generate_data_trajectory_sequence('val', **configs['data_opts'])
            self.beh_seq_test = imdb.generate_data_trajectory_sequence('test', **configs['data_opts'])

            if cache:
                if not os.path.exists('data'):
                    os.makedirs('data')

                if self._dataset == 'JAAD':
                    if not os.path.exists('data/JAAD_data'):
                        os.makedirs('data/JAAD_data')
                    with open("data/JAAD_data/beh_seq_train.pkl", 'wb') as f:
                        pickle.dump(self.beh_seq_train, f)
                    with open("data/JAAD_data/beh_seq_val.pkl", 'wb') as f:
                        pickle.dump(self.beh_seq_val, f)
                    with open("data/JAAD_data/beh_seq_test.pkl", 'wb') as f:
                        pickle.dump(self.beh_seq_test, f)
                elif self._dataset == 'PIE':
                    if not os.path.exists('data/PIE_data'):
                        os.makedirs('data/PIE_data')
                    with open("data/PIE_data/beh_seq_train.pkl", 'wb') as f:
                        pickle.dump(self.beh_seq_train, f)
                    with open("data/PIE_data/beh_seq_val.pkl", 'wb') as f:
                        pickle.dump(self.beh_seq_val, f)
                    with open("data/PIE_data/beh_seq_test.pkl", 'wb') as f:
                        pickle.dump(self.beh_seq_test, f)
        self.train_data = self.get_data('train', self.beh_seq_train, self.configs['model_opts'])
        self.val_data = self.get_data('val', self.beh_seq_val, self.configs['model_opts'])
        self.test_data = self.get_data('test', self.beh_seq_test, self.configs['model_opts'])

    def get_data_sequence(self, data_type, data_raw, opts):
        """
            Generates raw sequences from a given dataset
            Args:
                data_type: Split type of data, whether it is train, test or val
                data_raw: Raw tracks from the dataset
                opts:  Options for generating data samples
            Returns:
                A list of data samples extracted from raw data
                Positive and negative data counts
            """
        print('\n#####################################')
        print('Generating raw data')
        print('#####################################')
        d = {'center': data_raw['center'].copy(),
             'box': data_raw['bbox'].copy(),
             'ped_id': data_raw['pid'].copy(),
             'crossing': data_raw['activities'].copy(),
             'image': data_raw['image'].copy()}

        balance = opts['balance_data'] if data_type == 'train' else False
        obs_length = opts['obs_length']
        time_to_event = opts['time_to_event']
        normalize = opts['normalize_boxes']

        try:
            d['speed'] = data_raw['obd_speed'].copy()
        except KeyError:
            d['speed'] = data_raw['vehicle_act'].copy()
            print('Jaad dataset does not have speed information')
            print('Vehicle actions are used instead')
        if balance:
            self.balance_data_samples(d, data_raw['image_dimension'][0])
        d['box_org'] = d['box'].copy()
        d['tte'] = []

        if isinstance(time_to_event, int):
            for k in d.keys():
                for i in range(len(d[k])):
                    d[k][i] = d[k][i][- obs_length - time_to_event:-time_to_event]
            d['tte'] = [[time_to_event]] * len(data_raw['bbox'])
        else:
            overlap = opts['overlap']  # if data_type == 'train' else 0.0
            olap_res = obs_length if overlap == 0 else int((1 - overlap) * obs_length)
            olap_res = 1 if olap_res < 1 else olap_res
            for k in d.keys():
                seqs = []
                for seq in d[k]:
                    start_idx = max(0, len(seq) - obs_length - time_to_event[1])
                    end_idx = len(seq) - obs_length - time_to_event[0]
                    if end_idx < start_idx:
                        continue
                    seqs.extend([seq[i:i + obs_length] for i in
                                 range(start_idx, end_idx + 1, olap_res)
                                 if (i > 0) and (i + obs_length <= len(seq))])
                d[k] = seqs

            for seq in data_raw['bbox']:
                start_idx = len(seq) - obs_length - time_to_event[1]
                end_idx = len(seq) - obs_length - time_to_event[0]
                d['tte'].extend([[len(seq) - (i + obs_length)] for i in
                                 range(start_idx, end_idx + 1, olap_res)])
        if normalize:
            for k in d.keys():
                if k != 'tte':
                    if k != 'box' and k != 'center':
                        for i in range(len(d[k])):
                            d[k][i] = d[k][i][1:]
                    else:
                        for i in range(len(d[k])):
                            d[k][i] = np.subtract(d[k][i][1:], d[k][i][0]).tolist()
                d[k] = np.array(d[k])
        else:
            for k in d.keys():
                d[k] = np.array(d[k])

        d['crossing'] = np.array(d['crossing'])[:, 0, :]
        pos_count = np.count_nonzero(d['crossing'])
        neg_count = len(d['crossing']) - pos_count
        print("Negative {} and positive {} sample counts".format(neg_count, pos_count))

        return d, neg_count, pos_count

    def balance_data_samples(self, d, img_width, balance_tag='crossing'):
        """
        Balances the ratio of positive and negative data samples. The less represented
        data type is augmented by flipping the sequences
        Args:
            d: Sequence of data samples
            img_width: Width of the images
            balance_tag: The tag to balance the data based on
        """
        print("Balancing with respect to {} tag".format(balance_tag))
        gt_labels = [gt[0] for gt in d[balance_tag]]
        num_pos_samples = np.count_nonzero(np.array(gt_labels))
        num_neg_samples = len(gt_labels) - num_pos_samples

        # finds the indices of the samples with larger quantity
        if num_neg_samples == num_pos_samples:
            print('Positive and negative samples are already balanced')
        else:
            print('Unbalanced: \t Positive: {} \t Negative: {}'.format(num_pos_samples, num_neg_samples))
            if num_neg_samples > num_pos_samples:
                gt_augment = 1
            else:
                gt_augment = 0

            num_samples = len(d[balance_tag])
            for i in range(num_samples):
                if d[balance_tag][i][0][0] == gt_augment:
                    for k in d:
                        if k == 'center':
                            flipped = d[k][i].copy()
                            flipped = [[img_width - c[0], c[1]]
                                       for c in flipped]
                            d[k].append(flipped)
                        if k == 'box':
                            flipped = d[k][i].copy()
                            flipped = [np.array([img_width - b[2], b[1], img_width - b[0], b[3]])
                                       for b in flipped]
                            d[k].append(flipped)
                        if k == 'image':
                            flipped = d[k][i].copy()
                            flipped = [im.replace('.png', '_flip.png') for im in flipped]
                            d[k].append(flipped)
                        if k in ['speed', 'ped_id', 'crossing', 'walking', 'looking']:
                            d[k].append(d[k][i].copy())

            gt_labels = [gt[0] for gt in d[balance_tag]]
            num_pos_samples = np.count_nonzero(np.array(gt_labels))
            num_neg_samples = len(gt_labels) - num_pos_samples
            if num_neg_samples > num_pos_samples:
                rm_index = np.where(np.array(gt_labels) == 0)[0]
            else:
                rm_index = np.where(np.array(gt_labels) == 1)[0]

            # Calculate the difference of sample counts
            dif_samples = abs(num_neg_samples - num_pos_samples)
            # shuffle the indices
            np.random.seed(42)
            np.random.shuffle(rm_index)
            # reduce the number of indices to the difference
            rm_index = rm_index[0:dif_samples]

            # update the data
            for k in d:
                seq_data_k = d[k]
                d[k] = [seq_data_k[i] for i in range(0, len(seq_data_k)) if i not in rm_index]

            new_gt_labels = [gt[0] for gt in d[balance_tag]]
            num_pos_samples = np.count_nonzero(np.array(new_gt_labels))
            print('Balanced:\t Positive: %d  \t Negative: %d\n'
                  % (num_pos_samples, len(d[balance_tag]) - num_pos_samples))

    def get_data(self, data_type, data_raw, model_opts, cache=True):
        """
        Generates data train/test/val data
        Args:
            data_type: Split type of data, whether it is train, test or val
            data_raw: Raw tracks from the dataset
            model_opts: Model options for generating data
        Returns:
            A dictionary containing, data, data parameters used for model generation,
            effective dimension of data (the number of rgb images to be used calculated accorfing
            to the length of optical flow window) and negative and positive sample counts
        """
        try:
            if self._dataset == 'JAAD':
                with open("D:/pycharm files/learn/TrEP-main/data/JAAD_data/JAAD_{}".format(data_type), 'rb') as f:
                    dd = pickle.load(f)
            elif self._dataset == 'PIE':
                with open("D:/pycharm files/learn/TrEP-main/data/PIE_data/PIE_{}".format(data_type), 'rb') as f:
                    dd = pickle.load(f)
        except:

            self._generator = model_opts.get('generator', False)
            data_type_sizes_dict = {}
            process = model_opts.get('process', True)
            dataset = model_opts['dataset']
            data, neg_count, pos_count = self.get_data_sequence(data_type, data_raw, model_opts)

            data_type_sizes_dict['box'] = data['box'].shape[1:]
            if 'speed' in data.keys():
                data_type_sizes_dict['speed'] = data['speed'].shape[1:]

            # Store the type and size of each image
            _data = []
            data_sizes = []
            data_types = []

            for d_type in model_opts['obs_input_type']:
                if 'local' in d_type or 'context' in d_type:
                    features, feat_shape = self.get_context_data(model_opts, data, data_type, d_type)
                elif 'pose' in d_type:
                    path_to_pose, _ = get_path(save_folder='poses',
                                               dataset=dataset,
                                               save_root_folder='data/features')
                    features = get_pose(data['image'],
                                        data['ped_id'],
                                        data_type=data_type,
                                        file_path=path_to_pose,
                                        dataset=model_opts['dataset'])
                    feat_shape = features.shape[1:]
                else:
                    features = data[d_type]
                    feat_shape = features.shape[1:]
                _data.append(features)
                data_sizes.append(feat_shape)
                data_types.append(d_type)

            # create the final data file to be returned
            if self._generator:
                _data = (DataGenerator(data=_data,
                                       labels=data['crossing'],
                                       data_sizes=data_sizes,
                                       process=process,
                                       global_pooling=self._global_pooling,
                                       input_type_list=model_opts['obs_input_type'],
                                       batch_size=model_opts['batch_size'],
                                       shuffle=data_type != 'test',
                                       to_fit=data_type != 'test'), data['crossing'])  # set y to None
            else:
                _data = (_data, data['crossing'])
            dd = {'data': _data,
                  'ped_id': data['ped_id'],
                  'image': data['image'],
                  'tte': data['tte'],
                  'data_params': {'data_types': data_types, 'data_sizes': data_sizes},
                  'count': {'neg_count': neg_count, 'pos_count': pos_count}}
            if cache:
                if not os.path.exists('data'):
                    os.makedirs('data')
                if self._dataset == 'JAAD':
                    if not os.path.exists('data/JAAD_data'):
                        os.makedirs('data/JAAD_data')
                    with open("data/JAAD_data/JAAD_{}.pkl".format(data_type), 'wb') as f:
                        pickle.dump(dd, f)
                elif self._dataset == 'PIE':
                    if not os.path.exists('data/PIE_data'):
                        os.makedirs('data/PIE_data')
                    with open("data/PIE_data/PIE_{}.pkl".format(data_type), 'wb') as f:
                        pickle.dump(dd, f)

        return dd

