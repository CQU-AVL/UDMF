import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import glob

from utils.pie_data import PIE
from utils.pie_preprocessing import *
import gc
import torch
from torch import nn
from torch.utils.data import WeightedRandomSampler, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from networks.double_model import MyModel
from utils.my_dataset2 import CustomDataset, of_collate_fn, VideoAugmentation
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
import argparse

from networks.loss import BeliefMatchingLoss, Distill_Loss, FocalLoss, COLOSS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    if not args.learn: # 如果args.learn为False，则真实训练， 读取真实数据
        seed_all(args.seed)
        data_opts = {
            'fstride': 1,
            'sample_type': 'all',
            'height_rng': [0, float('inf')],
            'squarify_ratio': 0,
            'data_split_type': 'default',  # kfold, random, default
            'seq_type': 'crossing',  # crossing , intention
            'min_track_size': 15,  # discard tracks that are shorter
            'kfold_params': {'num_folds': 5, 'fold': 1},
            'random_params': {'ratios': None,
                            'val_data': True,
                            'regen_data': False},
            'tte': [30, 60],
            'batch_size': args.batch_size
        }
        imdb = PIE(data_path=args.set_path) 
        seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts) # 生成训练集
        tte_seq_train, traj_seq_train = tte_dataset(seq_train, data_opts['tte'], args.train_overlap, args.times_num) # 生成训练集的tte和轨迹
        del seq_train
        gc.collect()

        seq_valid = imdb.generate_data_trajectory_sequence('val', **data_opts)
        tte_seq_valid, traj_seq_valid = tte_dataset(seq_valid, data_opts['tte'], args.valid_overlap, args.times_num)
        del seq_valid
        gc.collect()

        seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)
        tte_seq_test, traj_seq_test = tte_dataset(seq_test, data_opts['tte'], 0, args.times_num)

        bbox_train = tte_seq_train['bbox'] # 训练集的bbox
        bbox_valid = tte_seq_valid['bbox']
        bbox_test = tte_seq_test['bbox']

        obd_train = tte_seq_train['obd_speed'] # 训练集的速度
        obd_valid = tte_seq_valid['obd_speed']
        obd_test = tte_seq_test['obd_speed']

        # gps_train = tte_seq_train['gps_speed'] # 训练集的速度
        # gps_valid = tte_seq_valid['gps_speed']
        # gps_test = tte_seq_test['gps_speed']

        action_train = tte_seq_train['activities'] # 训练集的动作
        action_valid = tte_seq_valid['activities']
        action_test = tte_seq_test['activities']

        # img_path_train = tte_seq_train["image"]
        # img_path_val = tte_seq_valid["image"]
        # img_path_test = tte_seq_test["image"]

        img_path_train = sorted(
            [f"{args.cache_dir}_all/train_cache/{i}.jpg" for i in range(len(bbox_train)) if
             os.path.exists(f"{args.cache_dir}_all/train_cache/{i}.jpg")],
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        img_path_val = sorted(
            [f"{args.cache_dir}_all/val_cache/{i}.jpg" for i in range(len(bbox_valid)) if
             os.path.exists(f"{args.cache_dir}_all/val_cache/{i}.jpg")],
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        img_path_test = sorted(
            [f"{args.cache_dir}_all/test_cache/{i}.jpg" for i in range(len(bbox_test)) if
             os.path.exists(f"{args.cache_dir}_all/test_cache/{i}.jpg")],
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        del tte_seq_train, tte_seq_valid, tte_seq_test, traj_seq_train, traj_seq_valid, traj_seq_test
        gc.collect()

        normalized_bbox_train = normalize_bbox(bbox_train) # 归一化bbox
        normalized_bbox_valid = normalize_bbox(bbox_valid)
        normalized_bbox_test = normalize_bbox(bbox_test)

        label_action_train = prepare_label(action_train) # 准备标签
        label_action_valid = prepare_label(action_valid)
        label_action_test = prepare_label(action_test)

        bbox_train, bbox_valid = torch.Tensor(normalized_bbox_train), torch.Tensor(normalized_bbox_valid) # 转换为tensor
        label_train, label_valid = torch.Tensor(label_action_train), torch.Tensor(label_action_valid)
        bbox_test = torch.Tensor(normalized_bbox_test)
        label_test = torch.Tensor(label_action_test)

        del label_action_train, label_action_valid, label_action_test
        gc.collect()

        vel_train = torch.Tensor(obd_train) # 转换为tensor
        vel_valid = torch.Tensor(obd_valid)
        vel_test = torch.Tensor(obd_test)

        transform_train = VideoAugmentation(is_train=True)
        transform_ntrain = VideoAugmentation(is_train=False)

        label_train_res, bbox_train_res, vel_train_res, img_path_train_res = apply_smote(
            label_train, bbox_train, vel_train, img_path_train, sampling_strategy=2.0
        )
        trainset_res = CustomDataset(args, label_train_res, bbox_train_res, vel_train_res, img_path_train_res,
                                     transform_train, "train")

        # trainset = CustomDataset(args, label_train, bbox_train, vel_train, img_path_train, transform_train,
        #                          "train")
        validset = CustomDataset(args, label_valid, bbox_valid, vel_valid, img_path_val, transform_ntrain,
                                 "val")
        testset = CustomDataset(args, label_test, bbox_test, vel_test, img_path_test, transform_ntrain,
                                "test")

        neg_weight = 0.05

        weights = torch.tensor([neg_weight if l == 0 else 1.0 for l in trainset_res.labels])
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

        train_loader = DataLoader(trainset_res, batch_size=args.batch_size, collate_fn=of_collate_fn,
                                  drop_last=True,num_workers=8, pin_memory=True,persistent_workers=True, sampler=sampler)
        valid_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False, collate_fn=of_collate_fn,
                                  drop_last=True)
        test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, collate_fn=of_collate_fn,
                                 drop_last=True)

    else: # args.learn为True，不真实训练，生成随机数据。
        train_loader = [[torch.randn(size=(args.batch_size, args.times_num, args.bbox_input)), # bbox
                         torch.randn(size=(args.batch_size, 1)),                                # label
                         torch.randn(size=(args.batch_size, args.times_num, args.vel_input)),   # velocity
                         torch.randn(size=(args.batch_size, args.times_num, args.bbox_input))]] # trajectory
        valid_loader = [[torch.randn(size=(args.batch_size, args.times_num, args.bbox_input)), 
                         torch.randn(size=(args.batch_size, 1)), 
                         torch.randn(size=(args.batch_size, args.times_num, args.vel_input)), 
                         torch.randn(size=(args.batch_size, args.times_num, args.bbox_input))]]
        test_loader = [[torch.randn(size=(args.batch_size, args.times_num, args.bbox_input)), 
                        torch.randn(size=(args.batch_size, 1)), 
                        torch.randn(size=(args.batch_size, args.times_num, args.vel_input)), 
                        torch.randn(size=(args.batch_size, args.times_num, args.bbox_input))]]
    print('Start Training Loop... \n')

    model = MyModel(args) # 生成模型
    model.to(device) # 放到gpu上

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=1.5e-4)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1.5e-4,  # 从上一轮 2e-4 保持
        total_steps=total_steps,  # 总迭代步数
        pct_start=0.15,  # warmup 占 10% 总 epochs (即 warmup_epochs=10)
        anneal_strategy='cos',
        three_phase=True,  # 添加三阶段：warmup + hold + anneal，稳定 PIE 长序列
        div_factor=30.0,  # 初始 LR = max_lr / div_factor (缓慢起步)
        final_div_factor=1e5  # 结束 LR 很小，防过拟合
    )

    pos_weight = torch.tensor([10.0], device=device)
    cls_criterion = FocalLoss(pos_weight=pos_weight)
    img_criterion = BeliefMatchingLoss()
    bv_criterion = COLOSS()
    dis_criterion = Distill_Loss()

    model_folder_name = 'checkpoint_new' + args.bh
    try:
        checkpoint_filepath = '/media/avl/disk1/lr/PedCMT/checkpoints/PIE_checkpoint'
        pth_files = glob.glob(os.path.join(checkpoint_filepath, "*.pth"))
        writer = SummaryWriter('/media/avl/disk1/lr/PedCMT/logs/{}'.format(model_folder_name))
    except:
        checkpoint_filepath = '/mnt/d/pycharm_files/learn/myprojcet/PedCMT/checkpoints'
        pth_files = glob.glob(os.path.join(checkpoint_filepath, "*.pth"))
        writer = SummaryWriter('/mnt/d/pycharm_files/learn/myprojcet/PedCMT/logs/{}'.format(model_folder_name))

    warm_start = False
    if warm_start:
        warm_start_path = '/media/avl/disk1/lr/PedCMT/checkpoints/checkpoint0089.pth'
        checkpoint = torch.load(warm_start_path, map_location=device)

        # 加载模型
        model.load_state_dict(checkpoint['model_state_dict'])

        # 加载优化器 (如果保存了)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 加载 scheduler (如果保存了)
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1  # 从下一 epoch 开始
        print(f"Loaded checkpoint0069.pth from epoch {checkpoint['epoch']}, continuing from epoch {start_epoch}")

        # 手动设置 LR (如果 scheduler 状态丢失，从日志查 e.g., E69 LR=0.000035)
        current_lr = 0.000035  # 从你的日志 E69 LR 替换
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
    else:
        start_epoch = 0
        print("No checkpoint found, starting from scratch")

    train(model, train_loader, valid_loader, cls_criterion, img_criterion, bv_criterion, dis_criterion, optimizer,
          checkpoint_filepath,
          writer, args, scheduler, True, start_epoch)

    # Test
    for pth_file in pth_files:
        model = MyModel(args)
        model.to(device)

        file_name = pth_file.split("/")[-1]
        checkpoint = torch.load(pth_file)
        model.load_state_dict(checkpoint['model_state_dict'])

        preds1, labels = test(model, test_loader)

        metrics = post_pro(preds1, labels)
        acc1 = metrics['acc']
        f1_1 = metrics['f1']
        pre_s1 = metrics['precision']
        recall_s1 = metrics['recall']
        auc1 = metrics['auroc']

        print(f'{file_name}:')
        print(
            f'Acc: {acc1}\n f1: {f1_1}\n precision_score: {pre_s1}\n recall_score: {recall_s1}\n roc_auc_score: {auc1}\n'
        )

if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser('Pedestrain Crossing Intention Prediction.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--set_path', type=str, default='/media/avl/disk1/lr/PIE',
                        help="/mnt/e/PIE_dataset")
    parser.add_argument('--bh', type=str, default='all', help='all or beh, in JAAD dataset.')
    parser.add_argument('--train_overlap', type=float, default=0.6, help='')
    parser.add_argument('--valid_overlap', type=float, default=0, help='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cache_dir', type=str, default="/media/avl/disk1/lr/PIE/select_image",
                        help="/mnt/e/PIE_dataset/select_image")
    parser.add_argument('--backbone', type=str, default="resnet50", help='resnet50 or dinov3_vits16')
    parser.add_argument('--loss_mod', type=str, default="digamma")
    parser.add_argument('--weight_path', type=str,
                        default='/media/avl/disk1/lr/PedCMT/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
                        help="/mnt/d/pycharm_files/learn/DEIMv2-main/ckpts/dinov3_vits16_pretrain_lvd1689m-08c60483.pth")

    parser.add_argument('--d_model', type=int, default=256, help='the dimension after embedding.')
    parser.add_argument('--dff', type=int, default=512, help='the number of the units.')
    parser.add_argument('--num_heads', type=int, default=8, help='number of the heads of the multi-head model.')
    parser.add_argument('--num_points', type=int, default=16, help='')
    parser.add_argument('--bbox_input', type=int, default=4, help='dimension of bbox.')
    parser.add_argument('--vel_input', type=int, default=1, help='dimension of velocity.')
    parser.add_argument('--time_crop', type=bool, default=False)
    parser.add_argument('--bv_input', type=int, default=5, help='dimension of bbox.')
    parser.add_argument('--num_class', type=int, default=2, help=' ')
    parser.add_argument('--annealing_step', type=int, default=10, help=' ')

    parser.add_argument('--batch_size', type=int, default=32, help='size of batch,default:64')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate to train.')

    parser.add_argument('--num_head', type=int, default=8, help='the number of heads.')
    parser.add_argument('--num_encoder', type=int, default=12, help='the number of encoder.')
    parser.add_argument('--num_decoder', type=int, default=6, help='the number of decoder.')
    parser.add_argument('--num_query', type=int, default=15, help='')
    parser.add_argument('--times_num', type=int, default=15, help='')
    parser.add_argument('--sta_f', type=int, default=8)
    parser.add_argument('--end_f', type=int, default=12)
    parser.add_argument('--learn', type=bool, default=False)
    args = parser.parse_args()
    main(args)
