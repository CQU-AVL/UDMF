import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import gc
import glob

import numpy
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR, LambdaLR, SequentialLR
from utils.jaad_data import JAAD
from utils.jaad_preprocessing import *
# from utils.my_dataset import OpticalFlowProcessor, CustomDataset, of_collate_fn
from utils.my_dataset2 import CustomDataset, of_collate_fn, VideoAugmentation
# from utils.my_dataset3 import *
from networks.loss import EDLLOSS, BeliefMatchingLoss, Distill_Loss, FocalLoss, COLOSS
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from networks.double_model import MyModel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    if not args.learn:
        seed_all(args.seed)
        data_opts = {'fstride': 1,
                     'sample_type': args.bh,  # 'beh'
                     'subset': 'default',
                     'height_rng': [0, float('inf')],
                     'squarify_ratio': 0,
                     'data_split_type': 'default',  # kfold, random, default
                     'seq_type': 'crossing',
                     'min_track_size': 15,
                     'random_params': {'ratios': None,
                                       'val_data': True,
                                       'regen_data': False},
                     'kfold_params': {'num_folds': 5, 'fold': 1},
                     }
        tte = [30, 60]
        imdb = JAAD(data_path=args.set_path)

        seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
        # seq_train = balance_dataset(seq_train)
        tte_seq_train, traj_seq_train = tte_dataset(seq_train, tte, args.train_overlap, args.times_num)
        del seq_train
        gc.collect()

        seq_valid = imdb.generate_data_trajectory_sequence('val', **data_opts)
        # seq_valid = balance_dataset(seq_valid)
        tte_seq_valid, traj_seq_valid = tte_dataset(seq_valid, tte, args.valid_overlap, args.times_num)
        del seq_valid
        gc.collect()

        seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)
        tte_seq_test, traj_seq_test = tte_dataset(seq_test, tte, 0, args.times_num)
        del seq_test
        gc.collect()

        bbox_train = tte_seq_train['bbox']
        bbox_valid = tte_seq_valid['bbox']
        bbox_test = tte_seq_test['bbox']

        vel_train = tte_seq_train['vehicle_act']
        vel_valid = tte_seq_valid['vehicle_act']
        vel_test = tte_seq_test['vehicle_act']

        action_train = tte_seq_train['activities']
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
        
        assert len(img_path_train) == len(bbox_train)
        assert len(img_path_val) == len(bbox_valid)
        assert len(img_path_test) == len(bbox_test)

        del tte_seq_train, tte_seq_valid, tte_seq_test, traj_seq_train, traj_seq_valid, traj_seq_test
        gc.collect()

        normalized_bbox_train = normalize_bbox(bbox_train)
        normalized_bbox_valid = normalize_bbox(bbox_valid)
        normalized_bbox_test = normalize_bbox(bbox_test)

        label_action_train = prepare_label(action_train)
        label_action_valid = prepare_label(action_valid)
        label_action_test = prepare_label(action_test)

        bbox_train, bbox_valid = torch.Tensor(numpy.array(normalized_bbox_train)), torch.Tensor(
            numpy.array(normalized_bbox_valid))
        label_train, label_valid = torch.Tensor(numpy.array(label_action_train)), torch.Tensor(
            numpy.array(label_action_valid))
        bbox_test = torch.Tensor(numpy.array(normalized_bbox_test))
        label_test = torch.Tensor(numpy.array(label_action_test))

        num_pos = (label_train == 1).sum().item()
        total_count = len(label_train)
        num_neg = total_count - num_pos

        del label_action_train, label_action_valid, label_action_test
        gc.collect()

        vel_train = torch.Tensor(numpy.array(vel_train))
        vel_valid = torch.Tensor(numpy.array(vel_valid))
        vel_test = torch.Tensor(numpy.array(vel_test))

        # prep = prepare_data(args)
        # train_data, val_data, test_data = prep.train_data, prep.val_data, prep.test_data

        # get_flow_train = OpticalFlowProcessor(img_path_train, bbox_train)
        # get_flow_valid = OpticalFlowProcessor(img_path_val, bbox_valid)
        # get_flow_test = OpticalFlowProcessor(img_path_test, bbox_test)
        #
        # of_data_train = get_flow_train.process(args.of_file, "train", args)
        # of_data_valid = get_flow_valid.process(args.of_file, "valid", args)
        # of_data_test = get_flow_test.process(args.of_file, "test", args)
        #
        transform_train = VideoAugmentation(is_train=True)
        transform_ntrain = VideoAugmentation(is_train=False)

        trainset = CustomDataset(args, label_train, bbox_train,vel_train, img_path_train, transform_train,
                                 "train")
        validset = CustomDataset(args, label_valid, bbox_valid,vel_valid, img_path_val, transform_ntrain,
                                 "val")
        testset = CustomDataset(args, label_test, bbox_test,vel_test, img_path_test, transform_ntrain,
                                "test")

        # pos_weight = 1.85

        # weights = torch.tensor([pos_weight if l == 0 else 1.0 for l in trainset.labels])
        # sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

        weight_neg = 1.0 / num_neg
        weight_pos = 1.0 / num_pos

        # 为数据集中的每个样本分配权重
        # 如果样本标签是 0，给 weight_neg；是 1，给 weight_pos
        # sample_weights = [weight_neg if t == 0 else weight_pos for t in trainset.labels]
        # sample_weights = torch.DoubleTensor(sample_weights)
        # sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        # trainset = CustomDataset(args, bbox_train, label_train, vel_train, bbox_train_dec, img_path_train,
        #                          False, "train")
        # validset = CustomDataset(args, bbox_valid, label_valid, vel_valid, bbox_valid_dec, img_path_val,
        #                          False, "val")
        # testset = CustomDataset(args, bbox_test, label_test, vel_test, bbox_test_dec, img_path_test,
        #                         False, "test")

        # train_loader = create_balanced_train_loader(trainset, batch_size=args.batch_size, of_collate_fn=of_collate_fn,
        #                                             drop_last=True)
        train_loader = DataLoader(trainset, batch_size=args.batch_size, collate_fn=of_collate_fn,
                                  drop_last=True, num_workers=8, pin_memory=True, persistent_workers=True,shuffle=True)
        valid_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False, collate_fn=of_collate_fn,
                                  drop_last=True)
        test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, collate_fn=of_collate_fn,
                                 drop_last=True)

    else:  # 生成随机数据
        train_loader = [[torch.randn(size=(args.batch_size, args.times_num, args.bbox_input)),
                         torch.randn(size=(args.batch_size, 1)),
                         torch.randn(size=(args.batch_size, args.times_num, args.vel_input)),
                         torch.randn(size=(args.batch_size, args.times_num, args.bbox_input))]]
        valid_loader = [[torch.randn(size=(args.batch_size, args.times_num, args.bbox_input)),
                         torch.randn(size=(args.batch_size, 1)),
                         torch.randn(size=(args.batch_size, args.times_num, args.vel_input)),
                         torch.randn(size=(args.batch_size, args.times_num, args.bbox_input))]]
        test_loader = [[torch.randn(size=(args.batch_size, args.times_num, args.bbox_input)),
                        torch.randn(size=(args.batch_size, 1)),
                        torch.randn(size=(args.batch_size, args.times_num, args.vel_input)),
                        torch.randn(size=(args.batch_size, args.times_num, args.bbox_input))]]
    print('Start Training Loop... \n')

    model = MyModel(args)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    # Warmup 示例 (用 torch.optim.lr_scheduler)
    # warmup_epochs = 4

    # def warmup_lambda(epoch):  # epoch 从 0 开始
    #     if epoch < warmup_epochs:
    #         return (epoch + 1) / warmup_epochs
    #     return 1.0

    # warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-5)

    # cls_criterion = FocalLoss()
    cls_criterion = nn.BCEWithLogitsLoss(torch.tensor([2.5]).to(device))
    img_criterion = BeliefMatchingLoss()
    bv_criterion = COLOSS()
    dis_criterion = Distill_Loss()

    model_folder_name = 'checkpoint_new' + args.bh
    try:
        checkpoint_filepath = '/media/avl/disk1/lr/PedCMT/checkpoints/JAAD_checkpoint'
        pth_files = glob.glob(os.path.join(checkpoint_filepath, "*.pth"))
        writer = SummaryWriter('/media/avl/disk1/lr/PedCMT/logs/{}'.format(model_folder_name))
    except:
        checkpoint_filepath = '/mnt/d/pycharm_files/learn/myprojcet/PedCMT/checkpoints'
        pth_files = glob.glob(os.path.join(checkpoint_filepath, "*.pth"))
        writer = SummaryWriter('/mnt/d/pycharm_files/learn/myprojcet/PedCMT/logs/{}'.format(model_folder_name))

    finetune = True
    if finetune:
        # 1. 【关键】必须加载表现最好的那个 checkpoint
        warm_start_path = os.path.join(checkpoint_filepath, 'checkpoint13nb.pth') 
        checkpoint = torch.load(warm_start_path, map_location=device)

        # 2. 只加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded BEST model weights from {warm_start_path} (Epoch {checkpoint['epoch']})")

        start_epoch = 0 
        
        # 5. 手动设置极小的 LR 进行微调
        # 既然 checkpoint0029 已经很好了，我们只需要极其轻微的调整
        current_lr = 1e-5  # 建议比之前(1e-5)再小 10 倍
        
        # 重新定义优化器 (清除历史动量)
        optimizer = torch.optim.AdamW(model.parameters(), lr=current_lr, weight_decay=1e-4)
        
        # 如果需要 scheduler，重新定义一个新的
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7) # 比如只微调 20 轮

    else:
        start_epoch = 0
        print("No checkpoint found, starting from scratch")

    # train(model, train_loader, valid_loader, cls_criterion, img_criterion, bv_criterion, dis_criterion, optimizer, checkpoint_filepath,
    #       writer, args, True, start_epoch, scheduler=scheduler)

    # Test
    for pth_file in pth_files:
        model = MyModel(args)
        model.to(device)

        file_name = pth_file.split("/")[-1]
        file_num = file_name.split(".")[0][-3:]

        skip_file = False

    # --- 如果 file_num 是数字，判断是否跳过 ---
        try:
            file_num = int(file_num)
            # 数字且大于 epochs 才跳过
            if file_num > int(args.epochs):
                print(f"Skipping {file_name}, because file_num {file_num} > epochs {args.epochs}.")
                skip_file = True
        except ValueError:
            # 非数字 → 不跳过，继续测试
            print(f"{file_name}: file_num '{file_num}' is not numeric, but will still be tested.")

        if skip_file:
            continue

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
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--set_path', type=str, default='/media/avl/disk1/lr/JAAD/JAAD-JAAD_2.0',
                        help="/mnt/d/datasets/JAAD")
    parser.add_argument('--bh', type=str, default='all', help='all or beh, in JAAD dataset.')
    parser.add_argument('--train_overlap', type=float, default=0.6, help='')
    parser.add_argument('--valid_overlap', type=float, default=0.0, help='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cache_dir', type=str, default="/media/avl/disk1/lr/JAAD/JAAD-JAAD_2.0/select_images",
                        help="/mnt/f/preprocess_data")
    parser.add_argument('--backbone', type=str, default="resnet50", help='resnet50 or dinov3_vits16')
    parser.add_argument('--loss_mod', type=str, default="digamma")
    parser.add_argument('--weight_path', type=str, default='/media/avl/disk1/lr/PedCMT/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
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
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate to train.')

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
