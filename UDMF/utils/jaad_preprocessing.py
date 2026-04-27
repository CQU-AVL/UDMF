import math
import sys
from pathlib import Path
import numpy
import torch
import numpy as np
import random
from PIL import Image
import os

from tqdm import tqdm
from torchmetrics import Accuracy

import torchvision.transforms.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.insert(0, "/media/avl/disk1/lr/PedCMT")
from networks.FFN import post_process
acc = Accuracy(task="binary", num_classes=1).to(device)
post_pro = post_process()

def seed_all(seed):
    torch.cuda.empty_cache()
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def end_point_loss(reg_criterion, pred, end_point):
    for i in range(4):
        if i == 0 or i == 2:
            pred[:, i] = pred[:, i] * 1920
            end_point[:, i] = end_point[:, i] * 1920
        else:
            pred[:, i] = pred[:, i] * 1080
            end_point[:, i] = end_point[:, i] * 1080
    return reg_criterion(pred, end_point)


def train(model, train_loader, valid_loader, class_criterion, img_criterion, bv_criterion, dis_loss, optimizer, checkpoint_filepath, writer,
          args, use_dis, start_epoch, scheduler=None, warmup_scheduler=None, warmup_epochs=None):

    best_valid_loss = np.inf
    num_steps_wo_improvement = 0
    save_times = 0
    train_neg_acc = 0.0
    epochs = args.epochs
    if args.learn:  # 调试模式： epoch = 5
        epochs = 5
    time_crop = args.time_crop

    for epoch in range(start_epoch, epochs):
        nb_batches_train = len(train_loader)
        train_acc = 0
        f_losses = 0.0
        cls_losses = 0.0
        img_losses = 0.0
        bv_losses = 0.0
        d_losses = 0.0
        total_neg_samples = 0
        aux_losses = 0.0

        if epoch < 5:
            w_aux = 0.5
            w_dis = 0.0
            w_img = 0.00 
            w_bv = 0.0

        # 2. 中期：开启蒸馏，让分支对齐
        elif epoch < 15:
            w_aux = 0.5
            w_dis = 0.5
            w_img = 0.0 
            w_bv = 0.0
            
        # 3. 后期：开启不确定性，进行正则化微调
        else:
            w_aux = 0.5
            w_dis = 0.5
            w_img = 0.0005 
            w_bv = 0.005

        model.train()
        i = 0
        print('Epoch: {} training...'.format(epoch + 1))
        for batch in tqdm(train_loader, total=len(train_loader), ncols=100, desc=f"Epoch [{epoch}]"):
            label = batch['label'][:, -1, :].to(device).float()
            bbox = batch['bbox'].to(device)
            vel = batch['vel'].to(device)
            img = batch['image'].to(device)

            if np.random.randint(10) >= 5 and time_crop:
                crop_size = np.random.randint(args.sta_f, args.end_f)
                vel = vel[:, -crop_size:, :]

            # tag1, tag2, point1, point2, u_img, u_bv = model(img, bbox, vel)
            tag_pred, tag_i, tag_bv, u_img, u_bv, mu = model(img, bbox, vel)

            cls_loss = class_criterion(tag_pred, label)
            loss_aux_i = class_criterion(tag_i, to_onehot(label))
            loss_aux_bv = class_criterion(tag_bv, to_onehot(label))

            img_loss = img_criterion(tag_i, label)
            bv_loss = bv_criterion(tag_bv, to_onehot(label), u_bv, mu)
            if use_dis:
                d_loss = (dis_loss(tag_i, tag_bv.detach(), u_img.detach()) + dis_loss(tag_bv, tag_i.detach(), u_bv.detach())) / 2
            else:
                d_loss = 0.0
            
            f_loss = cls_loss + w_img*img_loss + w_bv*bv_loss + w_dis*d_loss + w_aux * (loss_aux_bv + loss_aux_i)

            optimizer.zero_grad(set_to_none=True)
            f_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            f_losses += f_loss.item()
            cls_losses += cls_loss.item()
            # img_loss = bv_loss = d_loss = torch.tensor([0.0]).to(device)
            img_losses += img_loss
            bv_losses += bv_loss
            d_losses += d_loss
            aux_losses += (loss_aux_bv + loss_aux_i)

            with torch.no_grad():  # 无梯度
                neg_mask = (label.squeeze(-1) == 0)  # 负样本掩码，确保label是[bs,]（squeeze如果[bs,1]）
                if neg_mask.sum() > 0:  # 确保有负样本
                    # 修复：取整个batch的正类概率 [bs,]，而非[-1]（最后一个样本）
                    pos_probs = torch.sigmoid(tag_pred)  # 假设第1列（索引1）是正类logits，[bs,]
                    neg_pos_probs = pos_probs[neg_mask]  # 负样本的正类概率 [num_neg,]
                    neg_preds_correct = (neg_pos_probs < 0.5).float()  # 正确预测负的比例
                    neg_acc = neg_preds_correct.mean().item()  # scalar
                    train_neg_acc += neg_acc * neg_mask.sum().item()  # 加权累积（乘以负样本数，避免小batch偏差）
                    total_neg_samples += neg_mask.sum().item()  # 累积总负样本数（需在循环外初始化为0）
                else:
                    neg_acc = 0.0  # 无负样本时

            optimizer.step()  #
            acc_train = acc(tag_pred.sigmoid(), label)
            train_acc += acc_train
            torch.cuda.empty_cache()


        writer.add_scalar('training full_loss',
                          f_losses / nb_batches_train,
                          epoch + 1)
        writer.add_scalar('training cls_loss',
                          cls_losses / nb_batches_train,
                          epoch + 1)

        writer.add_scalar('training img_loss',
                          img_losses / nb_batches_train,
                          epoch + 1)
        writer.add_scalar('training bv_loss',
                          bv_losses / nb_batches_train,
                          epoch + 1)
        writer.add_scalar('training dis_loss',
                          d_losses / nb_batches_train,
                          epoch + 1)
        writer.add_scalar('training Acc',
                          train_acc / nb_batches_train,
                          epoch + 1)

        print(
            f"Epoch {epoch + 1}: | Train_Loss {f_losses / nb_batches_train} | Train Cls_loss {cls_losses / nb_batches_train}"
            f" | Train img_loss {img_losses / nb_batches_train} | Train Distil_loss {d_losses / nb_batches_train} \n"
            f"| Train bv_loss {bv_losses / nb_batches_train} | Train aux_loss {aux_losses / nb_batches_train}"
            f" | Train_Acc {train_acc / nb_batches_train} ")

        avg_neg_acc = train_neg_acc / nb_batches_train
        print(f"Epoch {epoch + 1}: Avg Train neg_acc = {avg_neg_acc:.3f} (>0.7? {avg_neg_acc > 0.7})")
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('learning_rate', current_lr, epoch + 1)
        print(f"Current Learning Rate: {current_lr:.6f}")

        # valid_f_loss, valid_cls_loss, valid_reg_loss, val_acc, val_dis_loss = evaluate(model, valid_loader, class_criterion,
        #                                                                  reg_criterion, dis_loss, args, epoch)
        valid_f_loss, valid_cls_loss, valid_img_loss, valid_bv_loss, valid_d_loss, val_acc = evaluate(model, valid_loader, class_criterion, img_criterion, bv_criterion, dis_loss, epoch, use_dis,
                                                                                                      w_img, w_bv, w_dis, w_aux)
        

        if scheduler is not None:
            scheduler.step()
            
        writer.add_scalar('validation full_loss',
                          valid_f_loss,
                          epoch + 1)
        writer.add_scalar('validation cls_loss',
                          valid_cls_loss,
                          epoch + 1)
        writer.add_scalar('validation l1_loss',
                          valid_img_loss,
                          epoch + 1)
        writer.add_scalar('validation giou_loss1',
                          valid_bv_loss,
                          epoch + 1)
        writer.add_scalar('validation dis_loss',
                          valid_d_loss,
                          epoch + 1)
        writer.add_scalar('validation Acc',
                          val_acc,
                          epoch + 1)

        is_best = valid_f_loss < best_valid_loss
        if is_best:
            best_valid_loss = valid_f_loss
            num_steps_wo_improvement = 0
            save_times += 1
            print(f'{save_times} time(s) File saved.\n')
            print('Update improvement.\n')
        else:
            num_steps_wo_improvement += 1
            print(f'{num_steps_wo_improvement}/30 times Not update.\n')

        # 构建需要保存的 checkpoint 内容
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'Accuracy': train_acc / nb_batches_train,
            'LOSS': f_losses / nb_batches_train,
        }

        # 定义需要保存的路径列表
        if epoch < 20:
            torch.save(checkpoint, os.path.join(checkpoint_filepath, f'checkpoint{epoch:04}.pth'))
        elif (epoch + 1) % 5 == 0:
            torch.save(checkpoint, os.path.join(checkpoint_filepath, f'checkpoint{epoch:04}.pth'))  # 每 10 轮保存额外的备份



def evaluate(model, val_data, class_criterion, img_criterion, bv_criterion, dis_loss, epoch, use_dis, w_img, w_bv, w_dis, w_aux):
    nb_batches = len(val_data)
    val_f_losses = 0.0
    val_cls_losses = 0.0
    val_img_losses = 0.0
    val_bv_losses = 0.0
    val_d_losses = 0.0
    val_precision = 0.0
    val_recall = 0.0
    val_f1 = 0.0
    valid_neg_acc = 0.0
    total_neg_samples=0.0
    val_aux_losses = 0.0

    print('in Validation...')
    model.eval()
    with torch.no_grad():
        val_acc = 0
        all_logits = []  # Collect all raw logits for distribution monitoring
        all_preds = []  # Collect all predictions (0 or 1)
        all_labels = []  # Collect all true labels for comparison
        total_samples = 0
        for i, batch in enumerate(val_data):

            label = batch['label'][:, -1, :].to(device).float()
            bbox = batch['bbox'].to(device)
            vel = batch['vel'].to(device)
            img = batch['image'].to(device)

            # tag1, tag2, point1, point2, u_img, u_bv = model(img, bbox, vel)
            tag_pred, tag_i, tag_bv, u_img, u_bv, mu = model(img, bbox, vel, False)
            # indices = get_indices(point, end_point, tag, label)
            # tag1, point1 = post_process(point1, end_point, tag1, label)
            # tag2, point2 = post_process(point2, end_point, tag2, label)

            # Compute probabilities and predictions for binary classification
            probs = torch.sigmoid(tag_pred)  # shape (bs,)
            preds = (probs > 0.5).float()  # shape (bs,), 0 or 1

            # Accumulate for monitoring
            all_logits.append(tag_pred.cpu())  # Keep as (bs, 2)
            all_preds.append(preds.cpu())
            all_labels.append(label.squeeze(-1).cpu())  # Assume label is (bs,1), flatten to (bs,)
            total_samples += len(preds)

            val_cls_loss = class_criterion(tag_pred, label)
            val_cls_loss_i = class_criterion(tag_i, to_onehot(label))
            val_cls_loss_bv = class_criterion(tag_bv, to_onehot(label))
            val_img_loss = img_criterion(tag_i, label)
            val_bv_loss = bv_criterion(tag_bv, to_onehot(label), u_bv, mu)
            if use_dis:
                val_d_loss = (dis_loss(tag_i, tag_bv.detach(), u_img.detach()) + dis_loss(tag_bv, tag_i.detach(), u_bv.detach())) / 2
            else:
                val_d_loss = 0.0

            # val_img_loss = val_bv_loss = val_d_loss = torch.tensor([0.0]).to(device)

            # 添加 neg_acc
            neg_mask = (label.squeeze(-1) == 0)
            if neg_mask.sum() > 0:
                neg_probs = probs[neg_mask]  # Positive probs for negative samples
                neg_preds_correct = (neg_probs < 0.5).float()  # Correct if prob < 0.5 for neg class
                val_neg_acc = neg_preds_correct.mean().item()
                valid_neg_acc += val_neg_acc * neg_mask.sum().item()  # Weighted by num neg samples
                total_neg_samples += neg_mask.sum().item()  # Accumulate total neg for avg
            else:
                val_neg_acc = 0.0
                total_neg_samples += 0

            f_loss = val_cls_loss + w_img * val_img_loss + w_bv * val_bv_loss + w_dis * val_d_loss + w_aux * (val_cls_loss_bv + val_cls_loss_i)

            val_f_losses += f_loss.item()
            val_cls_losses += val_cls_loss.item()
            val_img_losses += val_img_loss.item()
            val_bv_losses += val_bv_loss.item()
            if use_dis:
                val_d_losses += val_d_loss.item()
            val_aux_losses += (val_cls_loss_bv + val_cls_loss_i)

            metrics = post_pro.evaluate(tag_pred, label)
            val_precision += metrics[1]
            val_recall += metrics[2]
            val_f1 += metrics[3]
            val_acc += metrics[0]

    print(
        f'Valid_Full_Loss {val_f_losses / nb_batches} | Valid Cls_loss {val_cls_losses / nb_batches} | Valid img_loss '
        f'{val_img_losses / nb_batches} | Valid bv_loss {val_bv_losses / nb_batches} | valid aux_loss {val_aux_losses / nb_batches}\n'
        f'Valid Distill_loss {val_d_losses / nb_batches} | Valid_Acc {val_acc / nb_batches} | \n'
        f'val_precision {val_precision / nb_batches} | val_recall {val_recall / nb_batches} | val_f1 {val_f1 / nb_batches}'
    )

    all_logits = torch.cat(all_logits)  # [total_samples, 2]
    all_preds = torch.cat(all_preds)  # [total_samples,]
    all_labels = torch.cat(all_labels)  # [total_samples,]

    # Monitor distribution
    pos_pred_count = (all_preds == 1).sum().item()
    pos_pred_prop = pos_pred_count / total_samples
    unique_preds, counts = torch.unique(all_preds, return_counts=True)

    print(f"Validation set size: {total_samples}")
    print(f"Predicted positive proportion: {pos_pred_prop:.4f} (count: {pos_pred_count}/{total_samples})")
    print(f"Unique predictions: {unique_preds.tolist()} with counts: {counts.tolist()}")
    print(f"Logits mean (pos class): {all_logits.mean():.4f}, std: {all_logits.std():.4f}")
    print(
        f"Probs mean (pos class, last batch): {probs.mean():.4f}, std: {probs.std():.4f}"
    )  # For full: torch.softmax(all_logits, dim=1)[:, 1].mean()

    # Optional: Entropy of average prediction distribution (low entropy = collapsed to one class)
    avg_pos_prob = torch.sigmoid(all_logits).mean().item()
    avg_neg_prob = 1 - avg_pos_prob

    # Compute entropy of binary distribution
    entropy = - (avg_pos_prob * math.log(avg_pos_prob + 1e-8) + avg_neg_prob * math.log(avg_neg_prob + 1e-8))
    print(f"Average prediction entropy: {entropy:.4f} (higher = more diverse, ~0.693 for uniform)")

    # Compare to true labels
    true_pos_prop = (all_labels == 1).float().mean().item()
    print(f"True positive proportion: {true_pos_prop:.4f}")

    avg_neg_acc = valid_neg_acc / total_neg_samples if total_neg_samples > 0 else 0.0
    print(f"Epoch {epoch + 1}: Avg Valid neg_acc = {avg_neg_acc:.3f} (>0.7? {avg_neg_acc > 0.7})")

    # If all predictions are the same
    if len(unique_preds) == 1:
        print("WARNING: All predictions are the same class! Model collapsed.")

    # return val_f_losses / nb_batches, val_cls_losses / nb_batches, val_reg_losses / nb_batches, val_acc / nb_batches, val_distill_losses / nb_batches
    return (val_f_losses / nb_batches, val_cls_losses / nb_batches, val_img_losses / nb_batches, val_bv_losses / nb_batches,
            val_d_losses / nb_batches, val_acc / nb_batches)


def test(model, test_data):
    print('Tesing...')

    with torch.no_grad():
        model.eval()
        step = 0
        for batch in test_data:
            label = batch['label'][:, -1, :].to(device).float()
            bbox = batch['bbox'].to(device)
            vel = batch['vel'].to(device)
            img = batch['image'].to(device)

            # tag1, tag2, _, _, _, _ = model(img, bbox, vel)
            tag, _, _, _, _, _ = model(img, bbox, vel, False)
            tag = torch.nn.functional.dropout(tag, p=0.2, training=False)

            # evidence1 = class_criterion.relu_evidence(tag1)
            # evidence2 = class_criterion.relu_evidence(tag2)
            # alpha1 = evidence1 + 1
            # alpha2 = evidence2 + 1
            # prob1 = alpha1 / torch.sum(alpha1, dim=1, keepdim=True)
            # prob2 = alpha2 / torch.sum(alpha2, dim=1, keepdim=True)
            # tag1_hat = prob1  # [:,1]
            # tag2_hat = prob2
            # u = (u_bv.squeeze(-1) + u_img.squeeze(-1)) / 2

            tag1_hat = tag
            # tag2_hat = tag2
            if step == 0:
                preds1 = tag1_hat
                # preds2 = tag2_hat
                labels = label
                # us = u
            else:
                preds1 = torch.cat((preds1, tag1_hat), 0)
                # preds2 = torch.cat((preds2, tag2_hat), 0)
                labels = torch.cat((labels, label), 0)
                # us = torch.cat((us, u), 0)
            step += 1

    return preds1, labels


def balance_dataset(dataset, flip=True):
    d = {'bbox': dataset['bbox'].copy(),
         'pid': dataset['pid'].copy(),
         'activities': dataset['activities'].copy(),
         'image': dataset['image'].copy(),
         'center': dataset['center'].copy(),
         'vehicle_act': dataset['vehicle_act'].copy(),
         'image_dimension': (1920, 1080)}
    gt_labels = [gt[0] for gt in d['activities']]
    num_pos_samples = np.count_nonzero(np.array(gt_labels))
    num_neg_samples = len(gt_labels) - num_pos_samples

    if num_neg_samples == num_pos_samples:
        print('Positive samples is equal to negative samples.')
    else:
        print('Unbalanced: \t Postive: {} \t Negative: {}'.format(num_pos_samples, num_neg_samples))
        if num_neg_samples > num_pos_samples:
            gt_augment = 1
        else:
            gt_augment = 0

        img_width = d['image_dimension'][0]
        num_samples = len(d['pid'])

        for i in range(num_samples):
            if d['activities'][i][0][0] == gt_augment:
                flipped = d['center'][i].copy()
                flipped = [[img_width - c[0], c[1]] for c in flipped]
                d['center'].append(flipped)

                flipped = d['bbox'][i].copy()
                flipped = [np.array([img_width - c[2], c[1], img_width - c[0], c[3]]) for c in flipped]
                d['bbox'].append(flipped)

                d['pid'].append(dataset['pid'][i].copy())

                d['activities'].append(d['activities'][i].copy())
                d['vehicle_act'].append(d['vehicle_act'][i].copy())

                flipped = d['image'][i].copy()
                flipped_images = []

                for img_path in tqdm(flipped, desc="flip_images", unit="image"):

                    # 生成新的图像路径
                    new_img_path = img_path.replace('.png', '_flip.png')
                    if not os.path.exists(new_img_path):
                        # 打开图像
                        img = Image.open(img_path)

                        # 翻转图像
                        flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)

                        # 保存翻转后的图像
                        flipped_img.save(new_img_path)

                    # 添加新的图像路径到列表
                    flipped_images.append(new_img_path)

                # 将翻转后的图像路径添加到数据集中
                d['image'].append(flipped_images)

        gt_labels = [gt[0] for gt in d['activities']]
        num_pos_samples = np.count_nonzero(np.array(gt_labels))
        num_neg_samples = len(gt_labels) - num_pos_samples

        if num_neg_samples > num_pos_samples:
            rm_index = np.where(np.array(gt_labels) == 0)[0]
        else:
            rm_index = np.where(np.array(gt_labels) == 1)[0]

        dif_samples = abs(num_neg_samples - num_pos_samples)

        np.random.seed(42)
        np.random.shuffle(rm_index)
        rm_index = rm_index[0:dif_samples]

        for k in d:
            seq_data_k = d[k]
            d[k] = [seq_data_k[i] for i in range(0, len(seq_data_k)) if i not in rm_index]

        new_gt_labels = [gt[0] for gt in d['activities']]
        num_pos_samples = np.count_nonzero(np.array(new_gt_labels))
        print('Balanced: Postive: %d \t Negative: %d \n' % (num_pos_samples, len(d['activities']) - num_pos_samples))
        print('Total Number of samples: %d\n' % (len(d['activities'])))

    return d


def tte_dataset(dataset, time_to_event, overlap, obs_length):
    d_obs = {'bbox': dataset['bbox'].copy(),
             'pid': dataset['pid'].copy(),
             'activities': dataset['activities'].copy(),
             'image': dataset['image'].copy(),
             'vehicle_act': dataset['vehicle_act'].copy(),
             'center': dataset['center'].copy()
             }

    d_tte = {'bbox': dataset['bbox'].copy(),
             'pid': dataset['pid'].copy(),
             'activities': dataset['activities'].copy(),
             'image': dataset['image'].copy(),
             'vehicle_act': dataset['vehicle_act'].copy(),
             'center': dataset['center'].copy()}

    if isinstance(time_to_event, int):
        for k in d_obs.keys():
            for i in range(len(d_obs[k])):
                d_obs[k][i] = d_obs[k][i][- obs_length - time_to_event: -time_to_event]
                d_tte[k][i] = d_tte[k][i][- time_to_event:]
        d_obs['tte'] = [[time_to_event]] * len(dataset['bbox'])
        d_tte['tte'] = [[time_to_event]] * len(dataset['bbox'])

    else:
        olap_res = obs_length if overlap == 0 else int((1 - overlap) * obs_length)  # 0.4*32
        olap_res = 1 if olap_res < 1 else olap_res  # 12
        for k in d_obs.keys():
            seqs = []
            seqs_tte = []
            for seq in d_obs[k]:
                start_idx = len(seq) - obs_length - time_to_event[1]
                end_idx = len(seq) - obs_length - time_to_event[0]
                seqs.extend([seq[i:i + obs_length] for i in range(start_idx, end_idx, olap_res)])
                seqs_tte.extend([seq[i + obs_length:] for i in range(start_idx, end_idx, olap_res)])
                d_obs[k] = seqs
                d_tte[k] = seqs_tte
        tte_seq = []
        for seq in dataset['bbox']:
            start_idx = len(seq) - obs_length - time_to_event[1]
            end_idx = len(seq) - obs_length - time_to_event[0]
            tte_seq.extend([[len(seq) - (i + obs_length)] for i in range(start_idx, end_idx, olap_res)])
            d_obs['tte'] = tte_seq.copy()
            d_tte['tte'] = tte_seq.copy()

    remove_index = []
    try:
        time_to_event_0 = time_to_event[0]
    except:
        time_to_event_0 = time_to_event
    for seq_index, (seq_obs, seq_tte) in enumerate(zip(d_obs['bbox'], d_tte['bbox'])):
        if len(seq_obs) < obs_length or len(seq_tte) < time_to_event_0:
            remove_index.append(seq_index)

    for k in d_obs.keys():
        for j in sorted(remove_index, reverse=True):
            del d_obs[k][j]
            del d_tte[k][j]

    return d_obs, d_tte

def normalize_bbox(dataset, width=1920, height=1080):
    normalized_set = []
    for sequence in dataset:
        if sequence == []:
            continue
        normalized_sequence = []
        for bbox in sequence:
            np_bbox = np.zeros(4)
            np_bbox[0] = bbox[0] / width
            np_bbox[2] = bbox[2] / width
            np_bbox[1] = bbox[1] / height
            np_bbox[3] = bbox[3] / height
            normalized_sequence.append(np_bbox)
        normalized_set.append(np.array(normalized_sequence))

    return normalized_set


def normalize_traj(dataset, width=1920, height=1080):
    normalized_set = []
    for sequence in dataset:
        if sequence == []:
            continue
        normalized_sequence = []
        for bbox in sequence:
            np_bbox = np.zeros(4)
            np_bbox[0] = bbox[0] / width
            np_bbox[2] = bbox[2] / width
            np_bbox[1] = bbox[1] / height
            np_bbox[3] = bbox[3] / height
            normalized_sequence.append(np_bbox)
        normalized_set.append(np.array(normalized_sequence))

    return normalized_set


def prepare_label(dataset):
    labels = np.zeros(len(dataset), dtype='int64')
    for step, action in enumerate(dataset):
        if action == []:
            continue
        labels[step] = action[0][0]

    return labels


def pad_sequence(inp_list, max_len):
    padded_sequence = []
    for source in inp_list:
        target = np.array([source[0]] * max_len)
        source = numpy.array(source)
        target[-source.shape[0]:, :] = source

        padded_sequence.append(target)

    return padded_sequence


def cxcywh_to_xyxy(bbox):
    # bbox: Tensor of shape (4,) in [cx, cy, w, h] normalized
    cx, cy, w, h = bbox
    x_min = cx - w / 2
    y_min = cy - h / 2
    x_max = cx + w / 2
    y_max = cy + h / 2
    return np.array([x_min, y_min, x_max, y_max])

def scale_bbox(bbox_xyxy, img_size=(1920, 1080)):
    w, h = img_size
    bbox_xyxy = bbox_xyxy.cpu().numpy() if isinstance(bbox_xyxy, torch.Tensor) else bbox_xyxy
    bbox_scaled = bbox_xyxy.copy()
    bbox_scaled[0::2] *= w  # x_min and x_max
    bbox_scaled[1::2] *= h  # y_min and y_max
    return bbox_scaled


def to_onehot(tensor_data):
    y = tensor_data.squeeze()  # 假设 tensor_data 是 shape (bs, 1) 的 tensor，squeeze 后为 (bs,)
    y_mat = torch.zeros((len(y), 2), device=tensor_data.device, dtype=tensor_data.dtype)

    y_mat[y == 1, 1] = 1
    y_mat[y == 0, 0] = 1

    return y_mat