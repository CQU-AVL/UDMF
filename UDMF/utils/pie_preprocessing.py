import time
from pathlib import Path
from imblearn.over_sampling import SMOTE
import torch
import os
import numpy as np
import random
from torchmetrics import Accuracy
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from networks.FFN import post_process
acc = Accuracy(task="binary", num_classes=2).to(device)
post_pro = post_process()

def seed_all(seed): # 初始化
    torch.cuda.empty_cache()
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def end_point_loss(reg_criterion, pred, end_point):# 计算端点误差（未使用）
    for i in range(4):
        if i == 0 or i == 2:
            pred[:, i] = pred[:, i] * 1920 # 1920是视频的宽
            end_point[:, i] = end_point[:, i] * 1920 
        else:
            pred[:, i] = pred[:, i] * 1080 # 1080是视频的高
            end_point[:, i] = end_point[:, i] * 1080
    return reg_criterion(pred, end_point) 



def train(model, train_loader, valid_loader, class_criterion, img_criterion, bv_criterion, dis_loss, optimizer, checkpoint_filepath, writer,
          args, scheduler, use_dis, start_epoch):

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
        cap_lr = 3.5e-5
        start_time = time.time()
        model.train()
        print('Epoch: {} training...'.format(epoch + 1))
        for batch in tqdm(train_loader, total=len(train_loader), ncols=100,
                desc=f"Epoch [{epoch}]"):

            label = batch['label'][:, -1, :].to(device).float()
            bbox = batch['bbox'].to(device)
            vel = batch['vel'].to(device)
            img = batch['image'].to(device)

            if np.random.randint(10) >= 5 and time_crop:
                crop_size = np.random.randint(args.sta_f, args.end_f)
                vel = vel[:, -crop_size:, :]

            # tag1, tag2, point1, point2, u_img, u_bv = model(img, bbox, vel)
            tag_pred, tag_i, tag_bv, u_img, u_bv, mu = model(img, bbox, vel)
            tag_pred = torch.nn.functional.dropout(tag_pred, p=0.3, training=True)
            tag_i = torch.nn.functional.dropout(tag_i, p=0.3, training=True)
            tag_bv = torch.nn.functional.dropout(tag_bv, p=0.3, training=True)
            # indices = get_indices(point, end_point, tag, label)
            # tag1, point1 = post_process(point1, end_point, tag1, label)
            # tag2, point2 = post_process(point2, end_point, tag2, label)

            # cls_loss = class_criterion(tag, label, epoch, args.num_class, args.annealing_step)
            # cls_loss2 = class_criterion(tag2, label, epoch, args.num_class, args.annealing_step)
            #
            # reg_loss2 = reg_criterion(point2, end_point, epoch)

            label_smooth = to_onehot(label) * 0.9 + (1 - to_onehot(label)) * 0.1  # [bs,2]
            cls_loss = class_criterion(tag_pred, label_smooth)
            img_loss = img_criterion(tag_i, label)
            bv_loss = bv_criterion(tag_bv, label_smooth, u_bv, mu)
            if use_dis:
                d_loss = (dis_loss(tag_i, tag_bv.detach(), u_img.detach()) + dis_loss(tag_bv, tag_i.detach(), u_bv.detach())) / 2
            else:
                d_loss = 0.0
            # reg_loss, loss_bbox, loss_giou = reg_criterion(point, end_point, indices)
            # loss_img = binary_cross_entropy_with_logits(img_logit, label.argmax(dim=1).float())
            # cls_loss = class_criterion(tag, label)
            # cls_loss2 = class_criterion(tag2, label)
            # reg_loss1, loss_bbox1, loss_giou1 = reg_criterion(point1, end_point)
            # reg_loss2, loss_bbox2, loss_giou2 = reg_criterion(point2, end_point)

            # distill_loss1 = (torch.abs(tag1 - tag2.detach()) * u_img).sum() / (u_img + 1e-7).sum()
            # distill_loss2 = (torch.abs(tag2 - tag1.detach()) * u_bv).sum() / (u_bv + 1e-7).sum()

            # distill_loss1 = dis_loss(tag1, tag2, label)
            # distill_loss2 = dis_loss(tag2, tag1, label)

            # f_loss = (2 * (cls_loss1 + cls_loss2) + 1 * (reg_loss1 + reg_loss2) + 1*(distill_loss1 + distill_loss2))
            f_loss = cls_loss + 0.1 * img_loss + 0.7 * bv_loss + 0.5 * d_loss

            # f_loss = ((cls_loss1 + cls_loss2) / (sigma_cls * sigma_cls) + (reg_loss1 + reg_loss2) /
            #           (sigma_reg * sigma_reg) + (distill_loss1 + distill_loss2) / (sigma_dis * sigma_dis) +
            #           torch.log(sigma_cls) + torch.log(sigma_reg) + torch.log(sigma_dis))

            optimizer.zero_grad(set_to_none=True)
            f_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.7)

            f_losses += f_loss.item()
            cls_losses += cls_loss.item()
            img_losses += img_loss.item()
            bv_losses += bv_loss.item()
            if use_dis:
                d_losses += d_loss.item()

            # cls_losses += (cls_loss1.item() + cls_loss2.item())
            # reg_losses += (reg_loss1.item() + reg_loss2.item())
            # l1_losses += (loss_bbox1.item() + loss_bbox2.item())
            # giou_losses1 += loss_giou1.item()
            # giou_losses2 += loss_giou2.item()
            # distill_losses += (distill_loss1.item() + distill_loss2.item())

            with torch.no_grad():  # 无梯度
                neg_mask = (label.squeeze(-1) == 0)  # 负样本掩码，确保label是[bs,]（squeeze如果[bs,1]）
                if neg_mask.sum() > 0:  # 确保有负样本
                    # 修复：取整个batch的正类概率 [bs,]，而非[-1]（最后一个样本）
                    pos_probs = torch.sigmoid(tag_pred[:, 1])  # 假设第1列（索引1）是正类logits，[bs,]
                    neg_pos_probs = pos_probs[neg_mask]  # 负样本的正类概率 [num_neg,]
                    neg_preds_correct = (neg_pos_probs < 0.5).float()  # 正确预测负的比例
                    neg_acc = neg_preds_correct.mean().item()  # scalar
                    train_neg_acc += neg_acc * neg_mask.sum().item()  # 加权累积（乘以负样本数，避免小batch偏差）
                    total_neg_samples += neg_mask.sum().item()  # 累积总负样本数（需在循环外初始化为0）
                else:
                    neg_acc = 0.0  # 无负样本时

            optimizer.step()  #
            scheduler.step()

            # evidence = class_criterion.relu_evidence(tag)
            # # evidence2 = class_criterion.relu_evidence(tag2)
            # alpha = evidence + 1
            # # alpha2 = evidence2 + 1
            # prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
            # # prob2 = alpha2 / torch.sum(alpha2, dim=1, keepdim=True)
            # tag_hat = prob  # [:,1]
            # # tag2_hat = prob2

            # train_acc += (acc(tag1_hat, label) + acc(tag2_hat, label)) / 2
            # train_acc += (acc(tag1, label) + acc(tag2, label)) / 2
            # acc_train = post_pro(tag, point, label, end_point)
            acc_train = acc(tag_pred.sigmoid(), to_onehot(label))

            # acc_train, precision, recall, f1 = post_pro.evaluate(tag, label)
            # train_precision += precision
            # train_recall += recall
            # train_f1 += f1
            # train_acc += acc_train
            train_acc += acc_train
            torch.cuda.empty_cache()

        # epoch_end_time = time.time()
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch + 1}/{epochs}, Time: {epoch_time:.2f} seconds')

        writer.add_scalar('training full_loss',
                          f_losses / nb_batches_train,
                          epoch + 1)
        writer.add_scalar('training cls_loss',
                          cls_losses / nb_batches_train,
                          epoch + 1)
        # writer.add_scalar('training reg_loss',
        #                   reg_losses / nb_batches_train,
        #                   epoch + 1)
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
            f"| Train bv_loss {bv_losses / nb_batches_train}"
            f" | Train_Acc {train_acc / nb_batches_train} ")
        # print(
        #     f"Epoch {epoch + 1}: | Train_Loss {f_losses / nb_batches_train} | Train Cls_loss {cls_losses / nb_batches_train}"
        #     f" | Train l1_loss {l1_losses / nb_batches_train} | Train giou_loss {giou_losses / nb_batches_train}"
        #     f" | Train Distil_loss {distill_losses / nb_batches_train}"
        #     f" | Train_Acc {train_acc / nb_batches_train} ")
        avg_neg_acc = train_neg_acc / total_neg_samples
        print(f"Epoch {epoch + 1}: Avg Train neg_acc = {avg_neg_acc:.3f} (>0.7? {avg_neg_acc > 0.7})")
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('learning_rate', current_lr, epoch + 1)
        print(f"Current Learning Rate: {current_lr:.6f}")

        # valid_f_loss, valid_cls_loss, valid_reg_loss, val_acc, val_dis_loss = evaluate(model, valid_loader, class_criterion,
        #                                                                  reg_criterion, dis_loss, args, epoch)
        valid_f_loss, valid_cls_loss, valid_img_loss, valid_bv_loss, valid_d_loss, val_acc = evaluate(model, valid_loader, class_criterion, img_criterion, bv_criterion, dis_loss, epoch, use_dis)

        writer.add_scalar('validation full_loss',
                          valid_f_loss,
                          epoch + 1)
        writer.add_scalar('validation cls_loss',
                          valid_cls_loss,
                          epoch + 1)
        # writer.add_scalar('validation reg_loss',
        #                   valid_reg_loss,
        #                   epoch + 1)
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
        checkpoint_paths = []
        checkpoint_filepath = Path(checkpoint_filepath)
        if is_best:
            checkpoint_paths.append(checkpoint_filepath / 'checkpoint_best.pth')  # 主 checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_paths.append(checkpoint_filepath / f'checkpoint{epoch:04}.pth')  # 每 10 轮保存额外的备份

        # 实际保存所有指定 checkpoint 路径
        for checkpoint_path in checkpoint_paths:
            torch.save(checkpoint, checkpoint_path)

    print('save file times: ' + str(save_times) + '.\n')


def evaluate(model, val_data, class_criterion, img_criterion, bv_criterion, dis_loss, epoch, use_dis):
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

    print('in Validation...')
    with torch.no_grad():
        model.eval()
        val_acc = 0
        all_logits = []  # Collect all raw logits for distribution monitoring
        all_preds = []  # Collect all predictions (0 or 1)
        all_labels = []  # Collect all true labels for comparison
        total_samples = 0
        for batch in val_data:
            label = batch['label'][:, -1, :].to(device).float()
            bbox = batch['bbox'].to(device)
            vel = batch['vel'].to(device)
            img = batch['image'].to(device)

            # tag1, tag2, point1, point2, u_img, u_bv = model(img, bbox, vel)
            tag_pred, tag_i, tag_bv, u_img, u_bv, mu = model(img, bbox, vel, False)
            tag_pred = torch.nn.functional.dropout(tag_pred, p=0.2, training=False)
            # indices = get_indices(point, end_point, tag, label)
            # tag1, point1 = post_process(point1, end_point, tag1, label)
            # tag2, point2 = post_process(point2, end_point, tag2, label)

            # Compute probabilities and predictions for binary classification
            probs = torch.sigmoid(tag_pred)[:, 1]  # shape (bs,)
            preds = (probs > 0.5).float()  # shape (bs,), 0 or 1

            # Accumulate for monitoring
            all_logits.append(tag_pred.cpu())  # Keep as (bs, 2)
            all_preds.append(preds.cpu())
            all_labels.append(label.squeeze(-1).cpu())  # Assume label is (bs,1), flatten to (bs,)
            total_samples += len(preds)

            # point1 = project_normalized_bbox(point1, window)
            # point2 = project_normalized_bbox(point2, window)

            # val_cls_loss = class_criterion(tag, label, epoch, args.num_class, args.annealing_step)
            # val_cls_loss2 = class_criterion(tag2, label, epoch, args.num_class, args.annealing_step)
            #
            # val_reg_loss = reg_criterion(point, end_point)
            # val_reg_loss2 = reg_criterion(point2, end_point, epoch)

            val_cls_loss = class_criterion(tag_pred, to_onehot(label))
            val_img_loss = img_criterion(tag_i, label)
            val_bv_loss = bv_criterion(tag_bv, label, u_bv, mu)
            if use_dis:
                val_d_loss = (dis_loss(tag_i, tag_bv, to_onehot(label)) + dis_loss(tag_bv, tag_i, to_onehot(label))) / 2
            else:
                val_d_loss = 0.0

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
            # val_cls_loss = class_criterion(tag, label)

            # val_reg_loss, loss_bbox, loss_giou = reg_criterion(point, end_point, indices)
            # val_loss_img = binary_cross_entropy_with_logits(img_logit, label.argmax(dim=1).float())
            # val_reg_loss2, loss_bbox2, loss_giou2 = reg_criterion(point2, end_point)

            # val_distill_loss1 = (torch.abs(tag1 - tag2.detach()) * u_img).sum() / (u_img + 1e-7).sum()
            # val_distill_loss2 = (torch.abs(tag2 - tag1.detach()) * u_bv).sum() / (u_bv + 1e-7).sum()
            # val_distill_loss1 = torch.tensor(0)
            # val_distill_loss2 = torch.tensor(0)

            # val_distill_loss1 = dis_loss(tag1, tag2, label)
            # val_distill_loss2 = dis_loss(tag2, tag1, label)

            # f_loss = (2 * (val_cls_loss1 + val_cls_loss2) + (val_reg_loss1 + val_reg_loss2) + (val_distill_loss1 + val_distill_loss2))
            f_loss = val_cls_loss + 0.1 * val_img_loss + 0.7 * val_bv_loss + 0.5 * val_d_loss
            # f_loss = ((val_reg_loss1 + val_reg_loss2))
            # f_loss = ((val_cls_loss1 + val_cls_loss2) / (sigma_cls * sigma_cls) + (val_reg_loss1 + val_reg_loss2) /
            #           (sigma_reg * sigma_reg) + (val_distill_loss1 + val_distill_loss2) / (sigma_dis * sigma_dis) +
            #           torch.log(sigma_cls) + torch.log(sigma_reg) + torch.log(sigma_dis))

            # f_loss = ((val_cls_loss1 + val_cls_loss2) / (sigma_cls * sigma_cls) + (val_reg_loss1 + val_reg_loss2) /
            #           (sigma_reg * sigma_reg) + torch.log(sigma_cls) + torch.log(sigma_reg))

            val_f_losses += f_loss.item()
            val_cls_losses += val_cls_loss.item()
            val_img_losses += val_img_loss.item()
            val_bv_losses += val_bv_loss.item()
            if use_dis:
                val_d_losses += val_d_loss.item()

            # evidence = class_criterion.relu_evidence(tag)
            # # evidence2 = class_criterion.relu_evidence(tag2)
            # alpha = evidence + 1
            # # alpha2 = evidence2 + 1
            # prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
            # # prob2 = alpha2 / torch.sum(alpha2, dim=1, keepdim=True)
            # tag_hat = prob  # [:,1]
            # # tag2_hat = prob2

            # val_acc += acc(tag, label)

            metrics = post_pro.evaluate(tag_pred, label)
            val_precision += metrics[1]
            val_recall += metrics[2]
            val_f1 += metrics[3]
            val_acc += metrics[0]

    print(
        f'Valid_Full_Loss {val_f_losses / nb_batches} | Valid Cls_loss {val_cls_losses / nb_batches} | Valid img_loss '
        f'{val_img_losses / nb_batches} | Valid bv_loss {val_bv_losses / nb_batches} | \n'
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
    print(f"Logits mean (pos class): {all_logits[:, 1].mean():.4f}, std: {all_logits[:, 1].std():.4f}")
    print(
        f"Probs mean (pos class, last batch): {probs.mean():.4f}, std: {probs.std():.4f}"
    )  # For full: torch.softmax(all_logits, dim=1)[:, 1].mean()

    # Optional: Entropy of average prediction distribution (low entropy = collapsed to one class)
    avg_probs_dist = torch.softmax(all_logits, dim=1).mean(0)  # Average softmax dist over samples [2]
    entropy = -sum(p * torch.log(p + 1e-8) for p in avg_probs_dist if p > 0)
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


def balance_dataset(dataset, flip=True): # 数据集平衡
    d = {'bbox': dataset['bbox'].copy(),
         'pid': dataset['pid'].copy(),
         'activities': dataset['activities'].copy(),
         'image': dataset['image'].copy(),
         'center': dataset['center'].copy(),
         'obd_speed': dataset['obd_speed'].copy(),
         'gps_speed': dataset['gps_speed'].copy(),
         'image_dimension': (1920, 1080)}
    gt_labels = [gt[0] for gt in d['activities']] # 标签
    num_pos_samples = np.count_nonzero(np.array(gt_labels)) # 正样本数
    num_neg_samples = len(gt_labels) - num_pos_samples # 负样本数

    if num_neg_samples == num_pos_samples: # 正负样本数相等
        print('Positive samples is equal to negative samples.')
    else: # 正负样本数不相等
        print('Unbalanced: \t Postive: {} \t Negative: {}'.format(num_pos_samples, num_neg_samples))
        if num_neg_samples > num_pos_samples:
            gt_augment = 1 # 正样本数大于负样本数，增加负样本
        else:
            gt_augment = 0 # 负样本数大于正样本数，增加正样本

        img_width = d['image_dimension'][0] # 图片宽度
        num_samples = len(d['pid']) # 样本数

        for i in range(num_samples): # 遍历样本
            if d['activities'][i][0][0] == gt_augment: # 标签与增加的标签相同
                flipped = d['center'][i].copy() # 中心点
                flipped = [[img_width - c[0], c[1]] for c in flipped] # 水平翻转
                d['center'].append(flipped) # 添加到中心点

                flipped = d['bbox'][i].copy() # 边界框
                flipped = [np.array([img_width - c[2], c[1], img_width - c[0], c[3]]) for c in flipped] # 水平翻转
                d['bbox'].append(flipped) # 添加到边界框

                d['pid'].append(dataset['pid'][i].copy()) # 添加pid

                d['activities'].append(d['activities'][i].copy()) # 添加标签
                d['gps_speed'].append(d['gps_speed'][i].copy()) # 添加gps速度
                d['obd_speed'].append(d['obd_speed'][i].copy()) # 添加obd速度

                flipped = d['image'][i].copy() # 图片
                flipped = [c.replace('.png', '_flip.png') for c in flipped] # 水平翻转

                d['image'].append(flipped) # 添加图片

        gt_labels = [gt[0] for gt in d['activities']] # 标签
        num_pos_samples = np.count_nonzero(np.array(gt_labels)) # 正样本数
        num_neg_samples = len(gt_labels) - num_pos_samples # 负样本数

        if num_neg_samples > num_pos_samples: # 负样本数大于正样本数
            rm_index = np.where(np.array(gt_labels) == 0)[0] # 删除负样本
        else:
            rm_index = np.where(np.array(gt_labels) == 1)[0] # 删除正样本

        dif_samples = abs(num_neg_samples - num_pos_samples) # 正负样本数差值

        np.random.seed(42)
        np.random.shuffle(rm_index) # 打乱索引
        rm_index = rm_index[0:dif_samples] # 间隔删除

        for k in d: # 遍历数据
            seq_data_k = d[k] # 数据
            d[k] = [seq_data_k[i] for i in range(0, len(seq_data_k)) if i not in rm_index] # 删除数据

        new_gt_labels = [gt[0] for gt in d['activities']] # 新标签
        num_pos_samples = np.count_nonzero(np.array(new_gt_labels)) # 新正样本数
        print('Balanced: Postive: %d \t Negative: %d \n' % (num_pos_samples, len(d['activities']) - num_pos_samples))
        print('Total Number of samples: %d\n' % (len(d['activities'])))

    return d


def tte_dataset(dataset, time_to_event, overlap, obs_length): # 时间到事件数据集
    d_obs = {'bbox': dataset['bbox'].copy(), 
             'pid': dataset['pid'].copy(),
             'activities': dataset['activities'].copy(),
             'image': dataset['image'].copy(),
             'gps_speed': dataset['gps_speed'].copy(),
             'obd_speed': dataset['obd_speed'].copy(),
             'center': dataset['center'].copy()
             }

    d_tte = {'bbox': dataset['bbox'].copy(),
             'pid': dataset['pid'].copy(),
             'activities': dataset['activities'].copy(),
             'image': dataset['image'].copy(),
             'gps_speed': dataset['gps_speed'].copy(),
             'obd_speed': dataset['obd_speed'].copy(),
             'center': dataset['center'].copy()}

    if isinstance(time_to_event, int):
        for k in d_obs.keys():
            for i in range(len(d_obs[k])):
                d_obs[k][i] = d_obs[k][i][- obs_length - time_to_event: -time_to_event] # 观察长度
                d_tte[k][i] = d_tte[k][i][- time_to_event:] # 时间到事件
        d_obs['tte'] = [[time_to_event]] * len(dataset['bbox']) # 观察长度
        d_tte['tte'] = [[time_to_event]] * len(dataset['bbox']) # 时间到事件

    else: # 时间到事件为列表
        olap_res = obs_length if overlap == 0 else int((1 - overlap) * obs_length) # 重叠长度
        olap_res = 1 if olap_res < 1 else olap_res # 重叠长度

        for k in d_obs.keys(): # 遍历数据
            seqs = []
            seqs_tte = []
            for seq in d_obs[k]:
                start_idx = len(seq) - obs_length - time_to_event[1] # 开始索引
                end_idx = len(seq) - obs_length - time_to_event[0] # 结束索引 
                seqs.extend([seq[i:i + obs_length] for i in range(start_idx, end_idx, olap_res)]) # 观察长度
                seqs_tte.extend([seq[i + obs_length:] for i in range(start_idx, end_idx, olap_res)]) # 时间到事件
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
        time_to_event_0 = time_to_event[0] # 时间到事件
    except:
        time_to_event_0 = time_to_event # 时间到事件 
    for seq_index, (seq_obs, seq_tte) in enumerate(zip(d_obs['bbox'], d_tte['bbox'])): # 遍历数据
        if len(seq_obs) < obs_length or len(seq_tte) < time_to_event_0: # 观察长度小于16或时间到事件小于时间到事件
            remove_index.append(seq_index) # 删除索引

    for k in d_obs.keys():
        for j in sorted(remove_index, reverse=True): # 倒序删除
            del d_obs[k][j]
            del d_tte[k][j]

    return d_obs, d_tte


def normalize_bbox(dataset, width=1920, height=1080): # 归一化边界框
    normalized_set = []
    for sequence in dataset:
        if sequence == []:
            continue
        normalized_sequence = []
        for bbox in sequence:
            np_bbox = np.zeros(4)
            np_bbox[0] = bbox[0] / width # 左上角x 
            np_bbox[2] = bbox[2] / width # 右下角x 
            np_bbox[1] = bbox[1] / height # 左上角y 
            np_bbox[3] = bbox[3] / height # 右下角y 
            normalized_sequence.append(np_bbox)
        normalized_set.append(np.array(normalized_sequence))

    return normalized_set

def normalize_traj(dataset, width=1920, height=1080): # 归一化轨迹
    normalized_set = []
    for sequence in dataset:
        if sequence == []:
            continue
        normalized_sequence = []
        for bbox in sequence:
            np_bbox = np.zeros(4)
            np_bbox[0] = bbox[0]# / width
            np_bbox[2] = bbox[2]# / width
            np_bbox[1] = bbox[1]# / height
            np_bbox[3] = bbox[3]# / height
            normalized_sequence.append(np_bbox)
        normalized_set.append(np.array(normalized_sequence))

    return normalized_set


def prepare_label(dataset): # 准备标签
    labels = np.zeros(len(dataset), dtype='int64')
    for step, action in enumerate(dataset):
        if action == []:
            continue
        labels[step] = action[0][0]

    return labels

def pad_sequence(inp_list, max_len): # 填充序列
    padded_sequence = []
    for source in inp_list:
        target = np.array([source[0]] * max_len) # 填充序列
        source = source 
        target[-source.shape[0]:, :] = source # 填充序列
        
        padded_sequence.append(target)
        
    return padded_sequence

def to_onehot(tensor_data):
    y = tensor_data.squeeze()  # 假设 tensor_data 是 shape (bs, 1) 的 tensor，squeeze 后为 (bs,)
    y_mat = torch.zeros((len(y), 2), device=tensor_data.device, dtype=tensor_data.dtype)

    y_mat[y == 1, 1] = 1
    y_mat[y == 0, 0] = 1

    return y_mat


def apply_smote(label_train, bbox_train, vel_train, img_path_train, sampling_strategy=2.0):
    # **转换 & 检查 labels**
    labels_np = np.array(label_train, dtype=int)  # temp np for SMOTE
    img_path_train_list = list(img_path_train)  # 保持 list of lists
    print(f"Labels unique: {np.unique(labels_np, return_counts=True)}")

    n_pos = np.sum(labels_np == 1)
    if n_pos == 0:
        print("WARNING: No positive samples (label=1)! Skipping SMOTE, using original data.")
        return label_train, bbox_train, vel_train, img_path_train  # 返回原 tensor/list

    # **动态 feat_dim**
    N, T, bbox_dim = bbox_train.shape
    _, _, vel_dim = vel_train.shape
    feat_dim = T * (bbox_dim + vel_dim)

    feats = np.zeros((N, feat_dim))
    for i in range(N):
        bbox_feat = bbox_train[i].flatten()
        vel_feat = vel_train[i].flatten()
        feats[i] = np.concatenate([bbox_feat, vel_feat])

    # **SMOTE**
    n_neg = np.sum(labels_np == 0)
    n_pos_target = int(n_neg * sampling_strategy)
    target_strategy = {1: n_pos_target}

    k = min(5, n_pos - 1)
    print(f"n_neg: {n_neg}, n_pos_target: {n_pos_target}, k_neighbors: {k}")

    smote = SMOTE(sampling_strategy=target_strategy, random_state=42, k_neighbors=k)
    feats_res, labels_res_np = smote.fit_resample(feats, labels_np)

    # rus = RandomUnderSampler(sampling_strategy=0.8, random_state=42)  # 负数 = 正数
    # feats_bal, labels_bal = rus.fit_resample(feats_res, labels_res_np)
    # print(f"After RUS: 正 {np.sum(labels_bal == 1)}, 负 {np.sum(labels_bal == 0)}")  # 平衡 ~1:1

    # **修复：重建 bbox/vel (tile)**
    num_new = len(labels_res_np) - N
    repeat_factor = num_new // n_pos + 1
    pos_mask = labels_np == 1
    bbox_pos = np.repeat(bbox_train[pos_mask], repeat_factor, axis=0)[:num_new]
    vel_pos = np.repeat(vel_train[pos_mask], repeat_factor, axis=0)[:num_new]

    bbox_res_np = np.concatenate([bbox_train, bbox_pos], axis=0)
    vel_res_np = np.concatenate([vel_train, vel_pos], axis=0)

    # **img_path_res (list of lists)**
    pos_indices = np.where(labels_np == 1)[0]
    img_path_pos = [img_path_train[i] for i in pos_indices]
    num_pos_new = np.sum(labels_res_np == 1) - n_pos
    img_path_new_pos = [random.choice(img_path_pos) for _ in range(num_pos_new)]
    img_path_res = img_path_train + img_path_new_pos  # list of lists

    # **修复：转回 torch.tensor (匹配原类型/形状, dtype=float32)**
    label_train_res = torch.from_numpy(labels_res_np).float() # [N_new, 1], float32
    bbox_train_res = torch.from_numpy(bbox_res_np).float()  # [N_new, T, bbox_dim]
    vel_train_res = torch.from_numpy(vel_res_np).float()  # [N_new, T, vel_dim]

    print(f"Original: 正 {n_pos}, 负 {n_neg}")
    print(f"SMOTE after: 正 {np.sum(labels_res_np == 1)}, 负 {np.sum(labels_res_np == 0)}")
    print(f"label_train_res shape/type: {label_train_res.shape}, {label_train_res.dtype}")

    return label_train_res, bbox_train_res, vel_train_res, img_path_res  # tensor + list