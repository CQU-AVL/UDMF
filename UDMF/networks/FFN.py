from collections import Counter

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, precision_score, recall_score, f1_score
from torch import nn




class FFN(nn.Module): # 前馈网络
    def __init__(self, d_model, hidden_dim, rate=0.3, layer_norm_eps=1e-5):
        super(FFN, self).__init__()

        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps) # 归一化
        self.linear1 = nn.Linear(d_model, hidden_dim) # 线性层 
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(rate)
        self.linear2 = nn.Linear(hidden_dim, d_model)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x):
        y = self.linear2(self.dropout1(self.relu(self.linear1(x)))) # 前馈网络
        out = x + self.dropout2(y) 
        out = self.norm(out) # 归一化
        return out


class post_process(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, gt_labels, threshold=0.5):
        """
        logits: Tensor [B,2] or [B] (raw logits)
        gt_labels: Tensor [B,1] or [B] (0/1)
        returns: dict of metrics
        """
        # squeeze to 1D
        pos_logits = logits.squeeze(-1).detach().cpu()  # [B], positive class logits

        true_t = gt_labels.squeeze(-1).detach().cpu()

        # probabilities as numpy 1D (sigmoid on positive logits)
        prob = torch.sigmoid(pos_logits).numpy() if isinstance(pos_logits, torch.Tensor) else pos_logits
        true_binary = true_t.numpy() if isinstance(true_t, torch.Tensor) else true_t

        pred_binary = (prob > threshold).astype(int)
        pred_scores = prob

        metrics = {}
        if pred_binary.size > 0:
            metrics['acc'] = float(accuracy_score(true_binary, pred_binary))
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_binary, pred_binary, average='binary', zero_division=0
            )
            metrics['precision'] = float(precision)
            metrics['recall'] = float(recall)
            metrics['f1'] = float(f1)
            metrics['auroc'] = float(roc_auc_score(true_binary, pred_scores)) if len(np.unique(true_binary)) > 1 else 0.0
        else:
            metrics = {'acc': 0.0, 'f1': 0.0, 'auroc': 0.0, 'precision': 0.0, 'recall': 0.0}

        return metrics

    def evaluate(self, logits, gt_labels):
        # 验证阶段使用
        return self.compute_accuracy_with_threshold(logits, gt_labels)

    def compute_accuracy_with_threshold(self, logits, gt_labels, threshold=0.2):
        """
        logits: Tensor [B,2] or [B]
        gt_labels: Tensor [B,1] or [B]
        returns: acc, precision, recall, f1 (floats)
        """
        # ensure tensors
        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits)
        if not isinstance(gt_labels, torch.Tensor):
            gt_labels = torch.tensor(gt_labels)

        pos_logits = logits # Assume index 1 is positive class
        prob = torch.sigmoid(pos_logits)
        pred_labels = (prob > threshold).long().view(-1)
        true_labels = gt_labels.long().squeeze(-1).view(-1)

        acc = (pred_labels == true_labels).float().mean().item()

        # convert to numpy for sklearn metrics
        y_true = true_labels.cpu().numpy()
        y_pred = pred_labels.cpu().numpy()

        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        return acc, precision, recall, f1
