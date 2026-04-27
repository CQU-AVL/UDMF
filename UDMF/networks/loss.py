import math
from torch import digamma
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torchmetrics import Accuracy
from torchvision.ops import generalized_box_iou

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
acc = Accuracy(task="binary", num_classes=2).to(device)


class EDLLOSS(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target, epoch_num, num_classes, annealing_step, device=None):
        return self.edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, device)

    def get_device(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        return device

    def one_hot_embedding(self, labels, num_classes=10):
        # Convert to One Hot Encoding
        y = torch.eye(num_classes)
        return y[labels]

    def relu_evidence(self, y):
        return F.relu(y)

    def exp_evidence(self, y):
        return torch.exp(torch.clamp(y, -10, 10))

    def softplus_evidence(self, y):
        return F.softplus(y)

    def kl_divergence(self, alpha, num_classes, device=None):
        if not device:
            device = self.get_device()
        ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
        sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
        first_term = (
                torch.lgamma(sum_alpha)
                - torch.lgamma(alpha).sum(dim=1, keepdim=True)
                + torch.lgamma(ones).sum(dim=1, keepdim=True)
                - torch.lgamma(ones.sum(dim=1, keepdim=True))
        )
        second_term = (
            (alpha - ones)
            .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
            .sum(dim=1, keepdim=True)
        )
        kl = first_term + second_term
        return kl

    def loglikelihood_loss(self, y, alpha, device=None):
        if not device:
            device = self.get_device()
        y = y.to(device)
        alpha = alpha.to(device)
        S = torch.sum(alpha, dim=1, keepdim=True)
        loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
        loglikelihood_var = torch.sum(
            alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
        )
        loglikelihood = loglikelihood_err + loglikelihood_var
        return loglikelihood

    def mse_loss(self, y, alpha, epoch_num, num_classes, annealing_step, device=None):
        if not device:
            device = self.get_device()
        y = y.to(device)
        alpha = alpha.to(device)
        loglikelihood = self.loglikelihood_loss(y, alpha, device=device)

        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
        )

        kl_alpha = (alpha - 1) * (1 - y) + 1
        kl_div = annealing_coef * self.kl_divergence(kl_alpha, num_classes, device=device)
        return loglikelihood + kl_div

    def edl_loss(self, func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
        y = y.to(device)
        alpha = alpha.to(device)
        S = torch.sum(alpha, dim=1, keepdim=True)

        A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
        )

        kl_alpha = (alpha - 1) * (1 - y) + 1
        kl_div = annealing_coef * self.kl_divergence(kl_alpha, num_classes, device=device)
        return A + kl_div

    def edl_mse_loss(self, output, target, epoch_num, num_classes, annealing_step, device=None):
        if not device:
            device = self.get_device()
        evidence = self.relu_evidence(output)
        alpha = evidence + 1
        loss = torch.mean(
            self.mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
        )
        return loss

    def edl_log_loss(self, output, target, epoch_num, num_classes, annealing_step, device=None):
        if not device:
            device = self.get_device()
        evidence = self.relu_evidence(output)
        alpha = evidence + 1
        loss = torch.mean(
            self.edl_loss(
                torch.log, target, alpha, epoch_num, num_classes, annealing_step, device
            )
        )
        return loss

    def edl_digamma_loss(self,
                         output, target, epoch_num, num_classes, annealing_step, device=None
                         ):
        if not device:
            device = self.get_device()
        evidence = self.relu_evidence(output)
        alpha = evidence + 1
        loss = torch.mean(
            self.edl_loss(
                torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device
            )
        )
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0,
                 reduction: str = 'mean', eps: float = 1e-8):
        super().__init__()
        self.alpha = alpha  # 偏正类，提升少数类 (crossing) 学习
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps  # 防 probs 精确 0/1

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        inputs: Tensor[B], raw logits
        targets: Tensor[B], binary labels (0 or 1)
        """
        # BCE 支持 pos_weight
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction="none")

        # Probs
        prob = torch.sigmoid(inputs)

        # p_t: 正确类概率，clamp 防 0/1
        p_t = prob * targets + (1 - prob) * (1 - targets)
        p_t = p_t.clamp(self.eps, 1 - self.eps)

        # Focal 权重
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        # Alpha 平衡 (tensor 版，提升数值稳)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        # Reduction
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        else:
            loss = loss  # 'none'

        # 最终防 NaN (下游累加安全)
        return loss


class cross_entropy_loss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.class_weights = class_weights

    def forward(self, inputs, targets, indices=None):
        class_targets = targets.argmax(dim=1)  # shape: [bs]

        # Move class_weights to same device
        if self.class_weights is not None:
            class_weights = self.class_weights
            loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
        else:
            loss_fn = nn.CrossEntropyLoss(reduction='mean')

        loss = loss_fn(inputs, class_targets)
        return loss


def _get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


class Distill_Loss(nn.Module):
    def __init__(self, temperature=2.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, tag1, tag2, u):
        # tag1, tag2: [B, 2] logits and targets (binary cross-entropy per sample over 2 classes/dims)
        # u: [B, T, D] weights (e.g., uncertainty or attention; will average over T and D to match [B, 2])
        if len(u.shape) == 3:
            B, T, D = u.shape  # Extract shapes
            u = torch.clamp(u, min=1e-6)
            u_avg = u.mean(dim=(1, 2)).unsqueeze(-1).expand(-1, 2)  # [B, 1] -> [B, 2] (mean over T and D)
        else:
            B, T = u.shape
            u = torch.clamp(u, min=1e-6)
            u_avg = u.mean(dim=(-1)).unsqueeze(-1).expand(-1, 2)  # [B, 1] -> [B, 2] (mean over T and D)

        # Compute per-element BCE loss: [B, 2]
        p1 = nn.functional.log_softmax(tag1 / self.temperature, dim=-1)  # 学生 log probs
        p2 = nn.functional.softmax(tag2 / self.temperature, dim=-1)  # 教师 probs
        kl = nn.functional.kl_div(p1, p2, reduction='none').sum(dim=-1) # [B]
        # Average u over T and D to get per-sample weights [B, 2] (broadcast if needed)
        # Assuming u's D dimension aligns with the 2 in tag1/tag2; if not, adjust averaging dims


        # Weighted loss: element-wise multiply [B, 2] * [B, 2], then sum and normalize
        weighted_loss = (kl * u_avg.mean(-1)).mean()
        return weighted_loss * (self.temperature ** 2)


class BeliefMatchingLoss(nn.Module):
    def __init__(self, coeff=0.5, prior=1.0, num_classes=2):
        super(BeliefMatchingLoss, self).__init__()
        self.prior = prior
        self.coeff = coeff
        self.num_classes = num_classes  # 二分类: K=2

    def forward(self, logits, ys):
        # 形状/类型检查与修复 (二分类: K=2)
        if logits.dim() != 2:
            raise ValueError(f"logits must be 2D [N, K=2], got {logits.shape}")
        N, K = logits.shape
        if K != self.num_classes:
            raise ValueError(f"For binary classification, K must be 2, got {K}")
        ys = ys.long()  # 确保 int64 (0 or 1)
        if ys.shape != (N,):
            ys = ys.view(-1)  # 展平 [N]
            if ys.shape[0] != N:
                raise ValueError(f"ys must have N={N} elements, got {ys.shape}")
        if ys.max() > 1 or ys.min() < 0:
            raise ValueError("For binary classification, ys must be 0 or 1")

        alphas = torch.exp(logits)
        betas = self.prior * torch.ones_like(logits)

        # compute log-likelihood loss: psi(alpha_target) - psi(alpha_zero)
        a_ans = torch.gather(alphas, -1, ys.unsqueeze(-1)).squeeze(-1)
        a_zero = torch.sum(alphas, -1)
        ll_loss = digamma(a_ans) - digamma(a_zero)
        ll_loss = torch.clamp(ll_loss, min=-5.0, max=5.0)

        # compute kl loss: loss1 + loss2
        loss1 = torch.lgamma(a_zero) - torch.sum(torch.lgamma(alphas), -1)
        loss2 = torch.sum(
            (alphas - betas) * (digamma(alphas) - digamma(a_zero.unsqueeze(-1))),
            -1)
        kl_loss = loss1 + loss2

        return ((self.coeff * kl_loss - ll_loss)).mean()


class COLOSS(nn.Module):
    def __init__(self, lambda_cov=0.65, lambda_r=1.7):
        super().__init__()
        self.lambda_cov = lambda_cov
        self.lambda_r = lambda_r
        self.mse_loss = nn.MSELoss(reduction='none')  # Per-sample MSE for regression

    def forward(self, y_va, y_target, sigma_sq_v, mu_v):
        # y_va: [B, 2] fused regression output (e.g., intent score)
        # y_target: [B, 2] target intent values
        # sigma_sq_v: [B, T, D] variance (MVN, T=1 for single-frame or adjust), D = d_model
        # mu_v: [B, T, D] mean (matching sigma_sq_v)
        B, T, D = sigma_sq_v.shape  # D = d_model

        # Distance D: Prediction error (MSE per-sample, scalar [B] by averaging over the 2 dims)
        d_flat = ((y_va - y_target).pow(2)).mean(dim=-1)  # [B] (mean MSE over the 2 intent dimensions)
        d_v = d_flat.unsqueeze(1).unsqueeze(-1).expand(-1, T,
                                                       D)  # [B, T, D] (broadcast scalar error to match sigma_sq_v)

        # S: 1 / ||σ²||_2 [B, T]
        norm_v = torch.norm(sigma_sq_v, p=2, dim=-1, keepdim=False)  # [B, T]
        s_v = 1 / (norm_v + 1e-7)  # [B, T]

        # L_COV: Intra V (softmax KL, per-batch over T)
        # Average d_v over D for softmax input [B, T] (will be the scalar error repeated over T)
        d_v_avg = d_v.mean(-1)  # [B, T]
        p_d_v = F.softmax(d_v_avg, dim=-1)  # Softmax over T (temporal ranking)
        p_s_v = F.softmax(s_v, dim=-1)
        l_cov = F.kl_div(p_s_v.log(), p_d_v, reduction='batchmean') + \
                F.kl_div(p_d_v.log(), p_s_v, reduction='batchmean')

        # L_regu: Variance regularization (per-element)
        l_reg_v = -0.5 * (1 + torch.log(sigma_sq_v+ 1e-7) - mu_v.pow(2) - sigma_sq_v).mean()
        l_regu = l_reg_v

        # Total loss (no L_emo in original, but add for regression task if needed)
        loss = self.lambda_cov * l_cov + self.lambda_r * l_regu
        return loss