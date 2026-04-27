import math

import numpy as np
import torch
from torch import nn
from torch.nn.init import constant_, xavier_uniform_
import torch.nn.functional as F
# import MultiScaleDeformableAttention as MSDA
from torch.autograd import Function
from torch.autograd.function import once_differentiable


class TemporalAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(args.d_model, args.d_model // 4, bias=False),
            nn.ReLU(),
            nn.Linear(args.d_model // 4, args.d_model, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, b_v):
        b_v1 = self.mlp(self.avg_pool(b_v).squeeze(-1))
        b_v2 = self.mlp(self.max_pool(b_v).squeeze(-1))
        temp = b_v * self.sigmoid((b_v1 + b_v2).unsqueeze(-1))

        return temp.transpose(1, 2)


class SpatialAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.args = args
        self.conv2 = nn.Conv1d(args.d_model, args.d_model, 3, 1, 1)

    def forward(self, temp):
        avg_pool = torch.mean(temp, dim=-1, keepdim=True)  # [batchsize, 32, 1]
        max_pool, _ = torch.max(temp, dim=-1, keepdim=True)  # [batchsize, 32, 1]
        pool = torch.cat([avg_pool, max_pool], dim=-1)  # [batchsize, 32, 2]
        pool = pool.transpose(1, 2)
        attention = self.sigmoid(self.conv1(pool)).transpose(1, 2)  # [batchsize, 32, 1]

        out = self.conv2((temp * attention).transpose(1, 2)).transpose(1, 2)
        return out


class TempSpatioAttention(nn.Module):
    def __init__(self, args, dropout=0.1, d_model=256, dim_feedforward=2048):
        super().__init__()
        self.temp_atten = TemporalAttention(args)
        self.spatio_atten = SpatialAttention(args)
        self.dense = nn.Sequential(
            nn.BatchNorm1d(args.d_model), nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=args.d_model, out_channels=args.d_model // 4, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm1d(args.d_model // 4), nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=args.d_model // 4, out_channels=args.d_model, kernel_size=3, stride=1, padding=1,
                      bias=False)
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.num_layer = 2

    def forward(self, b_v):
        out = b_v
        for i in range(self.num_layer):
            out = self.single_layer(out)
        return out.transpose(1, 2)

    def single_layer(self, b_v):
        b_v = self.dense(b_v)
        temp = self.temp_atten(b_v)
        feat = self.spatio_atten(temp)

        src = b_v.transpose(1, 2) + self.dropout1(feat)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src.transpose(1, 2)


class AttentionBlocks(nn.Module):
    def __init__(self, d_model, num_heads, rate=0.3, layer_norm_eps=1e-5):
        super(AttentionBlocks, self).__init__()

        self.att = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True) # 多头注意力
        self.drop = nn.Dropout(rate)
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps) # 归一化

    def forward(self, x, y=None):
        y = x if y is None else y # 如果y为空，则y=x
        att_out, att_w = self.att(x, y, y) # 多头注意力
        att_out = self.drop(att_out) # dropout
        y = self.norm(x + att_out) # 归一化
        return y


class iAFF(nn.Module):
    def __init__(self, channels, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa.transpose(1, 2))
        xg = self.global_att(xa.transpose(1, 2))
        xlg = (xl + xg).transpose(1, 2)
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att(xi.transpose(1, 2))
        xg2 = self.global_att(xi.transpose(1, 2))
        xlg2 = (xl2 + xg2).transpose(1, 2)
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo


class dAFF(nn.Module):
    def __init__(self, channels, r, d):
        super().__init__()
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),  # input=output=[N,C,L]
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),  # input=output=[N,C,L]
        )

        self.sigmoid = nn.Sigmoid()
        self.d = d

    def forward(self, x, y):
        z1 = self.aff(x, y)
        result = [[z1]]
        for i in range(self.d - 1):
            zi = result[i]
            if isinstance(zi, list):
                cat_result = [x] + zi + [y]
            else:
                cat_result = [x] + [zi] + [y]
            # cat_result.insert(1, zi)
            result_i = []
            for j in range(len(cat_result) - 1):
                zij = self.aff(cat_result[j], cat_result[j + 1])
                result_i.append(zij)
            result.append(result_i)
        out = torch.stack(result[-1], dim=0)
        u = out.var(dim=0)
        tanh = nn.Tanh()
        output = out * (1 - tanh(u))

        return output.mean(dim=0)

    def aff(self, x, y):
        xa = x + y
        xl = self.local_att(xa.transpose(1, 2))
        xg = self.global_att(xa.transpose(1, 2))
        xlg = (xl + xg).transpose(1, 2)
        wei = self.sigmoid(xlg)
        atten = x * wei + y * (1 - wei)
        return atten


class SEfusion(nn.Module):
    def __init__(self, channel, r):
        super().__init__()
        inter_channel = channel // r
        self.gap = nn.AdaptiveAvgPool1d(1)
        # self.conv1 = nn.Linear(channel, inter_channel, bias=False)

        self.fc1 = nn.Sequential(
            nn.Linear(channel, inter_channel),
            nn.Dropout(p=0.5),  # 在全连接层后添加
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(inter_channel, channel),
            nn.Dropout(p=0.5),  # 在全连接层后添加
            nn.ReLU(),
        )
        # self.conv2 = nn.Linear(inter_channel, channel, bias=False)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.drop_out = nn.Dropout(p=0.5)

    def forward(self, image, bv, mod):
        if mod == "img":
            bv = self.gap(bv.permute(0, 2, 1)).transpose(1, 2)
            # bv = self.conv1(bv)
            bv = self.fc1(bv)
            # bv = self.conv2(bv)
            bv = self.fc2(bv)
            bv = self.sig(bv).transpose(1, 2)
            fus_img = image * bv
            return fus_img
        else:
            b, c, hw = image.shape
            img = self.gap(image).view(b, c)
            # img = self.conv1(img)
            img = self.fc1(img)
            # img = self.conv2(img)
            img = self.fc2(img)
            img = self.sig(img).view(b, 1, c)
            fus_bv = bv * img
            return fus_bv


class Time_att(nn.Module): # 在时间维度上进行注意力
    def __init__(self, dims):
        super(Time_att, self).__init__()
        self.linear1 = nn.Linear(dims, dims, bias=False)
        self.linear2 = nn.Linear(dims, 1, bias=False)
        self.time = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        y = self.linear1(x.contiguous())
        y = self.linear2(torch.tanh(y))
        beta = F.softmax(y, dim=-1)
        c = beta * x
        return self.time(c.transpose(-1, -2)).transpose(-1, -2).contiguous().squeeze()


class DEFORM_ATTEN(nn.Module):
    def __init__(self, d_model, n_heads, n_points):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points
        self.im2col_step = 64

        self.v_prj = nn.Linear(d_model, d_model)
        self.sample_offset = nn.Linear(d_model, n_heads * n_points * 2)
        self.atten_weight = nn.Linear(d_model, n_heads * n_points)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()  # 生成初始化的偏置位置 + 注意力权重初始化

    def _reset_parameters(self):
        # 生成初始化的偏置位置 + 注意力权重初始化
        constant_(self.sample_offset.weight.data, 0.)  # 初始化权重为0
        # [8, ]  0, pi/4, pi/2, 3pi/4, pi, 5pi/4, 3pi/2, 7pi/4
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        # [8, 2]
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        # [n_heads, n_levels, n_points, xy] = [8, 4, 4, 2]   [0]是值，[1]是索引👇
        grid_init = ((grid_init / grid_init.abs().max(-1, keepdim=True)[0])
                     .view(self.n_heads, 1, 2).repeat(1, self.n_points, 1))
        # 同一特征层中不同采样点的坐标偏移肯定不能够一样  因此这里需要处理
        # 对于第i个采样点，在8个头部和所有特征层中，其坐标偏移为：
        # (i,0) (i,i) (0,i) (-i,i) (-i,0) (-i,-i) (0,-i) (i,-i)   1<= i <= n_points
        # 从图形上看，形成的偏移位置相当于3x3正方形卷积核 去除中心 中心是参考点
        for i in range(self.n_points):
            grid_init[:, i, :] *= i + 1
        with torch.no_grad():
            # 把初始化的偏移量的偏置bias设置进去  不计算梯度
            self.sample_offset.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.atten_weight.weight.data, 0.)
        constant_(self.atten_weight.bias.data, 0.)
        xavier_uniform_(self.v_prj.weight.data)  # 均匀初始化权重
        constant_(self.v_prj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_padding_mask=None):
        N, Len_q, _ = query.shape  # bs   query length(每张图片所有特征点的数量)
        N, Len_in, _ = input_flatten.shape  # bs   query length(每张图片所有特征点的数量)
        assert (input_spatial_shapes[0] * input_spatial_shapes[1]) == Len_in

        # value = w_v * x  通过线性变换将输入的特征图变换成value  [bs, Len_q, 256] -> [bs, Len_q, 256]
        value = self.v_prj(input_flatten)
        # 将特征图mask过的地方（无效地方）的value用0填充(True的地方)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        # 把value拆分成8个head      [bs, Len_q, 256] -> [bs, Len_q, 8, 32]
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        # 预测采样点的坐标偏移  [bs,Len_q,256] -> [bs,Len_q,256] -> [bs, Len_q, n_head, n_level, n_point, 2] = [bs, Len_q, 8, 4, 4, 2]
        sampling_offsets = self.sample_offset(query).view(N, Len_q, self.n_heads, self.n_points, 2)
        # 预测采样点的注意力权重  [bs,Len_q,256] -> [bs,Len_q, 128] -> [bs, Len_q, 8, 4*4]
        attention_weights = self.atten_weight(query).view(N, Len_q, self.n_heads, self.n_points)
        # 每个query在每个注意力头部内，每个特征层都采样4个特征点，即16个采样点(4x4),再对这16个采样点的注意力权重进行初始化
        # [bs, Len_q, 8, 16] -> [bs, Len_q, 8, 16] -> [bs, Len_q, 8, 4, 4]
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_points)
        """初始化权重有什么用？👆"""
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:  # one stage
            # [4, 2]  每个(h, w) -> (w, h)
            offset_normalizer = torch.tensor([input_spatial_shapes[1], input_spatial_shapes[0]],
                                             device=input_spatial_shapes.device)
            # [bs, Len_q, 1, n_point, 1, 2] + [bs, Len_q, n_head, n_level, n_point, 2] / [1, 1, 1, n_point, 1, 2]
            # -> [bs, Len_q, 1, n_levels, n_points, 2]
            # 参考点 + 偏移量（可学习）/特征层宽高 = 采样点
            sampling_locations = reference_points[:, :, None, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, None, :]
        # two stage  +  iterative bounding box refinement
        elif reference_points.shape[-1] == 4:
            # reference_points = top-k proposal boxes
            # 前两个是xy 后两个是wh
            # 初始化时offset是在 -n_points ~ n_points 范围之间 这里除以self.n_points是相当于把offset归一化到 0~1
            # 然后再乘以宽高的一半 再加上参考点的中心坐标 这就相当于使得最后的采样点坐标总是位于proposal box内
            # 相当于对采样范围进行了约束 减少了搜索空间
            sampling_locations = reference_points[:, :, None, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        # 输入：采样点位置、注意力权重、所有点的value
        # 具体过程：根据采样点位置从所有点的value中拿出对应的value，并且和对应的注意力权重进行weighted sum
        # 调用CUDA实现的MSDeformAttnFunction函数  需要编译
        # [bs, Len_q, 256]  .apply()：接收参数并传递给forward
        output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)
        # 最后进行公式中的线性运算
        # [bs, Len_q, 256]
        output = self.output_proj(output)
        return output


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, Lq_, M_, D_ = value.shape
    _, Lq_, M_, P_, _ = sampling_locations.shape

    # 采样点坐标从[0,1] -> [-1, 1]  F.grid_sample要求采样坐标归一化到[-1, 1]
    sampling_grids = 2 * sampling_locations - 1
    H_, W_ = value_spatial_shapes.tolist()
    # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
    value = value.flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)  # 得到每个特征层的value list
    # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
    sampling_grid = sampling_grids.transpose(1, 2).flatten(0, 1)  # 得到每个特征层的采样点 list
    # N_*M_, D_, Lq_, P_  采样算法  根据每个特征层采样点到每个特征层的value进行采样  非采样点用0填充
    sampling_value = F.grid_sample(value, sampling_grid,
                                   mode='bilinear', padding_mode='zeros', align_corners=False)

    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, P_)
    # 注意力权重 和 采样后的value 进行 weighted sum
    output = (sampling_value * attention_weights).sum(-1).view(N_, M_ * D_, Lq_)
    return output.transpose(1, 2).contiguous()



