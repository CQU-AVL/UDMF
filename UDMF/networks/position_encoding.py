import math

import torch
from torch import nn, Tensor

from utils.my_dataset import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    Absolute pos embedding, Sine.  没用可学习参数  不可学习  定义好了就固定了
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, d_model=256, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = d_model / 2  # 128维度  x/y  = d_model/2
        self.temperature = temperature  # 常数 正余弦位置编码公式里面的10000
        self.normalize = normalize  # 是否对向量进行max规范化   True
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            # 这里之所以规范化到2*pi  因为位置编码函数的周期是[2pi, 20000pi]
            scale = 2 * math.pi  # 规范化参数 2*pi
        self.scale = scale

    def forward(self, feat):
        # [bs, 2048, 19, 26]  预处理后的 经过backbone 32倍下采样之后的数据  对于小图片而言多余部分用0填充
        # [bs, 19, 26]  用于记录矩阵中哪些地方是填充的（原图部分值为False，填充部分值为True）
        x = feat.tensors
        mask = feat.mask
        assert mask is not None
        not_mask = ~mask  # True的位置才是真实有效的位置

        # 考虑到图像本身是2维的 所以这里使用的是2维的正余弦位置编码
        # 这样各行/列都映射到不同的值 当然有效位置是正常值 无效位置会有重复值 但是后续计算注意力权重会忽略这部分的
        # 而且最后一个数字就是有效位置的总和，方便max规范化
        # 计算此时y方向上的坐标  [bs, 19, 26]
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        # 计算此时x方向的坐标    [bs, 19, 26]
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        # 最大值规范化 除以最大值 再乘以2*pi 最终把坐标规范化到0-2pi之间
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale  # 两张图，各自最大值不一样
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)  # 创建张量:0 1 2 .. 127
        # 2i/2i+1: 2 * (dim_t // 2)  self.temperature=10000   self.num_pos_feats = d/2
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)  # //:整除   分母
        # 002244....是为了一个sin，一个cos
        pos_x = x_embed[:, :, :, None] / dim_t  # 正余弦括号里面的公式 每一个元素除dim_t,生成新维度，shape:2,23,31,128
        pos_y = y_embed[:, :, :, None] / dim_t  # 正余弦括号里面的公式
        # x方向位置编码: [bs,19,26,64][bs,19,26,64] -> [bs,19,26,64,2] -> [bs,19,26,128]
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)  # .sin()是函数
        # y方向位置编码: [bs,19,26,64][bs,19,26,64] -> [bs,19,26,64,2] -> [bs,19,26,128]
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(
            3)  # 第一个取一个，再在第二个取一个
        # concat: [bs,19,26,128][bs,19,26,128] -> [bs,19,26,256] -> [bs,256,19,26]
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # cat:直接拼接

        # [bs,256,19,26]  dim=1时  前128个是y方向位置编码  后128个是x方向位置编码
        return pos


class BVPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.pe = pe

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)].to(x.device)
        return self.dropout(x)


class BVPositionalEncoding2(nn.Module):

    def __init__(self, args, dropout: float = 0.1, normalize=True, temperature=10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = 2 * math.pi
        shape = [args.batch_size, args.times_num + 1]
        self.num_pose_feat = args.d_model
        not_mask = torch.ones(shape)
        t_embed = not_mask.cumsum(1, dtype=torch.float32)
        if normalize:
            eps = 1e-6
            t_embed = t_embed / (t_embed[:, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pose_feat, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / self.num_pose_feat)
        pos_t = t_embed[:, :, None] / dim_t
        pos = torch.stack((pos_t[:, :, 0::2].sin(), pos_t[:, :, 0::2].cos()), dim=3).flatten(2)
        self.pos = pos

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len = x.size(0), x.size(1)
        pos = self.pos[:batch_size, :seq_len]
        x = x + pos.to(x.device)
        return self.dropout(x)
