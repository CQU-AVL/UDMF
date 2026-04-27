# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_

from networks.attention_block import DEFORM_ATTEN
# from util.misc import inverse_sigmoid


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        # 复制num_layers=6份encoder_layer=TransformerEncoderLayer
        self.layers = _get_clones(encoder_layer, num_layers)
        # 6层TransformerEncoderLayer
        self.num_layers = num_layers
        self.norm = norm  # layer norm

    @staticmethod
    def get_reference_points(spatial_shape, valid_ratios, device):
        """
        生成参考点   reference points  为什么参考点是中心点？  为什么要归一化？
        spatial_shapes: 4个特征图的shape [4, 2]
        valid_ratios: 4个特征图中非padding部分的边长占其边长的比例  [bs, 4, 2]  如全是1
        device: cuda:0
        """
        reference_points_list = []
        # 遍历4个特征图的shape  比如 H_=100  W_=150
        # 0.5 -> 99.5 取100个点  0.5 1.5 2.5 ... 99.5
        # 0.5 -> 149.5 取150个点 0.5 1.5 2.5 ... 149.5
        # ref_y: [100, 150]  第一行：150个0.5  第二行：150个1.5 ... 第100行：150个99.5  每一行的y坐标相同
        # ref_x: [100, 150]  第一行：0.5 1.5...149.5   100行全部相同  每一列的x坐标相同
        H_, W_ = spatial_shape
        ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                      torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
                                      indexing='ij')  # linspace:均匀分布的数
        # [100, 150] -> [bs, 15000]  150个0.5 + 150个1.5 + ... + 150个99.5 -> 除以100 归一化
        ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, 1] * H_)  # 参考点只在有效像素出现，但现在num还是所有像素点
        # [100, 150] -> [bs, 15000]  100个: 0.5 1.5 ... 149.5  -> 除以150 归一化
        ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, 0] * W_)
        # [bs, 15000, 2] 每一项都是xy
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
        # list4: [bs, H/8*W/8, 2] + [bs, H/16*W/16, 2] + [bs, H/32*W/32, 2] + [bs, H/64*W/64, 2] ->
        # [bs, H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64, 2]
        reference_points = torch.cat(reference_points_list, 1)  # list-->tensor
        # reference_points: [bs, H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64, 2] -> [bs, H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64, 1, 2]
        # valid_ratios: [1, 4, 2] -> [1, 1, 4, 2]
        # 复制4份 每个特征点都有4个归一化参考点 -> [bs, H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64, 4, 2]

        # 4个flatten后特征图的归一化参考点坐标
        return reference_points

    def forward(self, src, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        """
        src: [bs, h*w ,256]  经过Backbone输出的特征图（降维到256）
        mask: None
        src_key_padding_mask: [h*w, bs]  记录每个特征图的每个位置是否是被pad的（True无效   False有效）
        pos: [bs, h*w ,256]  每个特征图的位置编码
        """
        output = src
        encoder_outputs = []  # 用来保存每层的输出

        # 遍历这6层 TransformerEncoderLayer
        for layer in self.layers:
            output = layer(output, src_mask, src_key_padding_mask, pos)
            encoder_outputs.append(output)  # 保存每层输出

        if self.norm is not None:
            b, c, h, w = output.shape
            output = self.norm(output.flatten(2).transpose(1,2))
            encoder_outputs[-1] = output.view(b, c, h, w)  # 更新最后一层的归一化结果

        # 返回最终输出以及每层输出
        # 最终 output: [h*w, bs, 256]
        # encoder_outputs: list, 每个元素形状相同
        return encoder_outputs


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, return_intermediate=True):
        super().__init__()
        # 复制num_layers=decoder_layer=TransformerDecoderLayer
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers  # 6
        # 是否返回中间层 默认True  因为DETR默认6个Decoder都会返回结果，一起加入损失计算的
        # 每一层Decoder都是逐层解析，逐层加强的，所以前面层的解析效果对后面层的解析是有意义的，所以作者把前面5层的输出也加入损失计算
        self.return_intermediate = return_intermediate
        self.reference_points = nn.Linear(256, 2)
        self._reset_parameters()
        self.bbox_embed = None

        # self.bbox_embed = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(256, 256),
        #         nn.ReLU(),
        #         nn.Linear(256, 4)
        #     )
        #     for _ in range(num_layers)
        # ])
        # nn.init.constant_(self.bbox_embed[0][-1].bias.data[2:], -2.0)
        # hack implementation for iterative bounding box refinement
        # 不使用iterative bounding box refinement时self.transformer.decoder.bbox_embed=None
        # 反之decoder每一层都会预测bbox偏移量 使用这一层bbox偏移量对上一层的预测输出进行矫正

    def _reset_parameters(self):
        for p in self.parameters():  # self:DeformableTransformer
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)

    def forward(self, tgt, src, src_spatial_shapes, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        """
        tgt: [100, bs, 256] 需要预测的目标query embedding 和 query_embed形状相同  且全设置为0
                            在每层decoder层中不断的被refine，相当于一次次的被coarse-to-fine的过程
        memory: [h*w, bs, 256]  Encoder输出  具有全局相关性（增强后）的特征表示
        tgt_mask: None
        tgt_key_padding_mask: None
        memory_key_padding_mask: [bs, h*w]  记录Encoder输出特征图的每个位置是否是被pad的（True无效   False有效）
        pos: [h*w, bs, 256]                 特征图的位置编码
        query_pos: [100, bs, 256]    query embedding的位置编码  随机初始化的
        """
        output = tgt  # 初始化query embedding  全是0
        reference_points = self.reference_points(query_pos).sigmoid()
        intermediate = []  # 用于存放6层decoder的输出结果
        intermediate_reference_points = []

        # 遍历6层decoder
        for lid, layer in enumerate(self.layers):
            # 在本模型中 reference_points 恒为二维 (x, y)，直接按 valid_ratios 缩放
            reference_points_input = reference_points * src_valid_ratios[:, None]

            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_padding_mask)
            # tmp = self.bbox_embed[lid](output)  # [bs, 300, 256] -> [bs, 300, 4(xywh)]
            #
            # if reference_points.shape[-1] == 4:  # two stage
            #     new_reference_points = tmp + inverse_sigmoid(reference_points)
            #     new_reference_points = new_reference_points.sigmoid()
            # else:  # one stage
            #     assert reference_points.shape[-1] == 2
            #     new_reference_points = tmp
            #     # 根据decoder每层解码的特征图->回归头（不共享参数） 得到相对参考点的偏移量xy
            #     # 然后再加上参考点坐标（反归一化），再进行sigmoid归一化 得到矫正的参考点
            #     new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
            #     new_reference_points = new_reference_points.sigmoid()
            # # reference_points: [bs, 300, 2] -> [bs, 300, 4]
            # # .detach() 取消了梯度  因为这个参考点在各层相当于作为先验的角色
            # reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        # 默认执行这里
        # 最后把  6x[100,bs,256] -> [6(6层decoder输出),100,bs,256]
        if self.return_intermediate:
            output = torch.stack(intermediate)
            # all_reference_point = torch.stack(intermediate_reference_points)
            u = output.var(dim=0)
            tanh = nn.Tanh()
            output = output * (1 - tanh(u))
            return output[-1], intermediate_reference_points[-1], u

        return output, reference_points  # 不执行


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, n_points, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        """
        小encoder层  结构：multi-head Attention + add&Norm + feed forward + add&Norm
        d_model: mlp 前馈神经网络的dim
        nhead: 8头注意力机制
        dim_feedforward: 前馈神经网络的维度 2048
        dropout: 0.1
        activation: 激活函数类型
        normalize_before: 是否使用先LN  False
        """
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        # 这个操作是把词向量和位置编码相加操作
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask, src_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None):
        """
        src: [494, bs, 256]  backbone输入下采样32倍后 再 压缩维度到256的特征图
        src_mask: None
        src_key_padding_mask: [bs, 494]  记录哪些位置有pad True 没意义 不需要计算attention
        pos: [494, bs, 256]  位置编码
        """
        # 数据 + 位置编码  [494, bs, 256]
        # 这也是和原版encoder不同的地方，这里每个encoder的q和k都会加上位置编码  再用q和k计算相似度  再和v加权得到更具有全局相关性（增强后）的特征表示
        # 每用一层都加上位置编码  信息不断加强  最终得到的特征全局相全局相关性最强  原版的transformer只在输入加上位置编码  作者发现这样更好
        b, c, h, w = src.shape
        src = src.reshape(h*w, b, c)
        pos = pos.reshape(h*w, b, c)
        q = k = self.with_pos_embed(src, pos)
        # multi-head attention   [494, bs, 256]
        # q 和 k = backbone输出特征图 + 位置编码
        # v = backbone输出特征图
        # 这里对query和key增加位置编码 是因为需要在图像特征中各个位置之间计算相似度/相关性 而value作为原图像的特征 和 相关性矩阵加权，
        # 从而得到各个位置结合了全局相关性（增强后）的特征表示，所以q 和 k这种计算需要+位置编码  而v代表原图像不需要加位置编码
        # nn.MultiheadAttention: 返回两个值  第一个是自注意力层的输出  第二个是自注意力权重  这里取0
        # key_padding_mask: 记录backbone生成的特征图中哪些是原始图像pad的部分 这部分是没有意义的
        #                   计算注意力会被填充为-inf，这样最终生成注意力经过softmax时输出就趋向于0，相当于忽略不计
        # attn_mask: 是在Transformer中用来"防作弊"的,即遮住当前预测位置之后的位置，忽略这些位置，不计算与其相关的注意力权重
        #            而在encoder中通常为None 不适用  decoder中才使用
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]  # 调用pytorch的multi—head
        # add + norm + feed forward + add + norm
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src.view(b, c, h, w)

    def forward_pre(self, src, src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, src=src, attn_mask=src_mask, src_key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:  # False
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)  # 默认执行


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, n_points, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = DEFORM_ATTEN(d_model, nhead, n_points)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, src_padding_mask=None):
        """
        tgt: 需要预测的目标 query embedding  负责预测物体  用于建模图像当中的物体信息  在每层decoder层中不断的被refine
             [bs, h*w, 256]  和 query_embed形状相同  且全设置为0
        memory: [bs, h*w, 256]  Encoder输出  具有全局相关性（增强后）的特征表示
        tgt_mask: None
        memory_mask: None
        tgt_key_padding_mask: None
        memory_key_padding_mask: [bs, h*w]  记录Encoder输出特征图的每个位置是否是被pad的（True无效   False有效）
        pos: [h*w, bs, 256]  encoder输出特征图的位置编码
        query_pos: [100, bs, 256]  query embedding/tgt的位置编码  负责建模物体与物体之间的位置关系  随机初始化的
        tgt_mask、memory_mask、tgt_key_padding_mask是防止作弊的 这里都没有使用
        """
        # 第一个self-attention的目的：找到图像中物体的信息 -> tgt
        # 第一个多头自注意力层：输入qkv都和Encoder无关  都来自于tgt/query embedding
        # 通过第一个self-attention  可以不断建模物体与物体之间的关系  可以知道图像当中哪些位置会存在物体  物体信息->tgt
        # query embedding  +  query_pos
        q = k = self.with_pos_embed(tgt, query_pos)
        # masked multi-head self-attention  计算query embedding的自注意力
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)

        # add + norm
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 第二个self-attention的目的：不断增强encoder的输出特征，将物体的信息不断加入encoder的输出特征中去，更好地表征了图像中的各个物体
        # 第二个多头注意力层，也叫Encoder-Decoder self attention：key和value来自Encoder层输出   Query来自Decoder层输入
        # 第二个self-attention 可以建模图像 与 物体之间的关系
        # 根据上一步得到的tgt作为query 不断的去encoder输出的特征图中去问（q和k计算相似度）  问图像当中的物体在哪里呢？
        # 问完之后再将物体的位置信息融合encoder输出的特征图中（和v做运算）  这样我得到的v的特征就有 encoder增强后特征信息 + 物体的位置信息
        # query = query embedding  +  query_pos
        # key = encoder输出特征 + 特征位置编码
        # value = encoder输出特征
        tgt2 = self.multihead_attn(self.with_pos_embed(tgt, query_pos), reference_points,
                                   src, src_spatial_shapes, src_padding_mask)
        # ada + norm + Feed Forward + add + norm
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # [100, bs, 256]
        # decoder的输出是第一个self-attention输出特征 + 第二个self-attention输出特征
        # 最终的特征：知道图像中物体与物体之间的关系 + encoder增强后的图像特征 + 图像与物体之间的关系
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return TransformerEncoder(TransformerEncoderLayer(args.d_model, args.num_head, args.num_points), args.num_encoder, nn.LayerNorm(args.d_model))


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

