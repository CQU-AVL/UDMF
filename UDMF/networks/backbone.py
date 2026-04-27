from torch import nn
from torch.nn import functional as F
import torch
from torchvision.models import ResNet50_Weights
from torchvision.models._utils import IntermediateLayerGetter
import torchvision
from utils.my_dataset import NestedTensor


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            # layer0 layer1不需要训练 因为前面层提取的信息其实很有限 都是差不多的 不需要训练
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        # False 检测任务不需要返回中间层
        return_layers = {'layer4': "0"}
        # 检测任务直接返回layer4即可  执行torchvision.models._utils.IntermediateLayerGetter这个函数可以直接返回对应层的输出结果
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, images, mask):
        """
        tensor_list: pad预处理之后的图像信息
        tensor_list.tensors: [bs, 3, 608, 810]预处理后的图片数据 对于小图片而言多余部分用0填充
        tensor_list.mask: [bs, 608, 810] 用于记录矩阵中哪些地方是填充的（原图部分值为False，填充部分值为True）
        """
        # 取出预处理后的图片数据 [bs, 3, 608, 810] 输入模型中  输出layer4的输出结果 dict '0'=[bs, 2048, 19, 26]
        xs = self.body(images)
        # 保存输出数据
        out = {}
        for name, x in xs.items():
            m = mask  # 取出图片的mask [bs, 608, 810] 知道图片哪些区域是有效的 哪些位置是pad之后的无效的
            assert m is not None
            # 通过插值函数知道卷积后的特征的mask  知道卷积后的特征哪些是有效的  哪些是无效的
            # 因为之前图片输入网络是整个图片都卷积计算的 生成的新特征其中有很多区域都是无效的
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]  # m[None]增加1个维度，
            # out['0'] = NestedTensor: tensors[bs, 2048, 19, 26] + mask[bs, 19, 26]
            out[name] = NestedTensor(x, mask)
        # out['0'] = NestedTensor: tensors[bs, 2048, 19, 26] + mask[bs, 19, 26]
        return out['0']


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 train_backbone: bool
                 ):
        # 直接掉包 调用torchvision.models中的backbone
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, False],
            weights=ResNet50_Weights.DEFAULT, norm_layer=FrozenBatchNorm2d)
        # resnet50  2048
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        # num_channels = [512, 1024, 2048]
        super().__init__(backbone, train_backbone, num_channels)


class Backbone_dinov3(nn.Module):
    def __init__(
            self,
            name=None,
            weights_path=None,
            interaction_indexes=[5, 8, 11],
            finetune=True,
            embed_dim=384,
            num_heads=3,
            patch_size=16,
            conv_inplane=16,
            hidden_dim=None,
    ):
        super(Backbone_dinov3, self).__init__()
        if 'dinov3' in name:
            self.dinov3 = torch.hub.load('./dinov3', name, source='local', weights=weights_path)
            while len(self.dinov3.blocks) != (interaction_indexes[-1] + 1):
                del self.dinov3.blocks[-1]
            del self.dinov3.head
        else:
            raise NotImplementedError('Only dinov3 is supported.')

        self.interaction_indexes = interaction_indexes


        if not finetune:
            self.dinov3.eval()
            self.dinov3.requires_grad_(False)

        self.linear = nn.ModuleList([
            nn.Linear(embed_dim, hidden_dim), nn.Linear(embed_dim, hidden_dim), nn.Linear(embed_dim, hidden_dim)
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim), nn.BatchNorm1d(hidden_dim), nn.BatchNorm1d(hidden_dim)
        ])

        self.relu = nn.ModuleList(
            [nn.ReLU(inplace=True), nn.ReLU(inplace=True), nn.ReLU(inplace=True)]
        )
        self.dropout = nn.Dropout(0.1)


    def forward(self, x):
        # 获取中间层输出（可能是 list of (patch_tokens, cls_token)）
        if len(self.interaction_indexes) > 0:
            all_layers = self.dinov3.get_intermediate_layers(
                x, n=self.interaction_indexes, return_class_token=True
            )
        else:
            # 某些 backbone 当没有指定 interaction_indexes 会直接返回 tensor
            all_layers = self.dinov3(x)

        # 规范化 all_layers 结构为 list
        # 情况 A: all_layers 是单个 tensor 或单个 tuple -> 转成 list
        if not isinstance(all_layers, (list, tuple)):
            all_layers = [all_layers]

        # 情况 B: 若只有一层，则复制多份（保持原来逻辑兼容）
        if len(all_layers) == 1:
            all_layers = [all_layers[0], all_layers[0], all_layers[0]]

        cls_tokens = []  # 用于保存每层的 cls_token，元素为 [B, D]

        for layer_out in all_layers:
            # layer_out 可能是 (patch_tokens, cls_token) 或者是完整 token tensor [B, N+1, D]
            if isinstance(layer_out, (list, tuple)) and len(layer_out) >= 2:
                # 官方 get_intermediate_layers 返回 (patch_tokens, cls_token)
                _, cls = layer_out
                # cls shape 通常是 [B, D]
                cls_tokens.append(cls)
            else:
                # layer_out 可能直接是 tokens tensor: [B, N+1, D] 或 [B, N, D]
                tokens = layer_out
                # 若 tokens 维度为 3 且第 1 个维度 >= 1，尝试取第 0 个 token 作为 cls
                # 注意：如果 tokens 已经是 patch-only ([B, N, D])，那就没有 cls token —— 这里抛出友好错误或用池化替代
                if tokens.dim() == 3 and tokens.size(1) >= 1:
                    # 取第 0 个 token 作为 cls
                    cls = tokens[:, 0, :]  # [B, D]
                    cls_tokens.append(cls)
                else:
                    raise RuntimeError(
                        "无法从 backbone 输出中提取 cls token："
                        f"layer_out shape = {getattr(tokens, 'shape', None)}. "
                        "请确认 backbone.get_intermediate_layers(..., return_class_token=True) 被调用，"
                        "或者 layer_out 包含 (patch_tokens, cls_token) 形式。"
                    )

        c2_ = self.linear[0](cls_tokens[0])
        c3_ = self.linear[1](cls_tokens[1])
        c4_ = self.linear[2](cls_tokens[2])
        # 如果你仅想返回 stacked tensor，可以改为 return stacked_cls
        c2 = self.dropout(self.relu[0](self.norms[0](c2_)))
        c3 = self.dropout(self.relu[0](self.norms[1](c3_)))
        c4 = self.dropout(self.relu[0](self.norms[2](c4_)))
        return c2.unsqueeze(1), c3.unsqueeze(1), c4.unsqueeze(1)

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias
