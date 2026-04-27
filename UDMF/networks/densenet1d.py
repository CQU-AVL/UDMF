import torch
from torch import nn


class DenseLayer(nn.Module):
    """
    A single dense layer: BN-ReLU-Conv.
    """
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

    def forward(self, x):
        new_features = self.conv(self.relu(self.bn(x)))
        return torch.cat([x, new_features], dim=1)


class TransitionLayer(nn.Module):
    """
    Transition layer to reduce feature maps via Conv1d and downsample via AvgPool1d.
    """
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.relu(self.bn(x)))
        return self.pool(x)


class DenseBlock(nn.Module):
    """
    A dense block consisting of multiple dense layers.
    """
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DenseNet1D(nn.Module):
    def __init__(
        self,
        in_channels,
        num_init_features=16,
        growth_rate=12,
        block_channels=[32, 64, 128],
        num_layers_per_block=4,
    ):
        super(DenseNet1D, self).__init__()

        self.num_blocks = len(block_channels)

        # 初始卷积
        self.conv1 = nn.Conv1d(
            in_channels=in_channels, out_channels=num_init_features, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(num_init_features)
        self.relu = nn.ReLU(inplace=True)

        # Dense Blocks 和 Transition Layers
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        num_features = num_init_features
        for i, out_channels in enumerate(block_channels):
            self.blocks.append(DenseBlock(num_layers=num_layers_per_block, in_channels=num_features, growth_rate=growth_rate))
            num_features = num_features + num_layers_per_block * growth_rate
            if i != self.num_blocks - 1:  # Last block should not have transition
                self.transitions.append(TransitionLayer(in_channels=num_features, out_channels=out_channels))
                num_features = out_channels

        # 最后的 batch normalization
        self.bn_final = nn.BatchNorm1d(num_features)

        # 上采样层，确保能够将输出恢复到256
        self.upsample = nn.ConvTranspose1d(
            in_channels=num_features, out_channels=in_channels, kernel_size=16, stride=8, padding=4
        )

    def forward(self, x):
        # 初始卷积和池化
        x = self.relu(self.bn1(self.conv1(x)))

        # Dense blocks 和 Transition layers
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)

        x = self.bn_final(x)

        # 上采样
        x = self.upsample(x)

        return x