import torch
import torch.nn as nn


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        """
        Inception_Block_V1 模块，通过多尺度卷积核捕捉输入特征的不同尺度信息，并将结果聚合。

        参数:
        - in_channels: 输入特征的通道数。
        - out_channels: 输出特征的通道数。
        - num_kernels: 不同尺度卷积核的数量，默认为 6。
        - init_weight: 是否初始化权重，默认为 True。
        """
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels

        # 创建多个不同尺度的卷积核
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)  # 将卷积核存储在 ModuleList 中

        # 初始化权重
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        """
        初始化卷积层的权重。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # Kaiming 初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 初始化偏置为 0

    def forward(self, x):
        """
        前向传播函数，通过多尺度卷积核捕捉输入特征的不同尺度信息，并将结果聚合。

        参数:
        - x: 输入数据，形状为 [B, C, H, W]，其中 B 是批次大小，C 是通道数，H 和 W 是高度和宽度。

        输入:
        - x: [B, C, H, W]，输入数据。

        输出:
        - 聚合后的特征，形状为 [B, out_channels, H, W]。
        """
        res_list = []
        # 对每个卷积核进行卷积操作
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        # 将结果堆叠并取平均值
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class Inception_Block_V2(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels // 2):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[1, 2 * i + 3], padding=[0, i + 1]))
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[2 * i + 3, 1], padding=[i + 1, 0]))
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels // 2 * 2 + 1):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res
