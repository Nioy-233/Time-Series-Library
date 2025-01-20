import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        位置嵌入模块，用于为输入序列添加位置信息，以帮助模型捕捉序列中元素的顺序关系。

        参数:
        - d_model: 嵌入后的特征维度。
        - max_len: 最大序列长度，默认为 5000。
        """
        super(PositionalEmbedding, self).__init__()
        # 在 log 空间中计算位置编码。
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False # 位置编码不需要梯度

        # 生成位置和频率项
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()
        
        # 使用正弦和余弦函数生成位置编码
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用正弦函数
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用余弦函数

        pe = pe.unsqueeze(0)  # 增加批次维度
        self.register_buffer('pe', pe)  # 将位置编码注册为缓冲区，不参与梯度更新

    def forward(self, x):
        """
        前向传播函数，为输入序列添加位置编码。

        参数:
        - x: 输入数据，形状为 [B, T, C]。

        输入:
        - x: [B, T, C]，输入数据。

        输出:
        - 位置编码，形状为 [1, T, d_model]，与输入序列的长度一致。
        """
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        """
        令牌嵌入模块，用于将输入数据通过一维卷积映射到高维空间，以提取局部特征并生成嵌入表示。

        参数:
        - c_in: 输入数据的特征维度。
        - d_model: 嵌入后的特征维度。
        """
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2 # 根据 PyTorch 版本设置填充大小
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        # 初始化卷积层权重
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        """
        前向传播函数，通过一维卷积将输入数据映射到高维空间。

        参数:
        - x: 输入数据，形状为 [B, T, C]，其中 B 是批次大小，T 是时间步长，C 是特征维度。

        输入:
        - x: [B, T, C]，输入数据。

        输出:
        - 嵌入后的数据，形状为 [B, T, d_model]。
        """
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        """
        时间嵌入模块，用于将时间特征（如分钟、小时、星期、日期、月份）嵌入到高维空间中，以捕捉时间序列中的时间信息。

        参数:
        - d_model: 嵌入后的特征维度。
        - embed_type: 嵌入类型，默认为 'fixed'，可选值包括 'fixed' 和 'nn.Embedding'。
        - freq: 时间频率，默认为 'h'（小时），用于决定是否需要分钟级别的嵌入。
        """
        super(TemporalEmbedding, self).__init__()

        # 定义时间特征的维度
        minute_size = 4  # 分钟维度
        hour_size = 24  # 小时维度
        weekday_size = 7  # 星期维度
        day_size = 32  # 日期维度
        month_size = 13  # 月份维度

        # 根据嵌入类型选择嵌入方式
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':  # 如果需要分钟级别的嵌入
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)  # 小时嵌入
        self.weekday_embed = Embed(weekday_size, d_model)  # 星期嵌入
        self.day_embed = Embed(day_size, d_model)  # 日期嵌入
        self.month_embed = Embed(month_size, d_model)  # 月份嵌入

    def forward(self, x):
        """
        前向传播函数，将时间特征嵌入到高维空间中。

        参数:
        - x: 输入时间特征数据，形状为 [B, T, F]，F 是时间特征维度。

        输入:
        - x: [B, T, F]，输入时间特征数据，F 的维度顺序为 [月, 日, 星期, 小时, 分钟]。

        输出:
        - 嵌入后的时间特征，形状为 [B, T, d_model]，是所有时间特征嵌入的总和。
        """
        x = x.long()  # 将输入转换为长整型
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.  # 分钟嵌入（如果存在）
        hour_x = self.hour_embed(x[:, :, 3])  # 小时嵌入
        weekday_x = self.weekday_embed(x[:, :, 2])  # 星期嵌入
        day_x = self.day_embed(x[:, :, 1])  # 日期嵌入
        month_x = self.month_embed(x[:, :, 0])  # 月份嵌入

        # 返回所有时间特征嵌入的总和
        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        """
        数据嵌入模块，用于将输入数据和时间标记嵌入到高维空间中，以便模型能够更好地捕捉时序特征。
        共包含三种嵌入方式

        必要参数:
        - c_in: 输入数据的特征维度, 只用于Token嵌入。
        - d_model: 嵌入后的特征维度, 用于全部三种嵌入。
        - dropout: Dropout 概率, 默认为 0.1, 用于防止过拟合。

        可选参数: (只用于Temporal嵌入, 需要在前向传播中使用x_mark时才需要)
        - embed_type: 嵌入类型，默认为 'fixed'，可选值包括 'fixed' 和 'timeF'。
        - freq: 时间频率，默认为 'h'（小时），用于时间嵌入。
        """
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        前向传播函数，将输入数据和时间标记嵌入到高维空间中。

        参数:
        - x: 输入数据，形状为 [B, T, C]，C 是输入特征维度。
        - x_mark: 时间标记数据，形状为 [B, T, D]，D 是时间标记的维度。
            如果为 None，则不使用时间标记。

        输入:
        - x: [B, T, C]，输入数据。
        - x_mark: [B, T, D] 或 None，时间标记数据。

        输出:
        - 嵌入后的数据，形状为 [B, T, d_model]，经过 Dropout 处理。
        """
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars
