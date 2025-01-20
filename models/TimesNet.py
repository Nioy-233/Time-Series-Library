import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        """
        TimesBlock 模块，用于捕捉时间序列中的周期性特征，并通过参数高效的设计实现时间序列的建模。

        Configs中在TimesNet中已经指定了以下参数:
        - configs.seq_len: 输入序列的长度
        - configs.pred_len: 输出序列的长度
        - configs.d_model: Embedding的嵌入维度

        TimesBlock需要额外指定以下参数:
        - configs.top_k: 选取的周期数量
        - configs.d_ff: Inception模块的输出维度
        - configs.num_kernels: Inception模块中使用不同尺度卷积核的数量
        """
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # 参数高效的设计：使用 Inception 块和 GELU 激活函数
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        """
        前向传播函数，捕捉时间序列中的周期性特征并进行建模。

        参数:
        - x: 输入数据，形状为 [B, T, N]，其中 B 是批次大小，T 是时间步长，N 是特征维度。

        输入:
        - x: [B, T, N]，输入数据。

        输出:
        - 经过周期特征提取和建模后的输出，形状为 [B, T, N]。
        """
        B, T, N = x.size()  # 获取输入数据的形状
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]  # 获取当前周期
            # 填充以保证序列长度是周期的整数倍
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # 将序列重塑为 2D 形式以进行卷积操作
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 通过 2D 卷积捕捉周期特征
            out = self.conv(out)
            # 将序列重塑回原始形式
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])  # 截取有效部分

        # 堆叠所有周期的结果
        res = torch.stack(res, dim=-1)
        # 自适应聚合：根据周期权重加权求和
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # 残差连接
        res = res + x
        return res


class Model(nn.Module):
    def __init__(self, configs):
        """
        这是一个基于TimesNet的模型类，用于处理多种时间序列任务，包括长期预测、短期预测、插值、异常检测和分类。

        Configs中需要指定以下常规参数:
        - configs.task_name: 任务名称
        - configs.seq_len: 输入序列的长度
        - configs.pred_len: 输出序列的长度
        - configs.e_layers: TimesBlock的层数
        - configs.c_out: TimesNet模型的输出序列维度
        - configs.label_len: 只有Autoformer会用到这个参数
            - 即使用不到也要指定，这是一个冗余参数

        Configs中需要指定以下Embedding参数:
        - configs.enc_in: 嵌入层数据的输入维度，也是TimesNet模型的输入序列维度
        - configs.d_model: 嵌入层数据的输出维度，该维度也将作为每层TimesBlock的输入及输出维度
        - configs.embed和configs.freq: 嵌入层的mark嵌入方式和频率
            - 后续会固定读取这两个参数，因此即使没有mark也要指定这两个参数
        - configs.dropout: 嵌入层的Dropout比例，用于防止过拟合

        如果是分类任务，则需要额外指定以下参数:
        - configs.num_class: 分类数量

        用到了TimesBlock模块，该模块需要额外指定以下参数:
        - configs.top_k: 选取的FFT周期数量
        - configs.d_ff: Inception模块的输出维度
        - configs.num_kernels: Inception模块中使用不同尺度卷积核的数量
        """
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        执行时间序列的预测任务。

        参数:
        - x_enc: 编码器的输入数据。
        - x_mark_enc: 编码器的时间标记数据。
        - x_dec: 解码器的输入数据（在代码中未被使用，可能是为了未来的扩展或特定任务保留）。
        - x_mark_dec: 解码器的时间标记数据（在代码中未被使用，可能是为了未来的扩展或特定任务保留）。

        输入:
        - x_enc: [B, T, C]，其中B是批次大小，T是时间步长，C是特征维度。
        - x_mark_enc: [B, T, D]，其中D是时间标记的维度。
        - x_dec: [B, T, C]。
        - x_mark_dec: [B, T, D]。

        输出:
        - dec_out: [B, L, D]，其中L是预测的时间步长。
        """
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """
        执行时间序列的插值任务。

        参数:
        - x_enc: 编码器的输入数据。
        - x_mark_enc: 编码器的时间标记数据。
        - x_dec: 解码器的输入数据（在代码中未被使用，可能是为了未来的扩展或特定任务保留）。
        - x_mark_dec: 解码器的时间标记数据（在代码中未被使用，可能是为了未来的扩展或特定任务保留）。
        - mask: 用于指示缺失值的掩码。

        输入:
        - x_enc: [B, T, C]，其中B是批次大小，T是时间步长，C是特征维度。
        - x_mark_enc: [B, T, D]，其中D是时间标记的维度。
        - x_dec: [B, T, C]。
        - x_mark_dec: [B, T, D]。
        - mask: [B, T]，指示哪些位置是缺失的。

        输出:
        - dec_out: [B, L, D]，其中L是预测的时间步长。
        """
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        """
        执行时间序列的异常检测任务。

        参数:
        - x_enc: 编码器的输入数据。

        输入:
        - x_enc: [B, T, C]，其中B是批次大小，T是时间步长，C是特征维度。

        输出:
        - dec_out: [B, L, D]，其中L是预测的时间步长。
        """
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        """
        执行时间序列的分类任务。

        参数:
        - x_enc: 编码器的输入数据。
        - x_mark_enc: 编码器的时间标记数据。

        输入:
        - x_enc: [B, T, C]，其中B是批次大小，T是时间步长，C是特征维度。
        - x_mark_enc: [B, T, D]，其中D是时间标记的维度。

        输出:
        - output: [B, N]，其中N是类别数量。
        """
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        根据任务类型调用相应的函数进行前向传播。

        参数:
        - x_enc: 编码器的输入数据。
        - x_mark_enc: 编码器的时间标记数据。
        - x_dec: 解码器的输入数据（在某些任务中未被使用，可能是为了未来的扩展或特定任务保留）。
        - x_mark_dec: 解码器的时间标记数据（在某些任务中未被使用，可能是为了未来的扩展或特定任务保留）。
        - mask: 用于指示缺失值的掩码（仅在插值任务中使用）。

        输入:
        - x_enc: [B, T, C]，其中B是批次大小，T是时间步长，C是特征维度。
        - x_mark_enc: [B, T, D]，其中D是时间标记的维度。
        - x_dec: [B, T, C]。
        - x_mark_dec: [B, T, D]。
        - mask: [B, T]（仅在插值任务中使用）。

        输出:
        - dec_out: 根据任务类型返回相应的输出。
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None