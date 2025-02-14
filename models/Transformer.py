import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np


class Model(nn.Module):
    """  
    Transformer模型，用于时间序列预测和相关任务。  

    参数:  
    - configs: 模型配置参数，包含模型结构、任务类型和超参数等信息。  

    输入:  
    - x_enc: 编码器输入序列，形状为(Batch, Length, Features)。  
    - x_mark_enc: 编码器时间特征，形状为(Batch, Length, Features)。  
    - x_dec: 解码器输入序列，形状为(Batch, Length, Features)。  
    - x_mark_dec: 解码器时间特征，形状为(Batch, Length, Features)。  
    - mask: 遮罩张量，用于掩盖序列中的某些位置，形状为(Batch, Length)。  

    输出:  
    - 预测结果或分类结果，具体形状根据任务类型而定。  
    """  
    def __init__(self, configs):
        """  
        初始化模型。  

        参数:  
        - configs: 模型配置参数，包含以下信息：  
            - task_name: 任务类型，如预测、插值、异常检测或分类。  
            - pred_len: 预测序列的长度。  
            - enc_in: 编码器输入特征维度。  
            - dec_in: 解码器输入特征维度。  
            - d_model: 模型维度。  
            - embed: 嵌入层类型。  
            - freq: 频率特征类型。  
            - dropout: dropout比例。  
            - e_layers: 编码器层的数量。  
            - d_layers: 解码器层的数量。  
            - n_heads: 多头注意力的头数。  
            - d_ff: 前馈网络维度。  
            - activation: 激活函数类型。  
            - c_out: 输出特征维度。  
            - num_class: 分类任务的类别数。  
            - seq_len: 序列长度。  

        初始化过程中根据配置创建编码器、解码器和相关投影层。  
        """  
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                               configs.dropout)
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """  
        执行时间序列预测任务。  

        参数:  
        - x_enc: 编码器输入序列，形状为(Batch, Length, Features)。  
        - x_mark_enc: 编码器时间特征，形状为(Batch, Length, Features)。  
        - x_dec: 解码器输入序列，形状为(Batch, Length, Features)。  
        - x_mark_dec: 解码器时间特征，形状为(Batch, Length, Features)。  

        输入:  
        - 编码器和解码器的输入序列及其时间特征。  

        输出:  
        - 预测结果，形状为(Batch, Length, Features)。  
        """  
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """  
        执行数据插值任务。  

        参数:  
        - x_enc: 编码器输入序列，形状为(Batch, Length, Features)。  
        - x_mark_enc: 编码器时间特征，形状为(Batch, Length, Features)。  
        - x_dec: 解码器输入序列，形状为(Batch, Length, Features)。  
        - x_mark_dec: 解码器时间特征，形状为(Batch, Length, Features)。  
        - mask: 遮罩张量，用于标记缺失的位置，形状为(Batch, Length)。  

        输入:  
        - 编码器和解码器的输入序列及其时间特征，以及缺失位置的遮罩。  

        输出:  
        - 插值结果，形状为(Batch, Length, Features)。  
        """  
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        """  
        执行异常检测任务。  

        参数:  
        - x_enc: 编码器输入序列，形状为(Batch, Length, Features)。  

        输入:  
        - 编码器输入序列及其时间特征。  

        输出:  
        - 异常检测结果，形状为(Batch, Length, Features)。  
        """  
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        """  
        执行时间序列分类任务。  

        参数:  
        - x_enc: 编码器输入序列，形状为(Batch, Length, Features)。  
        - x_mark_enc: 编码器时间特征，形状为(Batch, Length, Features)。  

        输入:  
        - 编码器输入序列及其时间特征。  

        输出:  
        - 分类结果，形状为(Batch, NumClass)。  
        """ 
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        前向传播函数。

        参数:
        - x_enc: 编码器输入序列，形状为(Batch, Length, Features)。
        - x_mark_enc: 编码器时间特征，形状为(Batch, Length, Features)。
        - x_dec: 解码器输入序列，形状为(Batch, Length, Features)。
        - x_mark_dec: 解码器时间特征，形状为(Batch, Length, Features)。
        - mask: 遮罩张量，用于掩盖序列中的某些位置，形状为(Batch, Length)。

        输入:
        - 编码器和解码器的输入序列及其时间特征，以及缺失位置的遮罩。

        输出:
        - 根据任务类型返回预测结果、插值结果、异常检测结果或分类结果。
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None