import time
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads  # 多头注意力的头数
        self.d_model = d_model  # 输入维度（模型的总维度）
        self.head_dim = d_model // n_heads  # 每个注意力头的维度
        assert self.head_dim * n_heads == d_model, "d_model必须能够被n_heads整除"  # 断言，确保d_model可以被n_heads整除

        # 线性变换矩阵，用于将输入向量映射到查询、键和值空间
        self.wq = nn.Linear(d_model, d_model)  # 查询（Query）的线性变换
        self.wk = nn.Linear(d_model, d_model)  # 键（Key）的线性变换
        self.wv = nn.Linear(d_model, d_model)  # 值（Value）的线性变换

        # 最终输出的线性变换，将多头注意力结果合并回原始维度
        self.fc_out = nn.Linear(d_model, d_model)  # 输出的线性变换

    def forward(self, query, key, value, mask):
        # 将嵌入向量分成不同的头
        query = query.view(query.shape[0], -1, self.n_heads, self.head_dim)
        key = key.view(key.shape[0], -1, self.n_heads, self.head_dim)
        value = value.view(value.shape[0], -1, self.n_heads, self.head_dim)

        # 转置以获得维度 batch_size, self.n_heads, seq_len, self.head_dim
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # 计算注意力得分
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.head_dim
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = torch.nn.functional.softmax(scores, dim=-1)

        out = torch.matmul(attention, value)

        # 重塑以恢复原始输入形状
        out = out.transpose(1, 2).contiguous().view(query.shape[0], -1, self.d_model)

        out = self.fc_out(out)
        return out