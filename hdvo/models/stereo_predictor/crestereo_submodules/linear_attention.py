'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-12 15:20:18
LastEditors: Ziming Liu
LastEditTime: 2023-03-12 19:39:15
'''
"""
Linear Transformer proposed in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
Modified from: https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def elu(x, alpha=1.0):
    return torch.maximum(torch.zeros_like(x), x) + torch.minimum(torch.zeros_like(x), alpha * (torch.exp(x) - 1))


def elu_feature_map(x):
    return elu(x) + 1


class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * torch.unsqueeze(torch.unsqueeze(q_mask, 2), 3)  # [:, :, None, None]
        if kv_mask is not None:
            K = K * torch.unsqueeze(torch.unsqueeze(kv_mask, 2), 3)  # [:, :, None, None]
            values = values *torch.unsqueeze(torch.unsqueeze(kv_mask, 2), 3)  # [:, :, None, None]

        v_length = values.shape[1]
        values = values / v_length  # prevent fp16 overflow
        KV = torch.sum(torch.unsqueeze(K, -1) * torch.unsqueeze(values, 3), dim=1)
        Z = 1 / (torch.sum(Q * torch.sum(K, dim=1, keepdims=True), dim=-1) + self.eps)
        queried_values = (
            torch.sum(
                torch.unsqueeze(Q, -1) * torch.unsqueeze(KV, 1) * Z.unsqueeze(3).unsqueeze(4),
                dim=3,
            )
            * v_length
        )

        return queried_values


class FullAttention(nn.Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(drop_prob=attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.sum(torch.unsqueeze(queries, 2) * torch.unsqueeze(keys, 1), dim=-1)
        if kv_mask is not None:
            assert q_mask.dtype == np.bool_
            assert kv_mask.dtype == np.bool_
            QK[
                ~(q_mask.unsqueeze(2).unsqueeze(3) & kv_mask.unsqueeze(1).unsqueeze(3))
            ] = float("-inf")

        # Compute the attention and the weighted average
        softmax_temp = 1.0 / queries.shape[3] ** 0.5  # sqrt(D)
        A = F.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.sum(torch.unsqueeze(A, -1) * torch.unsqueeze(values, 1), dim=2)

        return queried_values