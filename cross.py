import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalFeatureEnhancement(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, num_heads=8, act='ReLU', dropout=0.2):
        super().__init__()
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out
        self.num_heads = num_heads

        self.v_proj = FCNet([v_dim, h_dim], act=act, dropout=dropout)
        self.q_proj = FCNet([q_dim, h_dim], act=act, dropout=dropout)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=h_dim,
            num_heads=self.num_heads,
            kdim=h_dim,
            vdim=h_dim
        )

        self.output_fc = FCNet([h_dim, h_dim], act=act, dropout=dropout)
        self.bn = nn.BatchNorm1d(h_dim)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size=512)

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, v, q, softmax=False):
        v_proj = self.v_proj(v)
        q_proj = self.q_proj(q)

        if v_proj.size(1) != q_proj.size(1):
            max_len = max(v_proj.size(1), q_proj.size(1))
            v_proj = self.adaptive_pool(v_proj.transpose(1, 2)).transpose(1, 2)
            q_proj = self.adaptive_pool(q_proj.transpose(1, 2)).transpose(1, 2)

        v_in = v_proj.permute(1, 0, 2)  # [N, B, D]
        q_in = q_proj.permute(1, 0, 2)

        attn_output, attn_weights = self.cross_attn(
            query=q_in,
            key=v_in,
            value=v_in
        )

        output = attn_output.permute(1, 0, 2)
        output = self.output_fc(output)
        output = self.bn(output.mean(dim=1))




        output = self.alpha * output + self.beta * v_proj.mean(dim=1)

        if softmax:
            attn_weights = F.softmax(attn_weights, dim=-1)

        return output, attn_weights

class FCNet(nn.Module):
    def __init__(self, dims, act='ReLU', dropout=0):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if act: layers.append(getattr(nn, act)())
            if dropout > 0: layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.shortcut = nn.Linear(dims[0], dims[-1]) if dims[0] != dims[-1] else None

    def forward(self, x):
        if self.shortcut:
            return self.net(x) + self.shortcut(x)
        return self.net(x) + x

class IterativeCrossModalFeatureEnhancement(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, num_heads=8, act='ReLU', dropout=0.2, num_iterations=1):
        super().__init__()
        self.num_iterations = num_iterations
        self.cfe = CrossModalFeatureEnhancement(v_dim, q_dim, h_dim, h_out, num_heads, act, dropout)

    def forward(self, v, q, softmax=False):
        outputs = []
        attn_weights_list = []
        for _ in range(self.num_iterations):
            output, attn_weights = self.cfe(v, q, softmax)
            outputs.append(output)
            attn_weights_list.append(attn_weights)
            v = output
            q = output

        return outputs[-1], attn_weights_list[-1]