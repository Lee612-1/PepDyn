from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv, LayerNorm, global_mean_pool


class GCNBackbone(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 128, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(LayerNorm(hidden_dim))

    def forward(self, data):
        x = torch.relu(self.input_proj(data.x))
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, data.edge_index, data.edge_weight)
            x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)
            x = x + residual
        return x


class RMSFGCN(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 128, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.backbone = GCNBackbone(in_channels, hidden_dim, num_layers, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        node_repr = self.backbone(data)
        return self.head(node_repr).squeeze(-1)


class MMGBSAGCN(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 128, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.backbone = GCNBackbone(in_channels, hidden_dim, num_layers, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        node_repr = self.backbone(data)
        graph_repr = global_mean_pool(node_repr, data.batch)
        return self.head(graph_repr).squeeze(-1)
