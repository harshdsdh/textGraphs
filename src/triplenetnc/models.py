"""create three network and a classifier head for prediction"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv
import torch.nn.functional as F


class SemanticModule(nn.Module):
    def __init__(self, model_dim, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            model_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, h_self, h_sem, sem_pad_mask=None):
        h_self_3d = h_self.unsqueeze(1)  # [B, 1, D]

        # insert self as an extra key/value token -- always valid, never masked
        h_kv = torch.cat([h_self_3d, h_sem], dim=1)  # [B, 1+k, D]

        if sem_pad_mask is not None:
            self_mask = torch.zeros(
                h_self.size(0), 1, dtype=torch.bool, device=h_self.device
            )
            kv_mask = torch.cat([self_mask, sem_pad_mask], dim=1)  # [B, 1+k]
        else:
            kv_mask = None

        out, weights = self.attn(
            query=h_self_3d, key=h_kv, value=h_kv, key_padding_mask=kv_mask
        )
        z = self.norm(h_self_3d + out)
        z = z.squeeze(1)
        return z, weights


class BiTopoModule(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()

        self.sage1 = SAGEConv(
            in_dim,
            hidden_dim,
        )
        self.sage2 = SAGEConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.sage1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.sage2(h, edge_index)

        return h


class DirTopoModule(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index)

        return h


class CompleteModel(nn.Module):
    def __init__(
        self,
        feat_dim,
        n_classes,
        model_dim,
        n_heads,
        hidden_dim=64,
        dropout=0.1,
    ):
        super().__init__()

        self.self_proj = nn.Linear(feat_dim, model_dim)
        self.sem_proj = nn.Linear(feat_dim, model_dim)

        self.semHead = SemanticModule(model_dim, n_heads, dropout)
        self.biTopo = BiTopoModule(feat_dim, hidden_dim, model_dim, dropout)
        self.dirTopo = DirTopoModule(feat_dim, hidden_dim, model_dim, dropout)

        self.classifier = nn.Sequential(
            nn.Linear(4 * model_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, n_classes),
        )

    def forward(self, x, sem_feats, sem_pad_mask, bi_edge_index, dir_edge_index):
        h_self = self.self_proj(x)

        h_sem = self.sem_proj(sem_feats)
        sem_out, sem_attn = self.semHead(h_self, h_sem, sem_pad_mask)

        bi_out = self.biTopo(x, bi_edge_index)
        dir_out = self.dirTopo(x, dir_edge_index)

        z = torch.cat([h_self, sem_out, bi_out, dir_out], dim=-1)
        logits = self.classifier(z)
        return logits
