import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, in_features, num_heads=8, dropout=0.3):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.in_features = in_features
        self.head_dim = in_features // num_heads
        assert self.head_dim * num_heads == in_features, "Embedding size must be divisible by num_heads"

        self.query = nn.Linear(in_features, in_features)
        self.key = nn.Linear(in_features, in_features)
        self.value = nn.Linear(in_features, in_features)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(in_features, in_features)

    def forward(self, x):
        batch_size, seq_length, in_features = x.size()

        Q = self.query(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, seq_length, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, in_features)

        out = self.out(out)
        out = F.relu(out)
        out = F.layer_norm(out, [in_features])
        return out


class AttentionLayer(nn.Module):
    def __init__(self, in_features, num_heads=8):
        super(AttentionLayer, self).__init__()
        self.self_attention = SelfAttention(in_features, num_heads)

    def forward(self, x_self, x_neigh, x_rela):
        batch_size, num_neighbors, in_features = x_neigh.shape
        x_self_expanded = x_self.unsqueeze(1).expand(-1, num_neighbors, -1)

        x_concat = torch.cat([x_self_expanded, x_neigh, x_rela], dim=1)
        x_concat = x_concat.reshape(batch_size, -1, in_features)

        attended_out = self.self_attention(x_concat)
        attended_ent = attended_out[:, :num_neighbors, :]

        x_neigh_weighted = attended_ent.mean(dim=1)

        return x_neigh_weighted


class FusionLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fullConnectionLayer = nn.Sequential(
            nn.Linear(args.embedding_num * 3 * 2, args.embedding_num * 3),
            nn.ReLU(),
            nn.BatchNorm1d(args.embedding_num * 3),
            nn.Dropout(args.dropout),
            nn.Linear(args.embedding_num * 3, args.embedding_num * 2),
            nn.ReLU(),
            nn.BatchNorm1d(args.embedding_num * 2),
            nn.Dropout(args.dropout),
            nn.Linear(args.embedding_num * 2, 65))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, arguments):
        gnn3_embedding, gnn2_embedding, gnn1_embedding, idx = arguments
        idx = idx.cpu().numpy().tolist()
        drugA = []
        drugB = []
        for i in idx:
            drugA.append(i[0])
            drugB.append(i[1])
        Embedding = torch.cat([gnn1_embedding[drugA], gnn2_embedding[drugA], gnn3_embedding[drugA],
                               gnn1_embedding[drugB], gnn2_embedding[drugB], gnn3_embedding[drugB]], 1).float()
        fusion_output = self.fullConnectionLayer(Embedding)
        return fusion_output
