import numpy as np
import torch
import torch.nn as nn

from model.task1.layers import AttentionLayer


class GNN1(nn.Module):
    def __init__(self, dataset, tail_len, relation_len, args, dict1, drug_name, **kwargs):
        super(GNN1, self).__init__(**kwargs)
        self.kg, self.dict1 = dataset["dataset1"], dict1
        self.drug_name, self.args = drug_name, args
        self.drug_embed = nn.Embedding(num_embeddings=572, embedding_dim=args.embedding_num)
        self.rela_embed = nn.Embedding(num_embeddings=relation_len["dataset1"], embedding_dim=args.embedding_num)
        self.ent_embed = nn.Embedding(num_embeddings=tail_len["dataset1"], embedding_dim=args.embedding_num)
        self.W1 = nn.Parameter(torch.randn(size=(572, args.embedding_num, args.embedding_num)))
        self.b1 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))
        self.W2 = nn.Parameter(torch.randn(size=(572, args.embedding_num, args.embedding_num)))
        self.b2 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))
        self.Linear1 = nn.Sequential(nn.Linear(args.embedding_num * 2, args.embedding_num),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(args.embedding_num))
        self.attention = AttentionLayer(args.embedding_num)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, datas):
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        kg, dict1, args, drug_name = self.kg, self.dict1, self.args, self.drug_name
        adj_tail, adj_relation = self.arrge(kg, dict1, args.neighbor_sample_size)

        drug_name = torch.LongTensor(drug_name).to(device)
        adj_tail = torch.LongTensor(adj_tail).to(device)
        adj_relation = torch.LongTensor(adj_relation).to(device)

        drug_embedding = self.drug_embed(drug_name)
        rela_embedding = self.rela_embed(adj_relation)
        ent_embedding = self.ent_embed(adj_tail)

        attended_ent_embedding = self.attention(drug_embedding, ent_embedding, rela_embedding)
        drug_e = torch.cat([attended_ent_embedding, drug_embedding], dim=1)
        drug_f = self.Linear1(drug_e)
        idx, train_or_test, invariant_adj, epoch = datas[0], datas[1], datas[2], datas[3]
        if epoch > 1:
            for i in invariant_adj[0].keys():
                pos = invariant_adj[0][i][0]
                length = len(pos)
                neighbor_avg = torch.sum(drug_f[pos], dim=0) / length
                if length > 0:
                    drug_f[i] = (neighbor_avg + drug_f[i]) / 2
        return drug_f, idx, invariant_adj, train_or_test, epoch

    def arrge(self, kg, drug_name_id, neighbor_sample_size, n_drug=572):
        adj_tail = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        adj_relation = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        for i in drug_name_id:
            all_neighbors = kg[drug_name_id[i]]
            n_neighbor = len(all_neighbors)
            sample_indices = np.random.choice(
                n_neighbor,
                neighbor_sample_size,
                replace=False if n_neighbor >= neighbor_sample_size else True
            )
            adj_tail[drug_name_id[i]] = np.array([all_neighbors[i][0] for i in sample_indices])
            adj_relation[drug_name_id[i]] = np.array([all_neighbors[i][1] for i in sample_indices])
        return adj_tail, adj_relation


class GNN2(nn.Module):
    def __init__(self, dataset, tail_len, relation_len, args, dict1, drug_name, **kwargs):
        super(GNN2, self).__init__(**kwargs)
        self.kg, self.dict1 = dataset["dataset2"], dict1
        self.drug_name, self.args = drug_name, args
        self.drug_embed = nn.Embedding(num_embeddings=572, embedding_dim=args.embedding_num)
        self.rela_embed = nn.Embedding(num_embeddings=relation_len["dataset2"], embedding_dim=args.embedding_num)
        self.ent_embed = nn.Embedding(num_embeddings=tail_len["dataset2"], embedding_dim=args.embedding_num)
        self.W1 = nn.Parameter(torch.randn(size=(572, args.embedding_num, args.embedding_num)))
        self.b1 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))
        self.W2 = nn.Parameter(torch.randn(size=(572, args.embedding_num, args.embedding_num)))
        self.b2 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))

        self.Linear1 = nn.Sequential(nn.Linear(args.embedding_num * 2, args.embedding_num),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(args.embedding_num))
        self.attention = AttentionLayer(args.embedding_num)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, arguments):
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        kg, dict1, args, drug_name = self.kg, self.dict1, self.args, self.drug_name
        gnn1_embedding, idx, invariant_adj, train_or_test, epoch = arguments
        adj_tail, adj_relation = self.arrge(kg, dict1, args.neighbor_sample_size)

        drug_name = torch.LongTensor(drug_name).to(device)
        adj_tail = torch.LongTensor(adj_tail).to(device)
        adj_relation = torch.LongTensor(adj_relation).to(device)

        drug_embedding = self.drug_embed(drug_name)
        rela_embedding = self.rela_embed(adj_relation)
        ent_embedding = self.ent_embed(adj_tail)

        attended_ent_embedding = self.attention(drug_embedding, ent_embedding, rela_embedding)
        drug_e = torch.cat([attended_ent_embedding, drug_embedding], dim=1)
        drug_f = self.Linear1(drug_e)
        if epoch > 1:
            for i in invariant_adj[1].keys():
                pos = invariant_adj[1][i][0]
                length = len(pos)
                neighbor_avg = torch.sum(drug_f[pos], dim=0) / length
                if length > 0:
                    drug_f[i] = (neighbor_avg + drug_f[i]) / 2
        if torch.isnan(drug_f).any() or torch.isinf(drug_f).any():
            print(f"NaN or Inf detected after GNN2")
        return drug_f, gnn1_embedding, idx, invariant_adj, train_or_test, epoch

    def arrge(self, kg, drug_name_id, neighbor_sample_size, n_drug=572):
        adj_tail = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        adj_relation = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        for i in drug_name_id:
            all_neighbors = kg[drug_name_id[i]]
            n_neighbor = len(all_neighbors)
            sample_indices = np.random.choice(
                n_neighbor,
                neighbor_sample_size,
                replace=False if n_neighbor >= neighbor_sample_size else True
            )
            adj_tail[drug_name_id[i]] = np.array([all_neighbors[i][0] for i in sample_indices])
            adj_relation[drug_name_id[i]] = np.array([all_neighbors[i][1] for i in sample_indices])
        return adj_tail, adj_relation


class GNN3(nn.Module):
    def __init__(self, dataset, tail_len, relation_len, args, dict1, drug_name, **kwargs):
        super(GNN3, self).__init__(**kwargs)
        self.kg, self.dict1 = dataset["dataset3"], dict1
        self.drug_name, self.args = drug_name, args
        self.drug_embed = nn.Embedding(num_embeddings=572, embedding_dim=args.embedding_num)
        self.rela_embed = nn.Embedding(num_embeddings=67, embedding_dim=args.embedding_num)
        self.ent_embed = nn.Embedding(num_embeddings=572, embedding_dim=args.embedding_num)
        self.W1 = nn.Parameter(torch.randn(size=(572, args.embedding_num, args.embedding_num)))
        self.b1 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))
        self.W2 = nn.Parameter(torch.randn(size=(572, args.embedding_num, args.embedding_num)))
        self.b2 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))
        self.Linear1 = nn.Sequential(nn.Linear(args.embedding_num * 2, args.embedding_num),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(args.embedding_num))
        self.attention = AttentionLayer(args.embedding_num)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, arguments):
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        kg, dict1, args, drug_name = self.kg, self.dict1, self.args, self.drug_name
        gnn2_embedding, gnn1_embedding, idx, invariant_adj, train_or_test, epoch = arguments
        adj_tail, adj_relation = self.arrge(kg, dict1, args.neighbor_sample_size)

        drug_name = torch.LongTensor(drug_name).to(device)
        adj_tail = torch.LongTensor(adj_tail).to(device)
        adj_relation = torch.LongTensor(adj_relation).to(device)

        drug_embedding = self.drug_embed(drug_name)
        rela_embedding = self.rela_embed(adj_relation)
        ent_embedding = self.ent_embed(adj_tail)

        attended_ent_embedding = self.attention(drug_embedding, ent_embedding, rela_embedding)
        drug_e = torch.cat([attended_ent_embedding, drug_embedding], dim=1)
        drug_f = self.Linear1(drug_e)
        if epoch > 1:
            for i in invariant_adj[3].keys():
                pos = invariant_adj[3][i][0]
                length = len(pos)
                neighbor_avg = torch.sum(drug_f[pos], dim=0) / length
                if length > 0:
                    drug_f[i] = (neighbor_avg + drug_f[i]) / 2
        return drug_f, gnn2_embedding, gnn1_embedding, idx, invariant_adj, train_or_test, epoch

    def arrge(self, kg, drug_name_id, neighbor_sample_size, n_drug=572, tails_num=570, relations_num=65):
        drug_number = []
        drug_list = []
        for i in drug_name_id:
            drug_number.append(drug_name_id[i])
        for key in kg:
            drug_list.append(key)
        surplus = set(drug_number).difference(set(drug_list))
        for i in list(surplus):
            kg[i].append((tails_num + 1, relations_num + 1))
        adj_tail = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        adj_relation = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        for i in drug_name_id:
            all_neighbors = kg[drug_name_id[i]]
            n_neighbor = len(all_neighbors)
            sample_indices = np.random.choice(
                n_neighbor,
                neighbor_sample_size,
                replace=False if n_neighbor >= neighbor_sample_size else True
            )
            adj_tail[drug_name_id[i]] = np.array([all_neighbors[i][0] for i in sample_indices])
            adj_relation[drug_name_id[i]] = np.array([all_neighbors[i][1] for i in sample_indices])
        return adj_tail, adj_relation


class GNN4(nn.Module):
    def __init__(self, dataset, tail_len, relation_len, args, dict1, drug_name, **kwargs):
        super(GNN4, self).__init__(**kwargs)
        self.kg, self.dict1 = dataset["dataset4"], dict1
        self.drug_name, self.args = drug_name, args
        self.drug_embed = nn.Embedding(num_embeddings=572, embedding_dim=args.embedding_num)
        self.rela_embed = nn.Embedding(num_embeddings=relation_len["dataset4"], embedding_dim=args.embedding_num)
        self.ent_embed = nn.Embedding(num_embeddings=tail_len["dataset4"], embedding_dim=args.embedding_num)
        self.W1 = nn.Parameter(torch.randn(size=(572, args.embedding_num, args.embedding_num)))
        self.b1 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))
        self.W2 = nn.Parameter(torch.randn(size=(572, args.embedding_num, args.embedding_num)))
        self.b2 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))

        self.Linear1 = nn.Sequential(nn.Linear(args.embedding_num * 2, args.embedding_num),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(args.embedding_num))
        self.attention = AttentionLayer(args.embedding_num)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, arguments):
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        kg, dict1, args, drug_name = self.kg, self.dict1, self.args, self.drug_name
        gnn3_embedding, gnn2_embedding, gnn1_embedding, idx, invariant_adj, train_or_test, epoch = arguments

        adj_tail, adj_relation = self.arrge(kg, dict1, args.neighbor_sample_size)

        drug_name = torch.LongTensor(drug_name).to(device)
        adj_tail = torch.LongTensor(adj_tail).to(device)
        adj_relation = torch.LongTensor(adj_relation).to(device)

        drug_embedding = self.drug_embed(drug_name)
        rela_embedding = self.rela_embed(adj_relation)
        ent_embedding = self.ent_embed(adj_tail)

        attended_ent_embedding = self.attention(drug_embedding, ent_embedding, rela_embedding)
        drug_e = torch.cat([attended_ent_embedding, drug_embedding], dim=1)
        drug_f = self.Linear1(drug_e)
        if epoch > 1:
            for i in invariant_adj[2].keys():
                pos = invariant_adj[2][i][0]
                length = len(pos)
                neighbor_avg = torch.sum(drug_f[pos], dim=0) / length
                if length > 0:
                    drug_f[i] = (neighbor_avg + drug_f[i]) / 2
        return drug_f, gnn3_embedding, gnn2_embedding, gnn1_embedding, idx

    def arrge(self, kg, drug_name_id, neighbor_sample_size, n_drug=572):
        adj_tail = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        adj_relation = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        for i in drug_name_id:
            all_neighbors = kg[drug_name_id[i]]
            n_neighbor = len(all_neighbors)
            sample_indices = np.random.choice(
                n_neighbor,
                neighbor_sample_size,
                replace=False if n_neighbor >= neighbor_sample_size else True
            )
            adj_tail[drug_name_id[i]] = np.array([all_neighbors[i][0] for i in sample_indices])
            adj_relation[drug_name_id[i]] = np.array([all_neighbors[i][1] for i in sample_indices])
        return adj_tail, adj_relation
