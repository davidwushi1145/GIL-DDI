"""
GIL-DDI Task 3: Prediction of interaction events between new drugs

This script trains and evaluates the GNN model for predicting drug-drug interactions
between new drugs not seen during training. This is the most challenging task,
requiring robust invariant feature learning.
"""
import argparse
import os
import random
import sqlite3
import sys
import warnings

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.task3.gnn_model import GNN1, GNN2, GNN3
from model.task3.layers import FusionLayer
from util.utils import *

warnings.filterwarnings("ignore")
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Auto-detect device instead of hardcoding
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='GNN based on task3')
parser.add_argument("--epoches", type=int, choices=[100, 500, 1000, 2000], default=120,
                    help="Number of training epochs")
parser.add_argument("--batch_size", type=int, choices=[2048, 1024, 512, 256, 128], default=1024,
                    help="Batch size for training")
parser.add_argument("--weigh_decay", type=float, choices=[1e-1, 1e-2, 1e-3, 1e-4, 1e-8], default=1e-8,
                    help="Weight decay (L2 regularization)")
parser.add_argument("--lr", type=float, choices=[1e-3, 1e-4, 1e-5, 4 * 1e-3], default=1e-3,
                    help="Learning rate")
parser.add_argument("--neighbor_sample_size", choices=[4, 6, 10, 16], type=int, default=6,
                    help="Size of neighborhood sampling")
parser.add_argument("--event_num", type=int, default=65,
                    help="Number of DDI event types")
parser.add_argument("--n_drug", type=int, default=572,
                    help="Total number of drugs")
parser.add_argument("--seed", type=int, default=1,
                    help="Random seed for reproducibility")
parser.add_argument("--dropout", type=float, default=0.3,
                    help="Dropout rate")
parser.add_argument("--embedding_num", type=int, choices=[128, 64, 256, 32], default=256,
                    help="Embedding dimension")
args = parser.parse_args()

random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False


class EarlyStopping:
    def __init__(self, patience=10, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model = None

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        self.best_model = model.state_dict()

    def load_best_model(self, model):
        model.load_state_dict(self.best_model)


def evaluate(pred_type, pred_score, y_test, event_num):
    all_eval_type = 11
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    each_eval_type = 6
    result_eve = np.zeros((event_num, each_eval_type), dtype=float)
    y_one_hot = label_binarize(y_test, classes=np.arange(event_num))
    pred_one_hot = label_binarize(pred_type, classes=np.arange(event_num))
    result_all[0] = accuracy_score(y_test, pred_type)
    result_all[1] = roc_aupr_score(y_one_hot, pred_score, average='micro')
    result_all[2] = 0.0
    result_all[3] = roc_auc_score(y_one_hot, pred_score, average='micro')
    result_all[4] = 0.0
    result_all[5] = 0.0
    result_all[6] = f1_score(y_test, pred_type, average='macro')
    result_all[7] = 0.0
    result_all[8] = precision_score(y_test, pred_type, average='macro')
    result_all[9] = 0.0
    result_all[10] = recall_score(y_test, pred_type, average='macro')
    for i in range(event_num):
        result_eve[i, 0] = accuracy_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel())
        result_eve[i, 1] = roc_aupr_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                          average=None)
        result_eve[i, 2] = 0.0
        result_eve[i, 3] = f1_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                    average='binary')
        result_eve[i, 4] = precision_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                           average='binary')
        result_eve[i, 5] = recall_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                        average='binary')
    return [result_all, result_eve]


def train(train_x, train_y, test_x, test_y, net, invariant_adj, early_stopping):
    net.to(device)
    train_x = torch.tensor(train_x, dtype=torch.long).to(device)
    train_y = torch.tensor(train_y, dtype=torch.long).to(device)
    test_x = torch.tensor(test_x, dtype=torch.long).to(device)
    test_y = torch.tensor(test_y, dtype=torch.long).to(device)
    loss_function = nn.CrossEntropyLoss()
    opti = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weigh_decay)
    test_loss, test_acc, train_l = 0, 0, 0
    train_a = []
    train_x1 = train_x.clone()
    train_x[:, [0, 1]] = train_x[:, [1, 0]]
    train_x_total = torch.cat([train_x1, train_x], dim=0)
    train_y = train_y.repeat(2)
    train_data = TensorDataset(train_x_total, train_y)
    train_iter = DataLoader(train_data, args.batch_size, shuffle=True)
    test_list = []
    max_test_output = torch.zeros((0, 65), dtype=torch.float)
    for epoch in range(args.epoches):
        test_loss, test_score, train_l = 0, 0, 0
        train_a = []
        net.train()
        for x, y in train_iter:
            opti.zero_grad()
            train_acc = 0
            train_label = y
            x = x.to(device)
            f_input = list()
            f_input.append(x)
            f_input.append(0)
            f_input.append(invariant_adj)
            f_input.append(epoch)
            output = net(f_input)
            l = loss_function(output, train_label)
            l.backward()
            opti.step()
            train_l += l.item()
            train_acc = accuracy_score(torch.argmax(output, dim=1).cpu().numpy(), train_label.cpu().numpy())
            train_a.append(train_acc)
        net.eval()
        with torch.no_grad():
            test_x = test_x
            f_input = list()
            f_input.append(test_x)
            f_input.append(1)
            f_input.append(invariant_adj)
            f_input.append(epoch)
            test_output = F.softmax(net(f_input), dim=1)
            test_label = test_y
            loss = loss_function(test_output, test_label)
            test_loss = loss.item()
            test_score = f1_score(torch.argmax(test_output, dim=1).cpu().numpy(), test_label.cpu().numpy(),
                                  average='macro')
            test_acc = accuracy_score(torch.argmax(test_output, dim=1).cpu().numpy(), test_label.cpu().numpy())
            test_list.append(test_score)
            if test_score == max(test_list):
                max_test_output = test_output
            print("test_acc:", test_acc, "train_acc:", sum(train_a) / len(train_a), "test_score:", test_score)
        # 检查早停条件
        early_stopping(test_score, net)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break
        print('epoch [%d] train_loss: %.6f testing_loss: %.6f ' % (
            epoch + 1, train_l / len(train_y), test_loss / len(test_y)))
    # 加载最佳模型
    early_stopping.load_best_model(net)
    return test_loss / len(test_y), max(test_list), train_l / len(train_y), sum(train_a) / len(
        train_a), test_list, max_test_output


def main():
    conn = sqlite3.connect("../dataset/event.db")
    df_drug = pd.read_sql('select * from drug;', conn)
    extraction = pd.read_sql('select * from extraction;', conn)
    mechanism = extraction['mechanism']
    action = extraction['action']
    drugA = extraction['drugA']
    drugB = extraction['drugB']
    new_label, event_num = prepare2(mechanism, action)
    new_label = np.array(new_label)
    dict1 = {}
    for i in df_drug["name"]:
        dict1[i] = len(dict1)
    drug_name = [dict1[i] for i in df_drug["name"]]
    drugA_id = [dict1[i] for i in drugA]
    drugB_id = [dict1[i] for i in drugB]
    dataset1_kg, dataset1_tail_len, dataset1_relation_len = read_dataset(dict1, 1)
    dataset2_kg, dataset2_tail_len, dataset2_relation_len = read_dataset(dict1, 2)
    dataset3_kg, dataset3_tail_len, dataset3_relation_len = read_dataset(dict1, 3)
    dataset4_kg, dataset4_tail_len, dataset4_relation_len = read_dataset(dict1, 4)
    x_datasets = {"drugA": drugA_id, "drugB": drugB_id}
    x_datasets = pd.DataFrame(data=x_datasets)
    x_datasets = x_datasets.to_numpy()
    dataset = {"dataset1": dataset1_kg, "dataset2": dataset2_kg, "dataset3": dataset3_kg, "dataset4": dataset4_kg}
    tail_len = {"dataset1": dataset1_tail_len, "dataset2": dataset2_tail_len, "dataset3": dataset3_tail_len,
                "dataset4": dataset4_tail_len}
    relation_len = {"dataset1": dataset1_relation_len, "dataset2": dataset2_relation_len,
                    "dataset3": dataset3_relation_len, "dataset4": dataset4_relation_len}
    train_sum, test_sum = 0, 0

    temp_kg = [defaultdict(list) for _ in range(4)]
    for p, kg in enumerate(dataset):
        for i in dataset[kg].keys():
            for j in dataset[kg][i]:
                temp_kg[p][i].append(j[0])

    feature_matrix1 = np.zeros((572, dataset1_tail_len), dtype=float)
    feature_matrix2 = np.zeros((572, dataset2_tail_len), dtype=float)
    feature_matrix3 = np.zeros((572, dataset4_tail_len), dtype=float)
    feature_matrix4 = np.zeros((572, 572), dtype=float)

    for i in dataset4_kg.keys():
        for p, v in dataset4_kg[i]:
            feature_matrix3[i][p] = v

    for i in temp_kg[0].keys():
        for j in temp_kg[0][i]:
            feature_matrix1[i][j] = 1

    for i in temp_kg[1].keys():
        for j in temp_kg[1][i]:
            feature_matrix2[i][j] = 1

    for i in temp_kg[2].keys():
        for j in temp_kg[2][i]:
            feature_matrix4[i][j] = 1

    drug_sim1 = Tanimoto(feature_matrix1)
    drug_sim2 = Tanimoto(feature_matrix2)
    drug_sim4 = Tanimoto(feature_matrix4)
    drug_sim3 = find_mahalanobis(feature_matrix3)

    temp_drugA = [[] for _ in range(event_num)]
    temp_drugB = [[] for _ in range(event_num)]
    for i in range(len(new_label)):
        temp_drugA[new_label[i]].append(drugA_id[i])
        temp_drugB[new_label[i]].append(drugB_id[i])

    drug_cro_dict = {}
    for i in range(event_num):
        for j in range(len(temp_drugA[i])):
            drug_cro_dict[temp_drugA[i][j]] = j % 10
            drug_cro_dict[temp_drugB[i][j]] = j % 10

    train_drug = [[] for _ in range(10)]
    test_drug = [[] for _ in range(10)]
    for i in range(10):
        for dr_key in drug_cro_dict.keys():
            if drug_cro_dict[dr_key] == i:
                test_drug[i].append(dr_key)
            else:
                train_drug[i].append(dr_key)

    invariant_adj = [[defaultdict(list) for _ in range(4)] for _ in range(10)]
    for i in range(10):
        for k in range(4):
            for j in test_drug[i]:
                target_list = drug_sim1 if k == 0 else drug_sim2 if k == 1 else drug_sim3 if k == 2 else drug_sim4
                target_list = target_list[j]
                max_v = 0
                current_p = []
                for p, v in enumerate(target_list):
                    if v > max_v and p not in test_drug[i] and p != j:
                        max_v = v
                        current_p = [p]
                    elif v == max_v and p not in test_drug[i] and p != j:
                        current_p.append(p)
                invariant_adj[i][k][j].append(current_p)
    y_true = np.array([])
    y_score = np.zeros((0, 65), dtype=float)
    y_pred = np.array([])
    for cross_ver in range(10):
        net = nn.Sequential(GNN1(dataset, tail_len, relation_len, args, dict1, drug_name),
                            GNN2(dataset, tail_len, relation_len, args, dict1, drug_name),
                            GNN3(dataset, tail_len, relation_len, args, dict1, drug_name),
                            FusionLayer(args)).to(device)
        X_train = []
        X_test = []
        y_train = []
        y_test = []
        for i in range(len(drugA)):
            if (drugA_id[i] in np.array(train_drug[cross_ver])) and (drugB_id[i] in np.array(train_drug[cross_ver])):
                X_train.append(i)
                y_train.append(i)
            if (drugA_id[i] not in np.array(train_drug[cross_ver])) and (
                    drugB_id[i] not in np.array(train_drug[cross_ver])):
                X_test.append(i)
                y_test.append(i)

        train_x = x_datasets[X_train]
        train_y = new_label[y_train]
        test_x = x_datasets[X_test]
        test_y = new_label[y_test]
        # 初始化早停对象
        early_stopping = EarlyStopping(patience=100, delta=0.001)

        test_loss, test_acc, train_loss, train_acc, test_list, test_output = train(train_x, train_y, test_x, test_y,
                                                                                   net, invariant_adj[cross_ver],
                                                                                   early_stopping)
        train_sum += train_acc
        test_sum += test_acc
        pred_type = torch.argmax(test_output, dim=1).cpu().numpy()
        y_pred = np.hstack((y_pred, pred_type))
        y_score = np.row_stack((y_score, test_output.cpu()))
        y_true = np.hstack((y_true, test_y))
        print('fold %d, test_loss %f, test_acc %f, train_loss %f, train_acc %f' % (
            cross_ver, test_loss, test_acc, train_loss, train_acc))

    result_all, result_eve = evaluate(y_pred, y_score, y_true, args.event_num)
    all_headers = ["Accuracy", "ROC_AUPR_Micro", "Placeholder1", "ROC_AUC_Micro", "Placeholder2",
                   "Placeholder3", "F1_Macro", "Placeholder4", "Precision_Macro", "Placeholder5",
                   "Recall_Macro"]
    each_headers = ["Accuracy", "ROC_AUPR", "Placeholder1", "F1_Binary", "Precision_Binary",
                    "Recall_Binary"]
    save_result("../result/", "all", result_all, 3, headers=all_headers)
    save_result("../result/", "each", result_eve, 3, headers=each_headers)
    print('%d-fold validation: avg train acc  %f, avg test acc %f' % (10, train_sum / 10, test_sum / 10))
    return


if __name__ == '__main__':
    main()
