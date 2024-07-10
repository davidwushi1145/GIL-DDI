import csv
import os
from collections import defaultdict

import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve


def read_dataset(drug_name_id, num):
    kg = defaultdict(list)
    tails = {}
    relations = {}
    drug_list = []
    filename = "../dataset/dataset" + str(num) + ".txt"
    with open(filename, encoding="utf8") as reader:
        for line in reader:
            string = line.rstrip().split('//', 2)
            head = string[0]
            tail = string[1]
            relation = string[2]
            drug_list.append(drug_name_id[head])
            if tail not in tails:
                tails[tail] = len(tails)
            if relation not in relations:
                relations[relation] = len(relations)
            if num == 3:
                kg[drug_name_id[head]].append((drug_name_id[tail], relations[relation]))
                kg[drug_name_id[tail]].append((drug_name_id[head], relations[relation]))
            else:
                kg[drug_name_id[head]].append((tails[tail], relations[relation]))
    return kg, len(tails), len(relations)


def prepare(mechanism, action):
    d_label = {}
    d_event = []
    new_label = []
    for i in range(len(mechanism)):
        d_event.append(mechanism[i] + " " + action[i])
    count = {}
    for i in d_event:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    list1 = sorted(count.items(), key=lambda x: x[1], reverse=True)
    for i in range(len(list1)):
        d_label[list1[i][0]] = i
    for i in range(len(d_event)):
        new_label.append(d_label[d_event[i]])
    return new_label


def prepare2(mechanism, action):
    d_label = {}
    d_event = []
    new_label = []
    for i in range(len(mechanism)):
        d_event.append(mechanism[i] + " " + action[i])
    count = {}
    for i in d_event:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    list1 = sorted(count.items(), key=lambda x: x[1], reverse=True)
    for i in range(len(list1)):
        d_label[list1[i][0]] = i
    for i in range(len(d_event)):
        new_label.append(d_label[d_event[i]])
    return new_label, len(count)


def l2_re(parameter):
    reg = 0
    for param in parameter:
        reg += 0.5 * (param ** 2).sum()
    return reg


def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)

    def _average_binary_score(binary_metric, y_true, y_score, average):
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)

    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)


def save_result(filepath, result_type, result, task_num, headers=None):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    full_filepath = os.path.join(filepath, result_type + f'task{task_num}' + '.csv')

    with open(full_filepath, "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if headers is not None:
            writer.writerow(headers)
        for row in result:
            writer.writerow(row)
    return 0


def find_mahalanobis(raw_matrix):
    cov_matrix = np.cov(raw_matrix, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    num_samples = raw_matrix.shape[0]
    sim_matrix4 = np.zeros((num_samples, num_samples), dtype=float)

    for i in range(num_samples):
        for j in range(num_samples):
            if i == j:
                sim_matrix4[i, j] = 0
            else:
                sim_matrix4[i, j] = mahalanobis(raw_matrix[i], raw_matrix[j], inv_cov_matrix)

    return sim_matrix4


def Tanimoto(matrix):
    matrix = np.array(matrix, dtype=float)
    numerator = np.dot(matrix, matrix.T)
    row_sum = np.sum(matrix ** 2, axis=1)
    denominator = row_sum[:, np.newaxis] + row_sum[np.newaxis, :] - numerator
    denominator = np.where(denominator == 0, 1, denominator)
    tanimoto_similarity = numerator / denominator
    return tanimoto_similarity
