import torch
import numpy as np
from collections import defaultdict

def get_user_positive_items(edge_index):
    user_pos_items = defaultdict(list)
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        user_pos_items[user].append(item)
    return user_pos_items

def recall_precision_atk(ground_truth, r, k):
    num_correct_pred = torch.sum(r, dim=-1)
    user_num_liked = torch.tensor([len(gt) for gt in ground_truth])
    recall = torch.mean(num_correct_pred / user_num_liked)
    precision = torch.mean(num_correct_pred) / k
    return recall.item(), precision.item()

def ndcg_atk_r(ground_truth, r, k):
    test_matrix = torch.zeros((len(r), k))
    for i, items in enumerate(ground_truth):
        length = min(len(items), k)
        test_matrix[i, :length] = 1
    
    max_r = test_matrix
    idcg = torch.sum(max_r * 1. / torch.log2(torch.arange(2, k + 2)), axis=1)
    dcg = r * (1. / torch.log2(torch.arange(2, k + 2)))
    dcg = torch.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0.
    return torch.mean(ndcg).item()