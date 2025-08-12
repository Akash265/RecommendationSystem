import torch
import random
from torch_geometric.utils import structured_negative_sampling
from torch_sparse import SparseTensor

def convert_r_to_adj(edge_index, num_users, num_items):
    R = torch.zeros((num_users, num_items))
    for i in range(edge_index.shape[1]):
        row = edge_index[0][i]
        col = edge_index[1][i]
        R[row, col] = 1
        
    adj_mat = torch.zeros((num_users + num_items, num_users + num_items))
    adj_mat[:num_users, num_users:] = R.clone()
    adj_mat[num_users:, :num_users] = R.t().clone()
    return adj_mat.to_sparse_coo().indices()

def convert_adj_to_r(edge_index, num_users, num_items):
    sparse_adj = SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        sparse_sizes=(num_users + num_items, num_users + num_items)
    )
    interact_mat = sparse_adj.to_dense()[:num_users, num_users:]
    return interact_mat.to_sparse_coo().indices()

def sample_mini_batch(edge_index, batch_size):
    user_indices, pos_item_indices, neg_item_indices = structured_negative_sampling(
        edge_index, contains_neg_self_loops=False)
    
    indices = random.choices(range(user_indices.size(0)), k=batch_size)
    batch = torch.stack([
        user_indices[indices],
        pos_item_indices[indices],
        neg_item_indices[indices]
    ])
    return batch[0], batch[1], batch[2]