import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_sparse import SparseTensor
from config import Config
from torch_geometric.nn.conv.gcn_conv import gcn_norm

class LightGCN(MessagePassing):
    def __init__(self, num_users, num_items):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.users_emb = nn.Embedding(num_users, Config.EMBEDDING_DIM)
        self.items_emb = nn.Embedding(num_items, Config.EMBEDDING_DIM)
        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)

    def forward(self, edge_index):
    
        edge_index_norm = gcn_norm(
            edge_index=edge_index,
            add_self_loops=Config.ADD_SELF_LOOPS
        )
        
        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight])
        embs = [emb_0]
        emb_k = emb_0
        
        for _ in range(Config.LAYERS):
            emb_k = self.propagate(
                edge_index=edge_index_norm[0],
                x=emb_k,
                norm=edge_index_norm[1]
            )
            embs.append(emb_k)
            
        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1)
        users_emb_final, items_emb_final = torch.split(
            emb_final, [self.num_users, self.num_items]
        )
        
        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

def bpr_loss(users_emb, users_emb_0, pos_items, pos_items_0, neg_items, neg_items_0, lambda_val):
    reg_loss = lambda_val * (
        users_emb_0.norm(2).pow(2) +
        pos_items_0.norm(2).pow(2) +
        neg_items_0.norm(2).pow(2)
    )
    
    pos_scores = torch.sum(users_emb * pos_items, dim=-1)
    neg_scores = torch.sum(users_emb * neg_items, dim=-1)
    
    bpr_loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
    return bpr_loss + reg_loss