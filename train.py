import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import LightGCN, bpr_loss
from metrics import get_user_positive_items, recall_precision_atk, ndcg_atk_r
from utils import convert_adj_to_r, sample_mini_batch
from config import Config

def evaluate(model, edge_index, exclude_indices, num_users, num_items, k):
    model.eval()
    with torch.no_grad():
        # Convert to rating matrix format
        r_edge_index = convert_adj_to_r(edge_index, num_users, num_items)
        
        # Get embeddings
        users_emb, _, items_emb, _ = model(edge_index)
        
        # Calculate metrics
        test_user_pos_items = get_user_positive_items(r_edge_index)
        users = r_edge_index[0].unique()
        
        # Generate predictions
        ratings = torch.mm(users_emb, items_emb.t())
        _, top_k_items = torch.topk(ratings, k=k)
        
        r = []
        for user in users:
            true_items = test_user_pos_items[user.item()]
            label = [1 if item in true_items else 0 for item in top_k_items[user]]
            r.append(label)
            
        r = torch.tensor(r, dtype=torch.float)
        ground_truth = [test_user_pos_items[u.item()] for u in users]
        
        recall, precision = recall_precision_atk(ground_truth, r, k)
        ndcg = ndcg_atk_r(ground_truth, r, k)
        
    model.train()
    return recall, precision, ndcg

def train(model, optimizer, scheduler, train_edge, val_edge, num_users, num_items):
    train_losses = []
    val_metrics = []
    
    for iter in tqdm(range(Config.ITERATIONS)):
        # Sample batch
        users, pos_items, neg_items = sample_mini_batch(
            convert_adj_to_r(train_edge, num_users, num_items),
            Config.BATCH_SIZE
        )
        users = users.to(Config.DEVICE)
        pos_items = pos_items.to(Config.DEVICE)
        neg_items = neg_items.to(Config.DEVICE)
        
        # Forward pass
        users_emb, users_emb0, items_emb, items_emb0 = model(train_edge)
        pos_emb = items_emb[pos_items]
        neg_emb = items_emb[neg_items]
        
        # Loss and backward
        loss = bpr_loss(
            users_emb[users], users_emb0[users],
            pos_emb, items_emb0[pos_items],
            neg_emb, items_emb0[neg_items],
            Config.LAMBDA
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Validation
        if iter % Config.ITERS_PER_EVAL == 0:
            recall, prec, ndcg = evaluate(
                model, val_edge, [train_edge], 
                num_users, num_items, Config.K
            )
            val_metrics.append((recall, prec, ndcg))
            train_losses.append(loss.item())
            
            print(f"Iter {iter}: Loss={loss.item():.4f}, "
                  f"Recall@{Config.K}={recall:.4f}, "
                  f"NDCG@{Config.K}={ndcg:.4f}")
        
        # LR scheduling
        if iter % Config.ITERS_PER_LR_DECAY == 0 and iter != 0:
            scheduler.step()
            
    return train_losses, val_metrics

def plot_results(train_losses, val_metrics):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(121)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(122)
    iterations = range(0, Config.ITERATIONS, Config.ITERS_PER_EVAL)
    recalls = [m[0] for m in val_metrics]
    ndcgs = [m[2] for m in val_metrics]
    plt.plot(iterations, recalls, label=f'Recall@{Config.K}')
    plt.plot(iterations, ndcgs, label=f'NDCG@{Config.K}')
    plt.xlabel('Iterations')
    plt.ylabel('Metric Value')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()