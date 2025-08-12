import torch
from data_preparation import load_data, preprocess_data, create_edge_index, split_data
from model import LightGCN
from train import train, plot_results,evaluate
from utils import convert_r_to_adj
from config import Config
import torch.optim as optim

def main():
    # Load and prepare data
    rating_df = load_data()
    rating_df, num_users, num_movies = preprocess_data(rating_df)
    edge_index = create_edge_index(rating_df, Config.RATING_THRESHOLD)
    train_edge, val_edge, test_edge = split_data(edge_index)
    
    # Convert to adjacency matrix format
    train_edge_adj = convert_r_to_adj(train_edge, num_users, num_movies)
    val_edge_adj = convert_r_to_adj(val_edge, num_users, num_movies)
    test_edge_adj = convert_r_to_adj(test_edge, num_users, num_movies)
    
    # Setup model and optimizer
    device = torch.device(Config.DEVICE)
    model = LightGCN(num_users, num_movies).to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=Config.GAMMA)
    
    # Move data to device
    train_edge_adj = train_edge_adj.to(device)
    val_edge_adj = val_edge_adj.to(device)
    
    # Train model
    train_losses, val_metrics = train(
        model, optimizer, scheduler,
        train_edge_adj, val_edge_adj,
        num_users, num_movies
    )
    
    # Plot results
    plot_results(train_losses, val_metrics)
    
    # Final evaluation on test set
    recall, prec, ndcg = evaluate(
        model, test_edge_adj, [train_edge_adj, val_edge_adj],
        num_users, num_movies, Config.K
    )
    print(f"\nFinal Test Results: Recall@{Config.K}={recall:.4f}, "
          f"Precision@{Config.K}={prec:.4f}, NDCG@{Config.K}={ndcg:.4f}")

if __name__ == "__main__":
    main()