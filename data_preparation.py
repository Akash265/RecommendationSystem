import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import download_url, extract_zip
from config import Config

def load_data():
    # Download and extract data
 
    extract_zip(download_url(Config.DATA_URL, '.'), '.')
    return pd.read_csv('./ml-latest-small/ratings.csv')

def preprocess_data(rating_df):
    # Encode user and movie IDs
    lbl_user = preprocessing.LabelEncoder()
    lbl_movie = preprocessing.LabelEncoder()
    rating_df.userId = lbl_user.fit_transform(rating_df.userId.values)
    rating_df.movieId = lbl_movie.fit_transform(rating_df.movieId.values)
    return rating_df, len(rating_df['userId'].unique()), len(rating_df['movieId'].unique())

def create_edge_index(rating_df, rating_threshold):
    src = [user_id for user_id in rating_df['userId']]
    dst = [movie_id for movie_id in rating_df['movieId']]
    edge_attr = torch.tensor(rating_df['rating'].values).view(-1, 1) >= rating_threshold
    
    edge_index = [[], []]
    for i in range(edge_attr.shape[0]):
        if edge_attr[i]:
            edge_index[0].append(src[i])
            edge_index[1].append(dst[i])
            
    return torch.tensor(edge_index, dtype=torch.long)

def split_data(edge_index):
    num_interactions = edge_index.shape[1]
    all_indices = np.arange(num_interactions)
    
    train_indices, test_indices = train_test_split(
        all_indices, test_size=Config.TEST_SIZE, random_state=1
    )
    val_indices, test_indices = train_test_split(
        test_indices, test_size=Config.VAL_SIZE, random_state=1
    )
    
    return (
        edge_index[:, train_indices],
        edge_index[:, val_indices],
        edge_index[:, test_indices]
    )