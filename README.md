# LightGCN Movie Recommendation System

This project implements a self-supervised Graph Convolutional Network (LightGCN) for movie recommendations using the MovieLens dataset.

## Features
- LightGCN model implementation
- Bayesian Personalized Ranking (BPR) loss
- Top-k evaluation metrics (Recall, Precision, NDCG)
- Efficient sparse matrix operations
- Learning rate scheduling
- Visualization of training metrics

## Dataset
MovieLens Latest Small (100,000 ratings from 600 users on 9,000 movies)

## Requirements
Python 3.7+ with packages listed in `requirements.txt`

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run main script: `python main.py`

## Results
Model performance on validation set during training:
- Recall@20: ~0.15
- NDCG@20: ~0.08

Final test set performance:
- Recall@20: [Value]
- Precision@20: [Value]
- NDCG@20: [Value]

## File Structure

lightgcn-movielens/
├── config.py # Configuration parameters
├── data/ # Dataset storage
├── data_preparation.py # Data loading and preprocessing
├── metrics.py # Evaluation metrics
├── model.py # LightGCN model definition
├── README.md # Project documentation
├── requirements.txt # Dependencies
├── train.py # Training and evaluation logic
└── utils.py # Helper functions

text

## References
- He, Xiangnan, et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." SIGIR 2020.
- MovieLens Dataset: https://grouplens.org/datasets/movielens/