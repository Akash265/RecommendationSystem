import torch
# Hyperparameters and configuration
class Config:
    # Data
    DATA_URL = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
    RATING_THRESHOLD = 3.5
    TEST_SIZE = 0.2
    VAL_SIZE = 0.5
    
    # Model
    EMBEDDING_DIM = 64
    LAYERS = 3
    ADD_SELF_LOOPS = False
    
    # Training
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    ITERATIONS = 10000
    BATCH_SIZE = 1024
    LR = 1e-3
    LAMBDA = 1e-6
    K = 20  # Top-k for metrics
    ITERS_PER_EVAL = 200
    ITERS_PER_LR_DECAY = 200
    GAMMA = 0.95  # LR decay factor