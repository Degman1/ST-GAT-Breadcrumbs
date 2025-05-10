import dgl
import torch
import numpy as np
import importlib
from dgl.dataloading import GraphDataLoader
import os
import pandas as pd
import networkx as nx
import sys
import random

import dataloader.breadcrumbs_dataloader
import hyperparameter_search
import models.trainer

if len(sys.argv) < 3:
    print("Usage: python run_hyperparameter_search.py <param_index> <fold_index>")
    sys.exit(1)

param_index = int(sys.argv[1])
fold_index = int(sys.argv[2])

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")
print(f"Version {torch.__version__}")
print(f"Version {dgl.__version__}")

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

config = {
    "EPOCHS": 80,
    "BATCH_SIZE": 50,
    "CHECKPOINT_DIR": "./trained_models/Predicting_Breadcrumbs_Movement",
}

param_grid = {
    "INITIAL_LR": [0.001, 7.5e-4, 5e-4],
    "FINAL_LR": [7.5e-5, 5e-5, 2.5e-5, 1e-5],
    "WEIGHT_DECAY": [1e-5, 2.5e-5, 5e-5],
    "DROPOUT": [0.25],
    "N_HIST": [24],
    "N_PRED": [9],
    "LSTM1_HIDDEN": [32],
    "LSTM2_HIDDEN": [128],
    "ATTENTION_HEADS": [4],
}

results, best_model, best_hyperparams = (
    hyperparameter_search.train_expanding_window_grid_search(
        config, param_grid, device, param_index, fold_index
    )
)
