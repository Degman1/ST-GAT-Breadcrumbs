import dgl
import torch
import numpy as np
import importlib
from dgl.dataloading import GraphDataLoader
import os
import pandas as pd
import networkx as nx
import sys

import dataloader.breadcrumbs_dataloader
import hyperparameter_search.hyperparameter_search
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

np.random.seed(0)
torch.manual_seed(0)

model_dir, runs_dir = models.trainer.setup_directories(
    "Predicting_Breadcrumbs_Movement"
)

config = {"BATCH_SIZE": 50, "CHECKPOINT_DIR": runs_dir}

param_grid = {
    "INITIAL_LR": [0.0005, 0.00025],
    "WEIGHT_DECAY": [5e-5],
    "DROPOUT": [0.3],
    "N_HIST": [24],
    "N_PRED": [12],
}

results, best_model, best_hyperparams = (
    hyperparameter_search.hyperparameter_search.train_expanding_window_grid_search(
        config, param_grid, device, param_index, fold_index
    )
)
