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
import models.hyperparameter_search
import models.trainer

if len(sys.argv) < 3:
    print("Usage: python run_hyperparameter_search.py <param_index> <fold_index>")
    sys.exit(1)

param_index = int(sys.argv[1])
fold_index = int(sys.argv[2])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device}")
print(f"Version {torch.__version__}")
print(f"Version {dgl.__version__}")

np.random.seed(0)
torch.manual_seed(0)

model_dir, runs_dir = models.trainer.setup_directories("Predicting_Breadcrumbs_Movement")

config = {
  'BATCH_SIZE': 50,
  'CHECKPOINT_DIR': runs_dir,
  'USE_GAT_WEIGHTS': True,
}

param_grid = {
  'INITIAL_LR': [1e-4, 5e-4, 1e-3, 5e-3],
  'WEIGHT_DECAY': [1e-6, 5e-5, 1e-5],
  'DROPOUT': [0.1, 0.2],
  'N_HIST': [12, 18, 24],
  'N_PRED': [9],
}

results, best_model, best_hyperparams = models.hyperparameter_search.train_expanding_window_grid_search(config, param_grid, device, param_index, fold_index)