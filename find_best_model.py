import dgl
import torch
import numpy as np
import importlib
from dgl.dataloading import GraphDataLoader
import os
import pandas as pd
import networkx as nx
import random
from enum import Enum

import dataloader.breadcrumbs_dataloader
import dataloader.splits
import models.st_gat
import models.trainer
import models.persist
import visualizations.attention_matrix
import visualizations.select_significant_pois
import visualizations.predictions
import visualizations.adjacency_matrix

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

config = {
    "BATCH_SIZE": 50,
    "EPOCHS": 66,
    "WEIGHT_DECAY": 1e-5,
    "INITIAL_LR": 1e-3,
    "FINAL_LR": 7.5e-5,
    "CHECKPOINT_DIR": "./trained_models/Predicting_Breadcrumbs_Movement",
    "N_PRED": 9,
    "N_HIST": 24,
    "DROPOUT": 0.25,
    "ATTENTION_HEADS": 4,
    "LSTM1_HIDDEN_SIZE": 32,
    "LSTM2_HIDDEN_SIZE": 128,
}

# dataset, config["D_MEAN"], config["D_STD_DEV"], d_train, d_val, d_test = (
#     dataloader.breadcrumbs_dataloader.get_processed_dataset(config)
# )
# print("Completed Data Preprocessing.")

# # Build the test set DGL dataloader
# test_dataloader = GraphDataLoader(
#     d_test, batch_size=config["BATCH_SIZE"], shuffle=False
# )

# # Dynamically define the number of nodes and edges in the processed graph
# config["N_NODES"] = dataset.graphs[0].number_of_nodes()
# config["N_EDGES"] = dataset.graphs[0].number_of_edges()

directory = "trained_models/Predicting_Breadcrumbs_Movement"

min_wmape = 10000
min_model = None

# Loop through all .pt files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".pt"):
        checkpoint_path = os.path.join(directory, filename)
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        # model = models.st_gat.ST_GAT(
        #     in_channels=config["N_HIST"],
        #     out_channels=config["N_PRED"],
        #     n_nodes=config["N_NODES"],
        #     dropout=config["DROPOUT"],
        #     heads=config["ATTENTION_HEADS"],
        #     lstm1_hidden_size=config["LSTM1_HIDDEN_SIZE"],
        #     lstm2_hidden_size=config["LSTM2_HIDDEN_SIZE"],
        # )

        # model.load_state_dict(checkpoint["model_state_dict"])

        print(
            f"The loaded model trained for {checkpoint['epoch']} epochs and resulted in a the following metrics:"
        )
        print(f"\tLoss: {checkpoint['loss']}")
        print(f"\tTrain MAE: {checkpoint['train_mae']}")
        print(f"\tTrain RMSE: {checkpoint['train_rmse']}")
        print(f"\tValidation MAE: {checkpoint['val_mae']}")
        print(f"\tValidation RMSE: {checkpoint['val_rmse']}")

        if "val_wmape" in checkpoint:
            print(f"{filename} : {checkpoint['val_wmape']}")
            if min_wmape > checkpoint["val_wmape"]:
                min_wmape = checkpoint["val_wmape"]
                min_model = filename

print("________________")
print(f"Found best model {min_model} with val_wmape {min_wmape}")
