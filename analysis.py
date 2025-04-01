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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")
print(f"Version {torch.__version__}")

# Settings for data preprocessing and training
config = {
    "BATCH_SIZE": 50,
    "EPOCHS": 30,
    "WEIGHT_DECAY": 5e-5,
    "INITIAL_LR": 5e-5,
    "CHECKPOINT_DIR": "./trained_models/Predicting_Breadcrumbs_Movement",
    "N_PRED": 9,
    "N_HIST": 24,
    "DROPOUT": 0.3,
    "SLOTS_PER_DAY": 24,
}

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

checkpoint_name = "finetune_testing_pred9/stage2_9epochs_5e-5.pt"

# Preprocess the input graph structure and timeseries data, splitting into train, validation and test sets
dataset, config["D_MEAN"], config["D_STD_DEV"], d_train, d_val, d_test = (
    dataloader.breadcrumbs_dataloader.get_processed_dataset(config)
)
print("Completed Data Preprocessing.")

# Build the test set DGL dataloader
test_dataloader = GraphDataLoader(
    d_test, batch_size=config["BATCH_SIZE"], shuffle=False
)

# Dynamically define the number of nodes and edges in the processed graph
config["N_NODES"] = dataset.graphs[0].number_of_nodes()
config["N_EDGES"] = dataset.graphs[0].number_of_edges()

if checkpoint_name is None or checkpoint_name == "":
    print(f"ERROR: Must input a valid checkpoint file for 'checkpoint_name' variable.")
    exit()

checkpoint_path = os.path.join(config["CHECKPOINT_DIR"], checkpoint_name)

if not os.path.exists(checkpoint_path):
    print(f"ERROR: Checkpoint file {checkpoint_path} does not exist.")
    exit()

print(f"Loading checkpoint found at {checkpoint_path}.")
checkpoint = torch.load(checkpoint_path, weights_only=False)

model = models.st_gat.ST_GAT(
    in_channels=config["N_HIST"],
    out_channels=config["N_PRED"],
    n_nodes=config["N_NODES"],
    dropout=config["DROPOUT"],
)

model.load_state_dict(checkpoint["model_state_dict"])

print(
    f"The loaded model trained for {checkpoint['epoch']} epochs and resulted in a the following metrics:"
)
print(f"\tLoss: {checkpoint['loss']}")
print(f"\tTrain MAE: {checkpoint['train_mae']}")
print(f"\tTrain RMSE: {checkpoint['train_rmse']}")
print(f"\tValidation MAE: {checkpoint['val_mae']}")
print(f"\tValidation RMSE: {checkpoint['val_rmse']}")

# Run inference on the test data
_, _, _, y_pred, y_truth, _ = models.trainer.model_test(
    model, test_dataloader, device, config
)

epoch = checkpoint["epoch"]
attention = checkpoint["attention_by_epoch"][epoch - 1]

np.save("output/attention.npy", attention)

# visualizations.attention_matrix.plot_heatmap(
#     attention,
#     epoch,
#     custom_node_ids=dataset.graphs[0].ndata["id"],
#     display_step=100,
# )

n_top_pois = 60

significant_pois, sorted_scores, sorted_indices = (
    visualizations.select_significant_pois.get_significant_pois(
        attention, dataset.graphs[0].ndata["id"], n_top_pois, plot=True
    )
)

print(f"Significant POIs:\n{significant_pois}")
print(f"Corresponding Scores:\n{sorted_scores}")

G = nx.read_adjlist("dataset/pruned_clustered_3hop_graph.adjlist")
adj_mtx = nx.to_numpy_array(G)
subgraph_adj_mtx = visualizations.adjacency_matrix.get_subgraph(
    adj_mtx, sorted_indices[:n_top_pois]
)
visualizations.adjacency_matrix.plot_adjacency_matrix(
    subgraph_adj_mtx, "./output/subgraph_adj_mtx.png"
)

subgraph_attn_mtx = visualizations.adjacency_matrix.get_subgraph(
    attention, sorted_indices[:n_top_pois]
)
visualizations.attention_matrix.plot_heatmap(subgraph_attn_mtx, epoch)

for i in range(n_top_pois):
    rank = i
    prediction_node_index = sorted_indices[i]
    node_label = significant_pois[i]
    visualizations.predictions.plot_prediction_full(
        test_dataloader,
        y_pred,
        y_truth,
        prediction_node_index,
        node_label,
        rank,
        config,
    )
