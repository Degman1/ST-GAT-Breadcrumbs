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

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

##################################################################################################
# Configuration Settings: Change the following parameters for training and inference settings

# Settings for data preprocessing and training
config = {
    "BATCH_SIZE": 50,
    "EPOCHS": 80,
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


class RunType(Enum):
    """Different types of run requests"""

    TRAIN = 1  # Train a model from scratch
    FINETUNE = 2  # Finetune a pre-trained model
    LOAD = 3  # Load a pre-trained model


# Set true to retrain or finetune model, false to load model
RUN_TYPE = RunType.TRAIN
# Set to name of saved model file if loading or finetuning (Applicable if RUN_TYPE = FINETUNE | LOAD_MODEL)
checkpoint_name = "model_05-08-23:03:29.pt"
# Add epochs after which to save the GAT's attention matrix (Applicable if RETRAIN=True; starts @ epoch 1)
save_attention_epochs = []
# Add epochs after which to save a checkpoint file (Applicable if RETRAIN=True; starts @ epoch 1)
checkpoint_epochs = [config["EPOCHS"]]

##################################################################################################

##################################################################################################
# Data Preprocessing

# Preprocess the input graph structure and timeseries data, splitting into train, validation and test sets
dataset, config["D_MEAN"], config["D_STD_DEV"], d_train, d_val, d_test = (
    dataloader.breadcrumbs_dataloader.get_processed_dataset(config)
)
print("Completed Data Preprocessing.")

# Build the test set DGL dataloader
test_dataloader = GraphDataLoader(
    d_test, batch_size=config["BATCH_SIZE"], shuffle=False
)

# NOTE This commented part was used for concept drift analysis
# train_ratio = 0.4
# d_train = dataset[: int(train_ratio * len(dataset))]
# d_val = dataset[
#     int(train_ratio * len(dataset)) : int(train_ratio * len(dataset))
#     + int(0.1 * len(dataset))
# ]

# Dynamically define the number of nodes and edges in the processed graph
config["N_NODES"] = dataset.graphs[0].number_of_nodes()
config["N_EDGES"] = dataset.graphs[0].number_of_edges()

##################################################################################################

##################################################################################################
# Training OR Model Loading

if RUN_TYPE == RunType.LOAD or RUN_TYPE == RunType.FINETUNE:
    if checkpoint_name is None or checkpoint_name == "":
        print(
            f"ERROR: Must input a valid checkpoint file for 'checkpoint_name' variable."
        )
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
        heads=config["ATTENTION_HEADS"],
        lstm1_hidden_size=config["LSTM1_HIDDEN_SIZE"],
        lstm2_hidden_size=config["LSTM2_HIDDEN_SIZE"],
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

    # NOTE This commented part was used for concept drift analysis
    # print("--")
    # _, _, _, y_pred, y_truth, _, _ = models.trainer.model_test(
    #     model,
    #     GraphDataLoader(d_train, batch_size=config["BATCH_SIZE"], shuffle=True),
    #     device,
    #     config,
    # )
    # print("--")
    # for i in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
    #     d_val = dataset[
    #         int(train_ratio * len(dataset))
    #         + int(i * len(dataset)) : int(train_ratio * len(dataset))
    #         + int((i + 0.1) * len(dataset))
    #     ]
    #     _, _, _, y_pred, y_truth, _, _ = models.trainer.model_test(
    #         model,
    #         GraphDataLoader(d_val, batch_size=config["BATCH_SIZE"], shuffle=False),
    #         device,
    #         config,
    #     )
    # print("--")

    if RUN_TYPE == RunType.FINETUNE:
        # Now finetune the model that was loaded.
        for name, layer in model.named_children():
            if name == "gat":
                for param in layer.parameters():
                    param.requires_grad = False

        train_dataloader = GraphDataLoader(
            d_train, batch_size=config["BATCH_SIZE"], shuffle=True
        )
        val_dataloader = GraphDataLoader(
            d_val, batch_size=config["BATCH_SIZE"], shuffle=True
        )

        config["PRETRAINED_EPOCHS"] = checkpoint["epoch"]
        for i in range(len(save_attention_epochs)):
            save_attention_epochs[i] += checkpoint["epoch"]
        for i in range(len(checkpoint_epochs)):
            checkpoint_epochs[i] += checkpoint["epoch"]

        print(f"Number of graphs in training dataset: {len(d_train)}")
        print(f"Number of graphs in validation dataset: {len(d_val)}")

        print(
            f"Training the pre-trained {checkpoint['epoch']}-epoch model over a maximum of {config['EPOCHS']} epochs."
        )

        print(f"Configuration: {config}")
        print(f"Save Attention Epochs: {save_attention_epochs}")
        print(f"Checkpoint Epochs: {checkpoint_epochs}")

        # Configure and train model
        models.trainer.model_train(
            train_dataloader,
            val_dataloader,
            config,
            device,
            save_attention_epochs,
            checkpoint_epochs,
            dataset.graphs[0],
            None,
            True,
            model,
        )
elif RUN_TYPE == RunType.TRAIN:
    train_dataloader = GraphDataLoader(
        d_train, batch_size=config["BATCH_SIZE"], shuffle=True
    )
    val_dataloader = GraphDataLoader(
        d_val, batch_size=config["BATCH_SIZE"], shuffle=True
    )

    print(f"Number of graphs in training dataset: {len(d_train)}")
    print(f"Number of graphs in validation dataset: {len(d_val)}")

    print(f"Training model over a maximum of {config['EPOCHS']} epochs.")

    print(f"Configuration: {config}")
    print(f"Save Attention Epochs: {save_attention_epochs}")
    print(f"Checkpoint Epochs: {checkpoint_epochs}")

    # Configure and train model
    model = models.trainer.model_train(
        train_dataloader,
        val_dataloader,
        config,
        device,
        save_attention_epochs,
        checkpoint_epochs,
        dataset.graphs[0],
        lr_scheduler=True,
    )
else:
    print("ERROR: Invalid run type specified.")
    exit()

print(f"Number of graphs in test dataset: {len(d_test)}")

##################################################################################################

##################################################################################################
# Inference On Test Data

# Run inference on the test data
_, _, _, y_pred, y_truth, _, _ = models.trainer.model_test(
    model, test_dataloader, device, config
)

##################################################################################################

##################################################################################################
# Attention Analysis

# # Build the attention matrix from the stored data.
# epoch = config['EPOCHS'] - 1
# # attention = visualizations.attention_matrix.build_attention_matrices(dataset, config, epoch, "attn_values")[0]

# # Normalize the attention with frobenius norm and save to file
# np.save("attention.npy", attn_matrices_by_epoch[epoch])

# # Plot the attention as a heat map
# visualizations.attention_matrix.plot_heatmap(
#     attn_matrices_by_epoch[epoch],
#     epoch,
#     custom_node_ids=dataset.graphs[0].ndata["id"],
#     display_step=100
# )

# # Compute POI rankings
# significant_pois, sorted_scores, sorted_indices = visualizations.select_significant_pois.get_significant_pois(
#     attn_matrices_by_epoch[epoch],
#     dataset.graphs[0].ndata["id"],
#     plot=True
# )

# print(f"Significant POIs:\n{significant_pois}")
# print(f"Corresponding Scores:\n{sorted_scores}")

# Plot the subgraph of just the top ranked POIs
# G = nx.read_adjlist("dataset/pruned_clustered_G3Hops.adjlist")
# adj_mtx = nx.to_numpy_array(G)
# subgraph_adj_mtx = visualizations.adjacency_matrix.get_subgraph(adj_mtx, sorted_indices[:60])
# visualizations.adjacency_matrix.plot_adjacency_matrix(subgraph_adj_mtx, "./visualizations/subgraph_adj_mtx.png")

# Plot the attention matrix of just the top ranked POIs
# subgraph_attn_mtx = visualizations.adjacency_matrix.get_subgraph(normalized_attention, sorted_indices[:60])
# visualizations.attention_matrix.plot_heatmap(subgraph_attn_mtx, epoch)

##################################################################################################
# Prediction Visualizations on Test Data

# for i in range(30):
#     rank = i
#     prediction_node_index = sorted_indices[i]
#     node_label = significant_pois[i]
#     visualizations.predictions.plot_prediction_full(test_dataloader, y_pred, y_truth, prediction_node_index, node_label, rank, config)
