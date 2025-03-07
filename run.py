import dgl
import torch
import numpy as np
import importlib
from dgl.dataloading import GraphDataLoader
import os
import pandas as pd
import networkx as nx

import dataloader.breadcrumbs_dataloader
import dataloader.splits
import models.st_gat
import models.trainer
import models.persist
import visualizations.attention_matrix
import visualizations.select_significant_pois
import visualizations.predictions
import visualizations.adjacency_matrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device}")
print(f"Version {torch.__version__}")
print(f"Version {dgl.__version__}")

np.random.seed(0)
torch.manual_seed(0)

model_dir, runs_dir = models.trainer.setup_directories("Predicting_Breadcrumbs_Movement")

RETRAIN = True
SAVE_MODEL = True
SAVE_ATTENTION = True

if RETRAIN and SAVE_ATTENTION:
    models.trainer.delete_files_in_directory("./attn_values")

config = {
    'BATCH_SIZE': 50,
    'EPOCHS': 80,
    'WEIGHT_DECAY': 5e-5,
    'INITIAL_LR': 3e-4,
    'CHECKPOINT_DIR': runs_dir,
    'N_PRED': 9,
    'N_HIST': 12,
    'DROPOUT': 0.2,
    'USE_GAT_WEIGHTS': True,
}

dataset, config['D_MEAN'], config['D_STD_DEV'], d_train, d_val, d_test = dataloader.breadcrumbs_dataloader.get_processed_dataset(config)
print("Completed Data Preprocessing.")

if config['D_MEAN'] is not None:
    print("NOTE: Normalized time series data during preprocessing.")

test_dataloader = GraphDataLoader(d_test, batch_size=config['BATCH_SIZE'], shuffle=False)

model_path = f'{model_dir}/model.pth'
attn_path = f'{model_dir}/attention.pth'

config['N_NODE'] = dataset.graphs[0].number_of_nodes()

attn_matrices_by_batch_by_epoch = None

if SAVE_ATTENTION:
    epochs_for_saving_attn = [config['EPOCHS'] - 1]
else:
    epochs_for_saving_attn = []

if os.path.exists(model_path) and os.path.exists(attn_path) and not RETRAIN:
    print(f"Model found at {model_path}, loading the saved model instead of training.")
    model = models.st_gat.ST_GAT(in_channels=config['N_HIST'], out_channels=config['N_PRED'], n_nodes=config['N_NODE'], dropout=config['DROPOUT'])
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print(f"Loading the saved attention matrices found at {attn_path}.")
    attn_matrices_by_batch_by_epoch = torch.load(attn_path)
else:
    train_dataloader = GraphDataLoader(d_train, batch_size=config['BATCH_SIZE'], shuffle=True)
    val_dataloader = GraphDataLoader(d_val, batch_size=config['BATCH_SIZE'], shuffle=True)

    print(f"Number of graphs in training dataset: {len(d_train)}")
    print(f"Number of graphs in validation dataset: {len(d_val)}")
    
    print(f"Training model over {config['EPOCHS']} epochs.")

    # Configure and train model
    model, attn_matrices_by_batch_by_epoch = models.trainer.model_train(train_dataloader, val_dataloader, config, device, epochs_for_saving_attn)

    if SAVE_ATTENTION:
        # Save the tensor to a file
        torch.save(attn_matrices_by_batch_by_epoch, attn_path)
    if SAVE_MODEL:
        # Save the trained model
        models.persist.save_model(model, model_path)
        
print(f"Number of graphs in test dataset: {len(d_test)}")

# Run inference on the test data
_, _, _, y_pred, y_truth, _ = models.trainer.model_test(model, test_dataloader, device, config)

# Build the attention matrix from the stored data.
epoch = config['EPOCHS'] - 1
attention = visualizations.attention_matrix.build_attention_matrices(dataset, config, epoch, "attn_values")[0]

# Normalize the attention with frobenius norm and save to file
normalized_attention = attention / np.linalg.norm(attention, 'fro')
np.save("normalized_attention.npy", normalized_attention)

# Plot the attention as a heat map
visualizations.attention_matrix.plot_heatmap(
    normalized_attention, 
    epoch, 
    custom_node_ids=dataset.graphs[0].ndata["id"], 
    display_step=100
)

# Compute POI rankings
significant_pois, sorted_scores, sorted_indices = visualizations.select_significant_pois.get_significant_pois(
    normalized_attention, 
    dataset.graphs[0].ndata["id"], 
    plot=True
)

print(f"Significant POIs:\n{significant_pois}")
print(f"Corresponding Scores:\n{sorted_scores}")

# Plot the subgraph of just the top ranked POIs
# G = nx.read_adjlist("dataset/pruned_clustered_G3Hops.adjlist")
# adj_mtx = nx.to_numpy_array(G)
# subgraph_adj_mtx = visualizations.adjacency_matrix.get_subgraph_adjacency(adj_mtx, sorted_indices[:60])
# visualizations.adjacency_matrix.plot_adjacency_matrix(subgraph_adj_mtx, "./visualizations/subgraph_adj_mtx.png")

# Plot the attention matrix of just the top ranked POIs
# subgraph_attn_mtx = visualizations.adjacency_matrix.get_subgraph_adjacency(normalized_attention, sorted_indices[:60])
# visualizations.attention_matrix.plot_heatmap(subgraph_attn_mtx, epoch)

# for i in range(30):
#     rank = i
#     prediction_node_index = sorted_indices[i]
#     node_label = significant_pois[i]
#     visualizations.predictions.plot_prediction_full(test_dataloader, y_pred, y_truth, prediction_node_index, node_label, rank, config)