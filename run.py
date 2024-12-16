import dgl
import torch
import numpy as np
import importlib
from dgl.dataloading import GraphDataLoader
import os
import pandas as pd

import dataloader.breadcrumbs_dataloader
import dataloader.splits
import models.st_gat
import models.trainer
import models.persist
import visualizations.attention_matrix

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

config = {
    'BATCH_SIZE': 50,
    'EPOCHS': 1,
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
test_dataloader = GraphDataLoader(d_test, batch_size=config['BATCH_SIZE'], shuffle=False)

# model_path = f'{model_dir}/model.pth'
# attn_path = f'{model_dir}/attention.pth'

# config['N_NODE'] = dataset.graphs[0].number_of_nodes()

# attn_matrices_by_batch_by_epoch = None

# if SAVE_ATTENTION:
#     epochs_for_saving_attn = [0, config['EPOCHS'] // 2, config['EPOCHS'] - 1]
# else:
#     epochs_for_saving_attn = []

# if os.path.exists(model_path) and os.path.exists(attn_path) and not RETRAIN:
#     print(f"Model found at {model_path}, loading the saved model instead of training.")
#     model = models.st_gat.ST_GAT(in_channels=config['N_HIST'], out_channels=config['N_PRED'], n_nodes=config['N_NODE'], dropout=config['DROPOUT'])
#     model.load_state_dict(torch.load(model_path, weights_only=True))
#     print(f"Loading the saved attention matrices found at {attn_path}.")
#     attn_matrices_by_batch_by_epoch = torch.load(attn_path)
# else:
#     train_dataloader = GraphDataLoader(d_train, batch_size=config['BATCH_SIZE'], shuffle=True)
#     val_dataloader = GraphDataLoader(d_val, batch_size=config['BATCH_SIZE'], shuffle=True)
    
#     # Get the number of graphs directly from datasets
#     num_train_graphs = len(d_train)
#     num_val_graphs = len(d_val)
#     num_test_graphs = len(d_test)

#     print(f"Number of graphs in training dataset: {num_train_graphs}")
#     print(f"Number of graphs in validation dataset: {num_val_graphs}")
#     print(f"Number of graphs in test dataset: {num_test_graphs}")

#     # Configure and train model
#     model, attn_matrices_by_batch_by_epoch = models.trainer.model_train(train_dataloader, val_dataloader, config, device, epochs_for_saving_attn)

#     if SAVE_ATTENTION:
#         # Save the tensor to a file
#         torch.save(attn_matrices_by_batch_by_epoch, attn_path)
#     if SAVE_MODEL:
#         # Save the trained model
#         models.persist.save_model(model, model_path)

# # Run inference on the test data
# _, _, _, y_pred, y_truth, _ = models.trainer.model_test(model, test_dataloader, device, config)

attention = visualizations.attention_matrix.build_attention_matrices(dataset, config, 0, "attn_values")
visualizations.attention_matrix.plot_heatmap(attention[0], 0, 0, dataset.graphs[0].ndata["id"], 100)