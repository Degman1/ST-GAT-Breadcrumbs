import torch
import torch.optim as optim
from tqdm import tqdm
import time
import os
import itertools
import dgl
from dgl.dataloading import GraphDataLoader
import gc
import json
from datetime import datetime

from models.st_gat import ST_GAT
from utils.math import *
import dataloader.breadcrumbs_dataloader
import models.trainer
import models.early_stopping
import math


def load_previous_results():
    # Open the file, load its current content, and add new data
    try:
        with open("results.json", "r") as file:
            # Load existing data
            return json.load(file)
    except FileNotFoundError:
        # If file doesn't exist, initialize with an empty dictionary
        return {"Results": []}


def train_expanding_window_grid_search(
    config, param_grid, device, param_index, fold_index
):
    """
    Train ST-GAT using an expanding window cross-validation approach with hyperparameter grid search.

    Expanding window schedule:
        Iteration 1: 80% of training data, 100% of validation data
        Iteration 2: 90% of training data, 100% of validation data
        Iteration 3: 100% of training data, 100% of validation data

    :param config: Dictionary containing training configurations.
    :param param_grid: Dictionary containing hyperparameters.
    :param device: Device for training ('cuda' or 'cpu').
    :return: Dictionary with results of each fold, best model state.
    """

    train_ratios = [0.8, 0.9, 1.0]  # Expanding train sizes
    val_ratio = 0.1  # Fixed validation size
    best_model = None
    best_hyperparams = None
    best_val_mae = float("inf")
    results = {}

    # Generate all hyperparameter combinations
    param_combinations = list(itertools.product(*param_grid.values()))

    # Individual script changes
    train_ratios = [train_ratios[fold_index]]
    param_combinations = [param_combinations[param_index]]

    prev_npred = None
    prev_nhist = None

    dataset = None
    d_train = None
    d_val = None
    d_test = None
    num_graphs = None

    for param_set in param_combinations:
        current_params = dict(zip(param_grid.keys(), param_set))
        print(f"\nTesting hyperparameters: {current_params}")

        # Rebuild the dataset if N_PRED or N_HIST changes
        if (
            current_params["N_PRED"] != prev_npred
            or current_params["N_HIST"] != prev_nhist
        ):
            config["N_PRED"] = current_params["N_PRED"]
            config["N_HIST"] = current_params["N_HIST"]
            prev_npred = current_params["N_PRED"]
            prev_nhist = current_params["N_HIST"]
            print("N_PRED or N_HIST changed, regenerating graph dataset.")
            dataset, config["D_MEAN"], config["D_STD_DEV"], d_train, d_val, d_test = (
                dataloader.breadcrumbs_dataloader.get_processed_dataset(config)
            )
            config["N_NODES"] = dataset.graphs[0].number_of_nodes()
            num_graphs = len(dataset)

        # weighted_average_mae = 0

        for fold, train_ratio in enumerate(train_ratios):
            train_size = int(train_ratio * len(d_train))
            val_size = len(d_val)

            if train_size + val_size > num_graphs:
                print(
                    "ERROR: Train or validation ratios combine to be more data than is present in the dataset"
                )
                break  # Prevent out-of-bounds errors

            train_subset = d_train[
                :train_size
            ]  # [dataset[i] for i in range(train_size)]
            val_subset = (
                d_val  # [dataset[i] for i in range(train_size, train_size + val_size)]
            )

            train_dataloader = dgl.dataloading.GraphDataLoader(
                train_subset, batch_size=config["BATCH_SIZE"], shuffle=False
            )
            val_dataloader = dgl.dataloading.GraphDataLoader(
                val_subset, batch_size=config["BATCH_SIZE"], shuffle=False
            )

            print(
                f"Fold {fold}: Train [{len(train_subset)} ({train_ratio})] - Val [{len(val_subset)} ({val_ratio})]"
            )

            # Initialize model with current hyperparameters
            model = ST_GAT(
                in_channels=current_params["N_HIST"],
                out_channels=current_params["N_PRED"],
                n_nodes=config["N_NODES"],
                dropout=current_params["DROPOUT"],
            ).to(device)

            optimizer = optim.Adam(
                model.parameters(),
                lr=current_params["INITIAL_LR"],
                weight_decay=current_params["WEIGHT_DECAY"],
            )
            loss_fn = torch.nn.MSELoss

            # 0.0025, 0.005, 0.0075
            # if (current_params["INITIAL_LR"] == 1e-4):
            #     epochs = 100
            # elif (current_params["INITIAL_LR"] == 5e-4):
            #     epochs = 80
            # elif (current_params["INITIAL_LR"] == 1e-3):
            #     epochs = 70
            # elif current_params["INITIAL_LR"] == 0.0025:
            #     epochs = 60
            # elif current_params["INITIAL_LR"] == 0.005:
            #     epochs = 55
            # elif current_params["INITIAL_LR"] == 0.0075:
            #     epochs = 50
            # elif current_params["INITIAL_LR"] == 0.01:
            #     epochs = 40
            # else:
            #     print("ERROR: Epoch cannot be computed.")
            #     raise

            val_mae = None

            es = models.early_stopping.EarlyStopping(patience=10, min_delta=0.0001)
            epochs = 120
            stopped = 120
            min_val_mae = math.inf
            for epoch in range(epochs):
                train_loss = models.trainer.train(
                    model, device, train_dataloader, optimizer, loss_fn, epoch
                )

                # if epoch % 5 == 0 or epoch == epochs - 1:
                with torch.no_grad():
                    train_rmse, train_mae, train_mape, _, _, attention_matrix = (
                        models.trainer.eval(
                            model, device, train_dataloader, config, "Train"
                        )
                    )
                    val_rmse, val_mae, val_mape, _, _, _ = models.trainer.eval(
                        model, device, val_dataloader, config, "Valid"
                    )
                    val_mae = val_mae.item()

                models.trainer.writer.add_scalar(f"MAE/train", train_mae, epoch)
                models.trainer.writer.add_scalar(f"RMSE/train", train_rmse, epoch)
                models.trainer.writer.add_scalar(f"MAPE/train", train_mape, epoch)
                models.trainer.writer.add_scalar(f"wMAE/val", val_mae, epoch)
                models.trainer.writer.add_scalar(f"RMSE/val", val_rmse, epoch)
                models.trainer.writer.add_scalar(f"MAPE/val", val_mape, epoch)

                if val_mae < min_val_mae:
                    min_val_mae = val_mae
                # Check early stopping
                if es(val_mae):
                    stopped = epoch
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            # Store results for this fold + hyperparameter set
            print(
                f"Achieved validation MAE of {min_val_mae} over {stopped} epochs for fold {fold}"
            )

            models.trainer.writer.flush()

            # Clear GPU memory
            model.to("cpu")
            del model
            del optimizer
            del train_dataloader
            del val_dataloader
            gc.collect()
            torch.cuda.empty_cache()

            # Add to weighted validation across folds
            # weighted_average_mae += train_ratio * val_mae

            # Write the updated data back to the JSON file
            data = load_previous_results()
            with open("results.json", "w") as file:
                new_data = {
                    "params": current_params,
                    "train_percent": train_ratio,
                    "mae": min_val_mae,
                    "completion_time": f"{datetime.now()}",
                }
                data["Results"].append(new_data)
                json.dump(data, file, indent=4)

        # Average over the weighted mae values
        # weighted_average_mae /= len(train_ratios)

        # Save best model & hyperparams
        # if weighted_average_mae < best_val_mae:
        #     best_val_mae = weighted_average_mae
        #     # best_model = model.state_dict()
        #     best_hyperparams = current_params
        #     print("NEW BEST PARAMETER SET FOUND!")

    # print("\nBest weighted average validation MAE:", best_val_mae)
    # print("Best hyperparameters:", best_hyperparams)

    return results, best_model, best_hyperparams
