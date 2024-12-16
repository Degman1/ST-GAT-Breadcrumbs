import torch
import torch.optim as optim
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt

from models.st_gat import ST_GAT
from utils.math import *
from torch.utils.tensorboard import SummaryWriter
from models.persist import save_attention_avg_compressed

# Make a tensorboard writer
writer = SummaryWriter()


def setup_directories(name):
    model_dir = f"./trained_models/{name}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    runs_dir = f"./runs/{name}"
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)

    return model_dir, runs_dir


@torch.no_grad()
def eval(
    model,
    device,
    dataloader,
    config,
    eval_type="",
    current_epoch=None,
    epochs_for_saving_attn=None,
):
    """
    Evaluation function to evaluate model on data
    :param model Model to evaluate
    :param device Device to evaluate on
    :param dataloader Data loader
    :param eval_type Name of evaluation type, e.g. Train/Val/Test
    :param current_epoch If type 'Train', pass the epoch number of which the evaluation is occurring
    :param epochs_for_saving_attn If type 'Train', pass the epochs at which you would like to save the attention values
    """
    model.eval()
    model.to(device)

    mae = 0
    rmse = 0
    mape = 0
    n = 0

    # Maps a batch to the corresponding attn matrices
    # Saving to disk instead of memory
    all_attn_matrices = None
    
    pbar = None

    # Evaluate model on all data
    for i, batch in enumerate(dataloader):
        batch = batch.to(device)
        
        if batch.ndata["feat"].shape[0] == 1:
            pass
        
        with torch.no_grad():
            pred, attn = model(batch, device)
            
        truth = batch.ndata["label"].view(pred.shape)
        
        if i == 0:
            y_pred = torch.zeros(len(dataloader), pred.shape[0], pred.shape[1])
            y_truth = torch.zeros(len(dataloader), pred.shape[0], pred.shape[1])
        
        if config["D_MEAN"] is not None and config["D_STD_DEV"] is not None:
            y_truth = un_z_score(truth, config["D_MEAN"], config["D_STD_DEV"])
            y_pred = un_z_score(pred, config["D_MEAN"], config["D_STD_DEV"])
        
        y_pred[i, : pred.shape[0], :] = pred
        y_truth[i, : pred.shape[0], :] = truth
        
        rmse += RMSE(truth, pred)
        mae += MAE(truth, pred)
        mape += MAPE(truth, pred)
        n += 1
        
        # Save the batch-attn matrix pair for visualization
        if (
            eval_type == "Train"
            and current_epoch is not None
            and epochs_for_saving_attn is not None
            and current_epoch in epochs_for_saving_attn
        ):
            # Save individual graph attention matrices
            if i == 0:
                pbar = tqdm(total=len(dataloader), desc=f"Saving Epoch {current_epoch} Attention")
            # src, dst = batch.edges()
            # num_graphs = batch.batch_size  # Number of graphs in the batch
            # print(f"Batch Stats: {len(src)} edges, {batch.num_nodes()} nodes, {num_graphs} graphs.")
            save_attention_avg_compressed(attn, i, current_epoch)
            pbar.update(1)
    
    # Close the progress bar if it was used
    if (
        eval_type == "Train"
        and current_epoch is not None
        and epochs_for_saving_attn is not None
        and current_epoch in epochs_for_saving_attn
        and pbar is not None
    ):
        pbar.close()

    rmse, mae, mape = rmse / n, mae / n, mape / n

    print(f"{eval_type}, MAE: {mae}, RMSE: {rmse}, MAPE: {mape}")

    # get the average score for each metric in each batch
    return rmse, mae, mape, y_pred, y_truth, all_attn_matrices


def train(model, device, dataloader, optimizer, loss_fn, epoch):
    """
    Evaluation function to evaluate model on data
    :param model Model to evaluate
    :param device Device to evaluate on
    :param dataloader Data loader
    :param optimizer Optimizer to use
    :param loss_fn Loss function
    :param epoch Current epoch
    """
    model.train()
    for _, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        batch = batch.to(device)
        optimizer.zero_grad()
        pred, attn = model(batch, device)
        y_pred = torch.squeeze(pred)
        loss = loss_fn()(y_pred.float(), torch.squeeze(batch.ndata["label"]).float())
        writer.add_scalar("Loss/train", loss, epoch)
        loss.backward()
        optimizer.step()

        # Clear GPU memory cache after each batch
        torch.cuda.empty_cache()

    return loss


def model_train(
    train_dataloader, val_dataloader, config, device, epochs_for_saving_attn
):
    """
    Model training function using DGL graph masks.
    """

    # Make the model. Each datapoint in the graph is 228x12: N x F (N = # nodes, F = time window)
    model = ST_GAT(
        in_channels=config["N_HIST"],
        out_channels=config["N_PRED"],
        n_nodes=config["N_NODE"],
        dropout=config["DROPOUT"],
    )
    optimizer = optim.Adam(
        model.parameters(), lr=config["INITIAL_LR"], weight_decay=config["WEIGHT_DECAY"]
    )
    loss_fn = torch.nn.MSELoss

    model.to(device)

    attn_matrices_by_batch_by_epoch = {}

    # For every epoch, train the model on training dataset. Evaluate model on validation dataset
    for epoch in range(config["EPOCHS"]):
        loss = train(model, device, train_dataloader, optimizer, loss_fn, epoch)
        print(f"Loss: {loss:.3f}")
        if epoch % 5 == 0 or epoch == config["EPOCHS"] - 1:
            train_mae, train_rmse, train_mape, _, _, attn_matrices_by_batch = eval(
                model,
                device,
                train_dataloader,
                config,
                "Train",
                epoch,
                epochs_for_saving_attn,
            )
            val_mae, val_rmse, val_mape, _, _, _ = eval(
                model, device, val_dataloader, config, "Valid"
            )

            # Clear GPU memory cache after each evaluation
            torch.cuda.empty_cache()

            # if attn_matrices_by_batch is not None and len(attn_matrices_by_batch) > 0:
            #     attn_matrices_by_batch_by_epoch[epoch] = attn_matrices_by_batch
            writer.add_scalar(f"MAE/train", train_mae, epoch)
            writer.add_scalar(f"RMSE/train", train_rmse, epoch)
            writer.add_scalar(f"MAPE/train", train_mape, epoch)
            writer.add_scalar(f"MAE/val", val_mae, epoch)
            writer.add_scalar(f"RMSE/val", val_rmse, epoch)
            writer.add_scalar(f"MAPE/val", val_mape, epoch)

    writer.flush()
    # Save the model
    timestr = time.strftime("%m-%d-%H%M%S")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        os.path.join(config["CHECKPOINT_DIR"], f"model_{timestr}.pt"),
    )

    return model, attn_matrices_by_batch_by_epoch


def model_test(model, test_dataloader, device, config):
    """
    Test the ST-GAT model
    :param test_dataloader Data loader of test dataset
    :param device Device to evaluate on
    """
    return eval(model, device, test_dataloader, config, "Test")
