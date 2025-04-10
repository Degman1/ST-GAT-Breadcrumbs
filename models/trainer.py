import torch
import torch.optim as optim
from tqdm import tqdm
import time
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, CosineAnnealingLR

from models.st_gat import ST_GAT
from utils.math import *
from torch.utils.tensorboard import SummaryWriter
from models.persist import save_attention_avg_compressed
import models.early_stopping
from visualizations.attention_matrix import *


@torch.no_grad()
def eval(
    model,
    device,
    dataloader,
    config,
    eval_type="",
    current_epoch=None,
    save_attention_epochs=None,
    graph=None,
):
    """
    Evaluates the model on a given dataset without updating gradients.

    Args:
        model (torch.nn.Module): The model to evaluate.
        device (torch.device): The device (CPU/GPU) to run evaluation on.
        dataloader (DataLoader): DataLoader providing batches of evaluation data.
        config (dict): Configuration dictionary with model and dataset details.
        eval_type (str, optional): Type of evaluation (e.g., "Train", "Valid", "Test"). Default is "".
        current_epoch (int, optional): Current epoch number, used for saving attention matrices during training. Default is None.
        save_attention_epochs (list, optional): List of epochs at which attention matrices should be saved (applies during training only). Default is None.
        graph (DGL graph, optional): DGL graph object, needed for attention matrix conversion. Default is None.

    Returns:
        tuple:
            - rmse (float): Average Root Mean Square Error across all batches.
            - mae (float): Average Mean Absolute Error across all batches.
            - mape (float): Average Mean Absolute Percentage Error across all batches.
            - y_pred (torch.Tensor): Collected predictions across all batches.
            - y_truth (torch.Tensor): Collected ground truths across all batches.
            - attn_matrix (np.ndarray or None): Saved attention matrix (if applicable), otherwise None.
    """

    model.eval()
    model.to(device)

    mae = 0
    rmse = 0
    mape = 0
    n = 0

    attn_matrix = None

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
            truth = un_z_score(truth, config["D_MEAN"], config["D_STD_DEV"])
            pred = un_z_score(pred, config["D_MEAN"], config["D_STD_DEV"])

        y_pred[i, : pred.shape[0], :] = pred
        y_truth[i, : pred.shape[0], :] = truth

        rmse += RMSE(truth, pred)
        mae += MAE(truth, pred)
        mape += MAPE(truth, pred)
        n += 1

        # Save the GAT attention matrix if the current epoch attention is requested
        if (
            eval_type == "Train"
            and save_attention_epochs is not None
            and (current_epoch + 1) in save_attention_epochs
        ):
            # Save individual graph attention matrices
            if i == 0:
                attn_npy = attn.cpu().numpy()
                attn_avg = np.mean(attn_npy, axis=1)
                attn_matrix = convert_attention_batch(graph, config, attn_avg)

    rmse, mae, mape = rmse / n, mae / n, mape / n

    print(f"{eval_type}, MAE: {mae}, RMSE: {rmse}")

    # get the average score for each metric in each batch
    return rmse, mae, mape, y_pred, y_truth, attn_matrix


def train(model, device, dataloader, optimizer, loss_fn, epoch):
    """
    Trains the model for one epoch on the provided data.

    Args:
        model (torch.nn.Module): The model to be trained.
        device (torch.device): The device (CPU/GPU) to run training on.
        dataloader (DataLoader): DataLoader providing batches of training data.
        optimizer (torch.optim.Optimizer): The optimizer used to update model weights.
        loss_fn (function): The loss function to calculate training loss.
        epoch (int): The current epoch number, used for logging progress.

    Returns:
        loss (torch.Tensor): The final loss value from the last batch of the epoch.
    """

    model.train()
    for _, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        batch = batch.to(device)
        optimizer.zero_grad()
        pred, attn = model(batch, device)
        y_pred = torch.squeeze(pred)
        loss = loss_fn(y_pred.float(), torch.squeeze(batch.ndata["label"]).float())
        writer.add_scalar("Loss/train", loss, epoch)
        loss.backward()
        optimizer.step()

    return loss


def model_train(
    train_dataloader,
    val_dataloader,
    config,
    device,
    save_attention_epochs,
    checkpoint_epochs,
    graph,
    optimizer_state_dict=None,
    lr_scheduler=True,
    trained_model=None,
):
    """
    Trains a spatiotemporal graph attention network (ST-GAT) model using DGL graph masks, with support for
    learning rate scheduling, attention matrix tracking, and tensorboard logging.

    Args:
        train_dataloader (DataLoader): DataLoader for the training dataset.
        val_dataloader (DataLoader): DataLoader for the validation dataset.
        config (dict): Configuration dictionary containing model hyperparameters, paths, and training settings.
        device (torch.device): The device (CPU/GPU) on which the model will run.
        save_attention_epochs (int): The epochs for saving attention matrices.
        checkpoint_epochs (int): The epochs for saving model checkpoints.
        graph (DGLGraph): The DGL graph structure to apply attention mechanisms on.
        optimizer_state_dict (Dict, Optional): The checkpointed optimizer state
        lr_scheduler (Bool, Optional): True if use a cosine annealing learning rate scheduler
        trained_model (ST_GAT): The pre-trained model if finetuning

    Returns:
        model (torch.nn.Module): The trained ST-GAT model.

    Notes:
        - Warm-up and early stopping functionality is implemented but not currently utilized.
        - MAPE metric is omitted due to instability with small magnitudes in the dataset.
        - For best performance, ensure CUDA is properly configured for GPU-based training.
    """

    # Make a tensorboard writer
    global writer
    writer = SummaryWriter()

    # Make the model. Each datapoint in the graph is 228x12: N x F (N = # nodes, F = time window)
    model = trained_model
    if model is None:
        model = ST_GAT(
            in_channels=config["N_HIST"],
            out_channels=config["N_PRED"],
            n_nodes=config["N_NODES"],
            dropout=config["DROPOUT"],
        )

    # Check which parameters are trainable
    print("The following model layers are trainable")
    found_trainable = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            found_trainable = True
            print(f"\t- {name}")
    if not found_trainable:
        print("\tNone")

    # Check which parameters are frozen
    print("The following model layers are frozen")
    found_frozen = False
    for name, param in model.named_parameters():
        if not param.requires_grad:
            found_frozen = True
            print(f"\t- {name}")
    if not found_frozen:
        print("\tNone")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["INITIAL_LR"],
        weight_decay=config["WEIGHT_DECAY"],
    )
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    loss_fn = torch.nn.MSELoss()

    model.to(device)

    attn_matrices_by_epoch = {}

    # Define the warm-up schedule
    # def warmup_scheduler(epoch):
    #     warm_up_epochs = 5
    #     min_lr_scale = 0.1  # Start at 10% of the base LR (not 0)

    #     if epoch < warm_up_epochs:  # First 5 epochs are warm-up
    #         return min_lr_scale + (1 - min_lr_scale) * (epoch / warm_up_epochs)
    #     return 1  # After warm-up, stay at the base LR

    # Combine the warm-up and plateau schedulers
    # warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_scheduler)
    # plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=6, factor=0.4, min_lr=1e-5)
    # es = models.early_stopping.EarlyStopping(patience=5, min_delta=0.0001)
    # last_lr = config["INITIAL_LR"]
    # print(f"*** Warm up learning rate starting at {warmup_scheduler.get_last_lr()[0]}")

    scheduler = None
    if lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=80,  # Total epochs (full cosine cycle)
            eta_min=1e-6,  # Minimum learning rate at the end
        )

    pretrained_epochs = (
        config["PRETRAINED_EPOCHS"] if "PRETRAINED_EPOCHS" in config else 0
    )

    # For every epoch, train the model on training dataset. Evaluate model on validation dataset
    for epoch in range(pretrained_epochs, pretrained_epochs + config["EPOCHS"]):
        # Single training pass
        loss = train(model, device, train_dataloader, optimizer, loss_fn, epoch)
        print(f"Loss: {loss:.3f}")

        # Compute evaluation metrics on the training data
        train_rmse, train_mae, train_mape, _, _, attention_matrix = eval(
            model,
            device,
            train_dataloader,
            config,
            "Train",
            epoch,
            save_attention_epochs,
            graph,
        )

        # Compute evaluation metrics on the validation data
        val_rmse, val_mae, val_mape, _, _, _ = eval(
            model, device, val_dataloader, config, "Valid", graph
        )

        # If we successfully composed the 2d attention matrix for this epoch, save it
        if attention_matrix is not None:
            attn_matrices_by_epoch[epoch] = attention_matrix

        # Clear GPU memory cache after each evaluation
        torch.cuda.empty_cache()

        # Save evaluation results to the tensorboard writer
        writer.add_scalar(f"MAE/train", train_mae, epoch)
        writer.add_scalar(f"RMSE/train", train_rmse, epoch)
        # NOTE Leave out the MAPE since it is not useful in the context
        # of this dataset. There are a number of considerably small magnitude
        # values that blow up the MAPE computation
        # writer.add_scalar(f"MAPE/train", train_mape, epoch)
        writer.add_scalar(f"MAE/val", val_mae, epoch)
        writer.add_scalar(f"RMSE/val", val_rmse, epoch)
        # writer.add_scalar(f"MAPE/val", val_mape, epoch)

        # Apply warm-up for the first epochs
        # if epoch < 5:
        #     warmup_scheduler.step()
        #     lr = warmup_scheduler.get_last_lr()[0]
        #     print(f"*** Warm up learning rate now at {lr}")
        # else:
        #     plateau_scheduler.step(loss)

        #     lr = plateau_scheduler.get_last_lr()[0]
        #     if lr != last_lr:
        #         print(f"*** Changing learning rate to {lr}")
        #         last_lr = lr

        # Step the cosine annealing learning rate scheduler
        if lr_scheduler:
            scheduler.step()

        # Check early stopping
        # if es(val_mae):
        #     stopped = epoch
        #     print(f"Early stopping at epoch {epoch + 1}")
        #     break

        # Add 1 to account for zero indexing
        if (epoch + 1) in checkpoint_epochs:
            # Save the model
            timestr = time.strftime("%m-%d-%H:%M:%S")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_type": optimizer.__class__.__name__,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_type": scheduler.__class__.__name__,
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": loss,
                    "train_rmse": train_rmse,
                    "train_mae": train_mae,
                    "val_rmse": val_rmse,
                    "val_mae": val_mae,
                    "attention_by_epoch": attn_matrices_by_epoch,
                },
                os.path.join(config["CHECKPOINT_DIR"], f"model_{timestr}.pt"),
            )

    writer.flush()

    return model


def model_test(model, test_dataloader, device, config):
    """
    Tests the ST-GAT model on a provided test dataset.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        test_dataloader (DataLoader): DataLoader providing batches of test data.
        device (torch.device): The device (CPU/GPU) to run the test on.
        config (dict): Configuration dictionary containing model and data settings.

    Returns:
        tuple: Returns the output from the `eval` function, which includes:
            - rmse (float): Average Root Mean Square Error on the test dataset.
            - mae (float): Average Mean Absolute Error on the test dataset.
            - mape (float): Average Mean Absolute Percentage Error on the test dataset.
            - y_pred (torch.Tensor): Collected predictions across all test batches.
            - y_truth (torch.Tensor): Collected ground truth labels across all test batches.
            - attn_matrix (np.ndarray or None): Attention matrix if applicable, otherwise None.
    """

    model.eval()
    return eval(model, device, test_dataloader, config, "Test")


def model_test_get_attention(model, anomalous_graph, device, config):
    """Runs a forward pass through the model for a single graph (single time step)
    to compute the attention matrix for that point in time.

    Args:
        model (_type_): _description_
        anomalous_graph (_type_): _description_
        device (_type_): _description_
        config (_type_): _description_
    """
    model.eval()
    anomalous_graph.to(device)
    with torch.no_grad():
        pred, edge_attention = model(anomalous_graph, device)
    ave_atn = torch.mean
    matrix = edge_attention_to_matrix(anomalous_graph, edge_attention.mean(dim=1))
    return matrix
