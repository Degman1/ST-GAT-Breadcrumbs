import torch
import os
import numpy as np


# Location to save batch-wise attention values
ATTENTION_SAVE_DIR = "attn_values"


def save_model(model, path="./model_checkpoint.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path="./trained_model.pth"):
    model.load_state_dict(torch.load(path))
    model.eval()  # Set model to evaluation mode
    print(f"Model loaded from {path}")


def save_attention_to_disk(attn, batch_idx, epoch):
    """
    Save attention matrix to disk.
    :param attn: Attention matrix (tensor)
    :param batch_idx: Batch index
    :param epoch: Current epoch
    :param ATTENTION_SAVE_DIR: Directory to save the matrices
    """
    os.makedirs(ATTENTION_SAVE_DIR, exist_ok=True)
    filename = f"{ATTENTION_SAVE_DIR}/epoch_{epoch}_batch_{batch_idx}.pt"
    torch.save(attn, filename)
    

def save_attention_compressed(attn, batch_idx, epoch):
    os.makedirs(ATTENTION_SAVE_DIR, exist_ok=True)
    filename = f"{ATTENTION_SAVE_DIR}/epoch_{epoch}_batch_{batch_idx}.npz"
    np.savez_compressed(filename, attn=attn.numpy())
    
    
def get_attention_filename(epoch, batch_idx):
    return f"{ATTENTION_SAVE_DIR}/epoch_{epoch}_batch_{batch_idx}.npz"
    
def save_attention_avg_compressed(attn, batch_idx, epoch):
    """
    Save the average attention matrix across heads in a compressed format.
    :param attn: Attention matrix (nodes x nodes x heads)
    :param batch_idx: Batch index
    :param epoch: Current epoch
    :param ATTENTION_SAVE_DIR: Directory to save the matrices
    """
    os.makedirs(ATTENTION_SAVE_DIR, exist_ok=True)
    filename = get_attention_filename(epoch, batch_idx)
    
    attn_avg = attn.mean(dim=1)  # Average along the head dimension

    # Save as compressed .npz file
    np.savez_compressed(filename, attn_avg=attn_avg.cpu().numpy())
