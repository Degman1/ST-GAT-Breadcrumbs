import torch


def save_model(model, path="./model_checkpoint.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path="./trained_model.pth"):
    model.load_state_dict(torch.load(path))
    model.eval()  # Set model to evaluation mode
    print(f"Model loaded from {path}")
