class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        """
        patience: Number of epochs to wait after last improvement.
        min_delta: Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss):
        # Check if validation loss improved
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset counter on improvement
        else:
            self.counter += 1
        
        # Stop if patience exceeded
        return self.counter >= self.patience
