import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_tensorboard_scalars(logdir, tag="MAE/val"):
    """
    Extracts scalar data for a given tag from a TensorBoard log directory.

    :param logdir: Path to the TensorBoard event directory.
    :param tag: Scalar tag to extract (e.g., 'Loss/train', 'Loss/val').
    :return: (steps, values) as two lists.
    """
    ea = EventAccumulator(logdir)
    ea.Reload()
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def plot_multiple_runs(
    logdirs_dict, tag="Loss/val", title=None, ylabel="Loss", save_path=None
):
    """
    Plot multiple TensorBoard scalar runs on the same graph.

    :param logdirs_dict: Dictionary mapping labels to log directories.
    :param tag: The scalar tag to extract from TensorBoard logs.
    :param title: Title of the plot.
    :param ylabel: Y-axis label.
    :param save_path: Path to save the figure as a PDF (optional).
    """
    plt.figure(figsize=(8, 5))

    for label, logdir in logdirs_dict.items():
        if not os.path.exists(logdir):
            print(f"Warning: {logdir} does not exist.")
            continue
        steps, values = extract_tensorboard_scalars(logdir, tag)
        plt.plot(steps, values, label=label)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title if title else f"{tag} over Epochs")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


# Example usage
if __name__ == "__main__":
    logdirs = {
        "High LR": "runs/high_lr",
        "Low LR": "runs/low_lr",
    }
    plot_multiple_runs(
        logdirs_dict=logdirs,
        tag="Loss/val",
        title="Validation Loss (MSE) Comparison",
        ylabel="Validation MSE",
        save_path="val_loss_comparison.pdf",
    )
