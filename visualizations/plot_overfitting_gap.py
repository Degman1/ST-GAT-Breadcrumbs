import json
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON file
with open("../results.json", "r") as f:
    data = json.load(f)["Results"]

# Filter out entries with train_percent == 0.1
filtered_data = [entry for entry in data if entry["train_percent"] != 0.1]

# Extract train_percent and corresponding loss values
train_percents = [(entry["train_percent"] * 2317.0 / 24.0) for entry in filtered_data]
train_loss = [entry["MAE/train"] for entry in filtered_data]
val_loss = [entry["MAE/val"] for entry in filtered_data]

# Plotting
plt.figure(figsize=(8, 3))
plt.plot(train_percents, train_loss, marker="s", label="Training MAE")
plt.plot(train_percents, val_loss, marker="s", label="Validation MAE")

# Add vertical lines and annotate absolute differences
for x, y_train, y_val in zip(train_percents, train_loss, val_loss):
    plt.vlines(x, y_train, y_val, colors="gray", linestyles="dotted")
    diff = abs(y_val - y_train)
    mid = (y_val + y_train) / 2
    plt.text(
        x,
        mid,
        f"{diff:.4f}",
        ha="center",
        va="center",
        fontsize=10,
        backgroundcolor="white",
    )
plt.xticks(ticks=range(int(min(train_percents)) + 1, int(max(train_percents) + 1), 5))

plt.xlabel("Training Dataset Size (Days)", fontsize=12)
plt.ylabel("Valid. MAE - Train MAE", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
filename = f"concept_drift_future2.png"
plt.tight_layout()
plt.savefig(filename, bbox_inches="tight")
print(f"Figure saved to {filename}")
