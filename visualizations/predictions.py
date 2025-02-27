import torch
import matplotlib.pyplot as plt


def plot_prediction(test_dataloader, y_pred, y_truth, node, node_label, rank, config, num_days=None):
    # Calculate the truth
    s = y_truth.shape
    y_truth = y_truth.reshape(s[0], config["BATCH_SIZE"], config["N_NODE"], s[-1])
    # just get the first prediction out for the nth node
    y_truth = y_truth[:, :, node, 0]
    # Flatten to get the predictions for entire test dataset
    y_truth = torch.flatten(y_truth)

    # Calculate the predicted
    s = y_pred.shape
    y_pred = y_pred.reshape(s[0], config["BATCH_SIZE"], config["N_NODE"], s[-1])
    # just get the first prediction out for the nth node
    y_pred = y_pred[:, :, node, 0]
    # Flatten to get the predictions for entire test dataset
    y_pred = torch.flatten(y_pred)
    
    # Determine the total number of available days
    total_slots = len(y_truth)
    total_days = total_slots // config["SLOTS_PER_DAY"]  # Number of full days available

    # Determine how many days to visualize
    if num_days is None or num_days > total_days:
        num_days = total_days  # Default to all available days

    # Slice the data to the required number of days
    end_idx = num_days * config["SLOTS_PER_DAY"]
    y_truth = y_truth[:end_idx]
    y_pred = y_pred[:end_idx]
    
    t = [t for t in range(0, end_idx)]
    plt.figure(figsize=(10, 5))
    plt.plot(t, y_truth, label="Truth", linestyle="solid", color="red")
    plt.plot(t, y_pred, label="ST-GAT", linestyle="dashed", color="blue")
    plt.xlabel("Time (Hours After Test Data Collection Start Timestamp)")
    plt.ylabel("Population Density")
    plt.title(f"Predictions of Population Density for Node {node_label} Over {num_days} Days")
    plt.legend()
    loc = f"./visualizations/predictions/rank{rank}_node{node_label}_predicted_densities.png"
    plt.savefig(loc)
    plt.close()
    
    print(f"Prediction visualization for node {node_label} saved to {loc}")
