import seaborn as sns
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def process_npz_file(npz_file, graph, dataset, npz_dir):
    """
    Process a single .npz file containing attention weights.
    """
    attn_weights = np.load(npz_file)["attn_avg"]  # Replace with the actual key
    num_time_slots_in_batch = attn_weights.shape[0] // dataset.n_edges

    attention_matrices = []
    for time_slot in range(num_time_slots_in_batch):
        start_idx = time_slot * dataset.n_edges
        end_idx = start_idx + dataset.n_edges
        sub_attn_weights = attn_weights[start_idx:end_idx, :]
        
        # Convert edge attention weights to node-to-node attention matrix
        attn_matrix = edge_attention_to_matrix(graph, sub_attn_weights.mean(axis=1))  # Average across heads
        attention_matrices.append(attn_matrix)
    
    return attention_matrices

def build_attention_matrices_multithreaded(dataset, config, epoch, npz_dir):
    graph = dataset.graphs[0]  # All graphs have the same structure

    npz_files = sorted(os.listdir(npz_dir))  # Get sorted .npz files
    npz_files = [os.path.join(npz_dir, f) for f in npz_files if f.endswith(".npz")]

    attention_matrices = []
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_npz_file, npz_file, graph, dataset, npz_dir) for npz_file in npz_files]
        with tqdm(total=len(futures), desc="Processing .npz files") as pbar_files:
            for future in futures:
                attention_matrices.extend(future.result())
                pbar_files.update(1)

    return attention_matrices



def edge_attention_to_matrix(graph, edge_attention):
    """
    Convert edge-based attention weights to a node-to-node adjacency matrix.

    :param graph: DGLGraph object (static structure used for edge definitions)
    :param edge_attention: Edge attention weights of shape (E,) or (E, 1), where E is the number of edges
    :return: Node-to-node attention matrix of shape (N, N), where N is the number of nodes in the graph
    """
    num_nodes = graph.num_nodes()
    adj_matrix = np.zeros((num_nodes, num_nodes))

    # Extract the source and destination node indices for edges
    src, dst = graph.edges()

    # Assign edge attention weights to the appropriate positions in the adjacency matrix
    adj_matrix[src, dst] = edge_attention.squeeze()  # Squeeze to handle shape (E, 1) or (E,)

    return adj_matrix


def build_attention_matrices(dataset, config, epoch, npz_dir):
    graph = dataset.graphs[0]  # All graphs have the same structure, so we use the first graph for the structure

    # Collect attention matrices for each time slot
    attention_matrices = []

    # epoch = config["EPOCHS"] - 1
    npz_files = sorted(os.listdir(npz_dir))  # Get sorted list of .npz files from directory
    npz_files = [os.path.join(npz_dir, f) for f in npz_files if f.endswith(".npz")]  # Filter to .npz files
    
    # Extract the numerical part from filenames for proper sorting
    npz_files = sorted(npz_files, key=lambda x: int(x.split('_batch_')[1].split('.')[0]))

    # Create a progress bar for the .npz files
    with tqdm(total=len(npz_files), desc="Building Attention Matrices") as pbar_files:
        for npz_file in npz_files:
            # Load the npz file containing attention weights
            attn_weights = np.load(npz_file)["attn_avg"]  # Replace with your specific key if needed
            
            assert attn_weights.shape[0] % dataset.n_edges == 0, \
                f"Invalid attention weights shape: {attn_weights.shape} for n_edges={dataset.n_edges}"

            # Calculate the number of time slots (i.e., number of sub-graphs in this batch)
            num_time_slots_in_batch = attn_weights.shape[0] // dataset.n_edges

            # Create a progress bar for each .npz file's time slots
            with tqdm(total=num_time_slots_in_batch, desc=f"Processing {os.path.basename(npz_file)}") as pbar_slots:
                for time_slot in range(num_time_slots_in_batch):
                    start_idx = time_slot * dataset.n_edges
                    end_idx = start_idx + dataset.n_edges

                    # Extract the corresponding attention weights for this time slot
                    sub_attn_weights = attn_weights[start_idx:end_idx, :]

                    # Convert edge attention weights to node-to-node attention matrix
                    attn_matrix = edge_attention_to_matrix(graph, sub_attn_weights)

                    # Store the attention matrix for this time slot
                    attention_matrices.append(attn_matrix)

                    # Update the time slot progress bar
                    pbar_slots.update(1)

            del attn_weights

            # Update the file progress bar
            pbar_files.update(1)
    
    # Save the attention matrices to a compressed .npz file
    # output_file = f"attention_matrices_epoch{epoch}.npz"
    print(f"Generated a total of {len(attention_matrices)} attention matrices")
    # np.savez_compressed(output_file, attention_matrices=attention_matrices)

    return attention_matrices


def plot_heatmap(attn_matrix, epoch_number, time_slot, custom_node_ids=None, display_step=50):
    """
    Plot a heatmap of the attention matrix.

    :param attn_matrix: Node-to-node attention matrix.
    :param epoch_number: The current epoch number for labeling.
    :param time_slot: The time slot associated with the attention matrix.
    :param custom_node_ids: Optional list of custom node IDs to display as axis labels.
    :param display_step: Step size to select custom node IDs for axis labels (default=50).
    """
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(attn_matrix, cmap="Blues", annot=False, fmt=".2f", cbar_kws={"shrink": 0.8})
    plt.title(f"Attention Heatmap (Time Slot {time_slot}, Epoch {epoch_number})")
    
    if custom_node_ids is not None:
        # Ensure node IDs are integers
        custom_node_ids = [int(node_id) for node_id in custom_node_ids]
        
        # Select spaced-out labels for better readability
        labels = [custom_node_ids[i] if i % display_step == 0 else "" for i in range(len(custom_node_ids))]
        
        ax.set_xticks(np.arange(len(custom_node_ids)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        ax.set_yticks(np.arange(len(custom_node_ids)))
        ax.set_yticklabels(labels, rotation=0)

        # Remove axis spines to avoid overlap
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Optionally adjust tick mark lengths
        ax.tick_params(axis="both", which="both", length=0)  # Removes tick marks

    plt.xlabel("Source Node")
    plt.ylabel("Destination Node")
    name = f"visualizations/attention_heatmap_epoch{epoch_number}_slot{time_slot}.png"
    plt.savefig(
        name, 
        bbox_inches="tight"
    )
    print(f"Attention heat map saved to {name}")
