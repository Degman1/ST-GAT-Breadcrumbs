import seaborn as sns
import matplotlib.pyplot as plt
import torch


def edge_attention_to_matrix(graph, attn_weights):
    """
    Convert edge attention weights into a node-to-node attention matrix.

    :param graph: DGL graph object
    :param attn_weights: Attention weights from the GAT layer (edges x heads)
    :param num_nodes: Number of nodes in the graph
    :return: Node-to-node attention matrix (adjacency matrix with attention values)
    """
    # Extract the source and destination nodes for each edge
    src, dst = graph.edges()

    # Initialize the attention matrix with zeros
    attn_matrix = torch.zeros(graph.number_of_nodes(), graph.number_of_nodes())

    avg_attn_weights = attn_weights.mean(dim=1)  # Average across heads

    # Fill the attention matrix using the attention weights
    for i in range(len(src)):
        attn_matrix[src[i], dst[i]] = avg_attn_weights[i]

    return attn_matrix


def build_attention_matrices(dataset, attn_matrices_by_batch_by_epoch, config, epoch):
    graph = dataset.graphs[
        0
    ]  # All graphs have the same structure, but different features,
    # so use the first graph for the structure

    # Collect attention matrices for each time slot
    attention_matrices = []

    # epoch = config["EPOCHS"] - 1

    for batch in attn_matrices_by_batch_by_epoch[epoch]:  # Loop over batches
        # Get the combined graph and attention weights from the batch
        batch_graph = batch["batch"].cpu()  # This is a single large graph for the batch
        attn_weights = batch[
            "attn"
        ].cpu()  # Attention weights corresponding to the entire batch

        # Handle the last batch which might be smaller
        num_time_slots = (
            batch_graph.number_of_nodes() // dataset.n_nodes
        )  # Number of time slots in this batch

        for time_slot in range(num_time_slots):
            start_idx = time_slot * dataset.n_edges
            end_idx = start_idx + dataset.n_edges

            # Extract the corresponding attention weights for this time slot
            sub_attn_weights = attn_weights[
                start_idx:end_idx, :
            ]  # Subset of attention weights for this time slot

            # Convert edge attention weights to node-to-node attention matrix
            attn_matrix = edge_attention_to_matrix(graph, sub_attn_weights)

            # Store the attention matrix for this time slot
            attention_matrices.append(attn_matrix)

    return attention_matrices

def plot_heatmap(attn_matrix, epoch_number, time_slot):
    """
    Plot a heatmap of the attention matrix.

    :param attn_matrix: Node-to-node attention matrix.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_matrix.cpu().numpy(), cmap="Blues", annot=False, fmt=".2f")
    plt.title(f"Attention Heatmap (Time Slot {time_slot}, Epoch {epoch_number})")
    plt.xlabel("Source Node")
    plt.ylabel("Destination Node")
    plt.savefig(
        "visualizations/attention_heatmap", bbox_inches="tight"
    )  # Save the figure
    plt.show()
