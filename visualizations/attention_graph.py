import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import dgl
import torch


def normalize_weights(weights):
    """Normalize weights to the range [0, 1]."""
    min_weight = np.min(weights)
    max_weight = np.max(weights)
    return (weights - min_weight) / (max_weight - min_weight)


def visualize_attention_graph(attn_matrix, threshold=0):
    """
    Visualize a graph with attention weights using NetworkX from the attention matrix.

    :param attn_matrix: 2D matrix of attention weights (nodes x nodes)
    :param threshold: Minimum attention weight to consider an edge
    """
    # Create a NetworkX graph
    nx_graph = nx.Graph()

    # Add nodes (assuming the nodes are indexed from 0 to attn_matrix.shape[0] - 1)
    num_nodes = attn_matrix.shape[0]
    nx_graph.add_nodes_from(range(num_nodes))

    # Create edges based on the attention matrix (only keep weights above the threshold)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if attn_matrix[i, j] > threshold:  # Apply threshold to filter edges
                nx_graph.add_edge(i, j, weight=attn_matrix[i, j])

    # Get edge weights for visualization
    edge_weights = nx.get_edge_attributes(nx_graph, "weight")

    # Prepare edge colors and widths for drawing
    edge_colors = []
    scaled_weights = []  # To hold the scaled weights for normalization
    for edge in edge_weights.keys():
        # Collect weights for normalization
        scaled_weights.append(edge_weights[edge])

    # Apply logarithmic scaling and normalization
    log_weights = np.log1p(scaled_weights)  # Logarithmic transformation
    normalized_weights = normalize_weights(log_weights)  # Normalize to [0, 1]

    for idx, edge in enumerate(edge_weights.keys()):
        # Create an RGB color based on the normalized weight
        rgb_color = (
            1 - normalized_weights[idx],
            0,
            normalized_weights[idx],
        )  # Gradient from red to blue
        edge_colors.append(rgb_color)

    # Scale edge widths and adjust for less congestion
    edge_widths = [
        normalized_weights[idx] * 5 for idx in range(len(edge_weights))
    ]  # Adjust scaling factor for edge width

    # Set the figure size
    plt.figure(figsize=(12, 12))  # Increase figure size for better visibility

    # Draw the graph with adjusted sizes
    pos = nx.spring_layout(nx_graph)  # Use a spring layout for better visualization
    nx.draw(
        nx_graph,
        pos,
        with_labels=True,
        node_size=100,  # Decrease node size
        width=edge_widths,  # Edge width
        edge_color=edge_colors,
    )  # Edge color based on RGB

    # Create a custom colormap that goes from red to blue
    custom_cmap = LinearSegmentedColormap.from_list(
        "red_blue", [(1, 0, 0), (0, 0, 1)], N=256
    )

    # Create a ScalarMappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=custom_cmap)
    sm.set_array([0, 1])  # Set array to the range of values for color mapping

    # Add colorbar with axes specified
    plt.colorbar(sm, ax=plt.gca(), label="Attention Weight")  # Add colorbar

    plt.title("Graph with Attention Matrix Weights")
    plt.savefig(
        "visualizations/networkx_attention_graph", bbox_inches="tight"
    )  # Save the figure
    plt.show()


def visualize_attention_neighborhood(
    graph, attn_weights, center_node, epoch, time_slot, k=2
):
    """
    Visualize the k-hop neighborhood of a specific node with attention weights.

    :param graph: DGL graph object
    :param attn_weights: Attention weights from the GAT layer (edges x heads)
    :param center_node: The node at the center of the neighborhood
    :param k: Number of hops to consider in the neighborhood
    :param head_idx: Index of the attention head to visualize
    """
    # Extract the k-hop neighborhood around the center_node
    neighbors, _ = dgl.khop_in_subgraph(graph, center_node, k)  # Unpack the subgraph

    # Create a NetworkX graph from the neighborhood
    nx_neighborhood = neighbors.to_networkx()

    # Get the edges of the original graph and the subgraph
    original_edges = graph.edges()
    neighborhood_edges = nx_neighborhood.edges()

    # Create a mapping from edges to their indices in the original graph
    edge_to_index = {
        (u.item(), v.item()): i for i, (u, v) in enumerate(zip(*original_edges))
    }

    attn_avg = attn_weights.mean(dim=1)

    # Get the attention weights for the neighborhood's edges
    attn_neighborhood_weights = []

    for u, v in neighborhood_edges:
        # Get the index of the edge (u, v) from the mapping
        edge_index = edge_to_index.get((u, v))
        if edge_index is not None:
            attn_neighborhood_weights.append(attn_avg[edge_index].item())
        else:
            attn_neighborhood_weights.append(
                0
            )  # Assign a default value if the edge is not found

    # Convert the edge weights to a numpy array
    attn_neighborhood_weights = torch.tensor(attn_neighborhood_weights).numpy()

    # Visualize the neighborhood
    pos = nx.spring_layout(nx_neighborhood)  # Use a fast layout
    plt.figure(figsize=(10, 8))  # Adjust figure size
    nx.draw(
        nx_neighborhood,
        pos,
        with_labels=True,
        node_size=50,  # Reduce node size for clarity
        width=[
            attn_neighborhood_weights[i] * 5
            for i in range(len(attn_neighborhood_weights))
        ],  # Edge width by attention
        edge_color=attn_neighborhood_weights,  # Color edges by attention
        edge_cmap=plt.cm.Blues,
    )
    plt.title(
        f"{k}-Hop Neighborhood of Node {center_node} (Timeslot {time_slot}, Epoch: {epoch})"
    )
    plt.savefig(
        f"visualizations/networkx_attention_graph_{k}-hop_center-{center_node}_timeslot-{time_slot}_epoch-{epoch}",
        bbox_inches="tight",
    )  # Save the figure

    plt.show()
