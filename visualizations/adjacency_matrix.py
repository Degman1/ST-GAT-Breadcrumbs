import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def plot_adjacency_matrix(matrix, filename="adjacency_matrix.png"):
    """
    Plots a binary adjacency matrix with white for 0 and black for 1.

    Args:
      matrix: A 2D NumPy array representing the binary adjacency matrix.
      filename: The desired filename for the saved image.
    """

    plt.figure(figsize=(8, 8))
    plt.imshow(matrix, cmap="binary", interpolation="nearest")

    # Determine appropriate tick spacing based on matrix size
    num_ticks = min(len(matrix), 5)  # Adjust 10 as needed
    tick_interval = len(matrix) // num_ticks

    plt.xticks(np.arange(0, len(matrix), tick_interval), fontsize=20)
    plt.yticks(np.arange(0, len(matrix), tick_interval), fontsize=20)

    plt.savefig(filename, bbox_inches="tight")
    print(f"Adjacency matrix plot saved to {filename}")


def get_subgraph(adj_matrix, node_indices):
    """
    Extracts the adjacency matrix for a subset of nodes.

    Parameters:
    - adj_matrix (numpy.ndarray): The original adjacency matrix (N x N).
    - node_indices (list or array): The indices of the nodes in the subgraph.

    Returns:
    - sub_adj_matrix (numpy.ndarray): The adjacency matrix for the subgraph.
    """
    node_indices = np.array(node_indices)  # Ensure it's an array
    return adj_matrix[np.ix_(node_indices, node_indices)]


if __name__ == "__main__":
    graph_path = "dataset/pruned_clustered_3hop_graph.adjlist"
    G = nx.read_adjlist(graph_path)
    graph_path = "dataset/clustered_G3Hops.adjlist"
    G2 = nx.read_adjlist(graph_path)
    subgraph = G2.subgraph(G.nodes()).copy()
    adj_mtx = nx.to_numpy_array(subgraph)
    plot_adjacency_matrix(adj_mtx)
    print(f"{G.number_of_edges()} edges.")
