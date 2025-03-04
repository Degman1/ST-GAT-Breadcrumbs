import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_adjacency_matrix(matrix, filename="adjacency_matrix.png"):
  """
  Plots a binary adjacency matrix with white for 0 and black for 1.

  Args:
    matrix: A 2D NumPy array representing the binary adjacency matrix.
    filename: The desired filename for the saved image.
  """

  plt.figure(figsize=(8, 6))
  plt.imshow(matrix, cmap='binary', interpolation='nearest')
  plt.title("Adjacency Matrix")

  # Determine appropriate tick spacing based on matrix size
  num_ticks = min(len(matrix), 10)  # Adjust 10 as needed
  tick_interval = len(matrix) // num_ticks

  plt.xticks(np.arange(0, len(matrix), tick_interval))
  plt.yticks(np.arange(0, len(matrix), tick_interval))

  plt.savefig(filename)
  plt.close()

def get_subgraph_adjacency(adj_matrix, node_indices):
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
    G = nx.read_adjlist("../dataset/clustered_G3Hops.adjlist")
    adj_mtx = nx.to_numpy_array(G)
    plot_adjacency_matrix(adj_mtx)