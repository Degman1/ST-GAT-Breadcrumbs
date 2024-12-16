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
  plt.imshow(matrix, cmap='gray', interpolation='nearest')
  plt.title("Adjacency Matrix")

  # Determine appropriate tick spacing based on matrix size
  num_ticks = min(len(matrix), 10)  # Adjust 10 as needed
  tick_interval = len(matrix) // num_ticks

  plt.xticks(np.arange(0, len(matrix), tick_interval))
  plt.yticks(np.arange(0, len(matrix), tick_interval))

  plt.savefig(filename)
  plt.close()
    
if __name__ == "__main__":
    G = nx.read_adjlist("../dataset/G3Hops_full.adjlist")
    adj_mtx = nx.to_numpy_array(G)
    plot_adjacency_matrix(adj_mtx)