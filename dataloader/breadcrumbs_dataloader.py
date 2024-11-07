import dgl
import torch
import numpy as np
import pandas as pd
import os
from dgl.data import DGLDataset

from utils.math import z_score
from . import splits


class BreadcrumbsDataset(DGLDataset):
    def __init__(
        self,
        config,
        root="",
        force_reload=False,
        verbose=False,
    ):
        self.config = config
        self.graphs = []
        self.n_nodes = None
        self.n_edges = None
        self.mean = None
        self.std_dev = None
        super().__init__(
            name="breadcrumbs_dataset",
            raw_dir=root,
            force_reload=force_reload,
            verbose=verbose,
        )

    def process(self):
        data = pd.read_csv(self.raw_file_names[0], header=None).values
        self.mean = np.mean(data)
        self.std_dev = np.std(data)
        data = z_score(data, self.mean, self.std_dev)

        self.n_times, self.n_nodes = data.shape

        # TODO read in adjacency matrix
        adj_mtx = pd.read_csv("...").values

        # Get the indices of non-zero elements (edges)
        src, dst = np.nonzero(adj_mtx)

        # Initialize edge index arrays based on non-zero elements
        num_edges = len(src)
        self.n_edges = self.n_nodes ** 2    # Will always be using the fully connected graph
        edge_index = torch.zeros((2, self.n_edges), dtype=torch.long)
        # Optionally if want to add edge attributes
        edge_attr = torch.zeros((self.n_edges, 1))

        # Populate edge indices
        edge_index[0, :] = src
        edge_index[1, :] = dst

        # Optionally add edge weights if needed
        # for i in range(num_edges):
        #     edge_attr[i] = adj_mtx[src[i], dst[i]]

        self.n_pred = self.config["N_PRED"]
        self.n_hist = self.config["N_HIST"]

        # Iterate through each time window to create graphs
        n_timepoints = self.data.shape[0]
        for t in range(self.n_hist, n_timepoints - self.n_pred):
            # Create a new graph based on the adjacency matrix structure
            g = dgl.graph((self.edge_index[0], self.edge_index[1]), num_nodes=self.n_nodes)

            # Set edge weights if needed (optional)
            # g.edata["weight"] = torch.FloatTensor(edge_attr)

            # Create a full window of data: history + prediction
            full_window = self.data[t - self.n_hist:t + self.n_pred, :]  # Shape: (n_hist + n_pred, n_nodes)
            full_window = np.swapaxes(full_window, 0, 1)  # Shape: (n_nodes, n_hist + n_pred)

            # Node features: last `n_hist` time samples (history)
            g.ndata["feat"] = torch.FloatTensor(full_window[:, :self.n_hist])

            # Node labels: next `n_pred` time samples (prediction)
            g.ndata["label"] = torch.FloatTensor(full_window[:, self.n_hist:])

            # Append graph to the list
            self.graphs.append(g)

        print("Completed Data Preprocessing")

    def save(self):
        # Save processed dataset
        torch.save(
            {
                "graphs": self.graphs,
                "n_nodes": self.n_nodes,
                "mean": self.mean,
                "std_dev": self.std_dev,
            },
            self.processed_paths[0],
        )

    def load(self):
        # Load the saved dataset
        self.graphs, self.n_nodes, self.mean, self.std_dev = torch.load(
            self.processed_paths[0]
        )

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        print(f"Found preprocessed data to load at {self.processed_paths[0]}.")
        return os.path.exists(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [os.path.join(self.raw_dir, "dataset/timeseriesByPOI.csv")]

    @property
    def processed_file_names(self):
        return ["breadcrumbs_data.pt"]

    @property
    def processed_paths(self):
        return [os.path.join(self.raw_dir, f) for f in self.processed_file_names]

    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.graphs)
    

def get_processed_dataset(config):
    # Number of possible windows in a day
    
    dataset = BreadcrumbsDataset(
        config, root="../dataset"
    )

    d_mean = dataset.mean
    d_std_dev = dataset.std_dev

    # total of 44 days in the dataset, use 34 for training, 5 for val, 5 for test
    # TODO
    # d_train, d_val, d_test = splits.get_splits(dataset, config["N_SLOT"], (34, 5, 5))

    return dataset, d_mean, d_std_dev, d_train, d_val, d_test