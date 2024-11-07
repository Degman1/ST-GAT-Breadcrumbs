import dgl
import torch
import numpy as np
import pandas as pd
import os
from dgl.data import DGLDataset

from utils.math import z_score
from . import splits


def distance_to_weight(W, sigma2=0.1, epsilon=0.5, gat_version=False):
    n = W.shape[0]
    W = W / 10000.0
    W2, W_mask = W * W, np.ones([n, n]) - np.identity(n)
    W = np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask

    if gat_version:
        W[W > 0] = 1
        W += np.identity(n)

    return W


class TrafficDataset(DGLDataset):
    def __init__(
        self,
        config,
        W,
        root="",
        force_reload=False,
        verbose=False,
        fully_connected=False,
    ):
        self.config = config
        self.W = W
        self.graphs = []
        self.n_nodes = None
        self.n_edges = None
        self.mean = None
        self.std_dev = None
        self.fully_connected = fully_connected
        super().__init__(
            name="traffic_dataset",
            raw_dir=root,
            force_reload=force_reload,
            verbose=verbose,
        )

    def process(self):
        data = pd.read_csv(self.raw_file_names[0], header=None).values
        self.mean = np.mean(data)
        self.std_dev = np.std(data)
        data = z_score(data, self.mean, self.std_dev)

        _, self.n_nodes = data.shape
        n_window = self.config["N_PRED"] + self.config["N_HIST"]

        edge_index = torch.zeros((2, self.n_nodes**2), dtype=torch.long)
        edge_attr = torch.zeros((self.n_nodes**2, 1))
        self.n_edges = 0
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if self.fully_connected or self.W[i, j] != 0.0:
                    edge_index[0, self.n_edges] = i
                    edge_index[1, self.n_edges] = j
                    edge_attr[self.n_edges] = self.W[i, j]
                    self.n_edges += 1
        edge_index = edge_index[:, : self.n_edges]
        edge_attr = edge_attr[: self.n_edges]

        self.graphs = []
        for i in range(self.config["N_DAYS"]):
            for j in range(self.config["N_SLOT"]):
                # for each time point construct a different graph with data object
                g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=self.n_nodes)
                g.edata["weight"] = edge_attr

                sta = i * self.config["N_DAY_SLOT"] + j
                end = sta + n_window
                full_window = np.swapaxes(data[sta:end, :], 0, 1)
                g.ndata["feat"] = torch.FloatTensor(
                    full_window[:, : self.config["N_HIST"]]
                )
                g.ndata["label"] = torch.FloatTensor(
                    full_window[:, self.config["N_HIST"] :]
                )

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
        return [os.path.join(self.raw_dir, "PeMSD7_V_228.csv")]

    @property
    def processed_file_names(self):
        return ["traffic_data.pt"]

    @property
    def processed_paths(self):
        return [os.path.join(self.raw_dir, f) for f in self.processed_file_names]

    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.graphs)


def get_processed_dataset(config):
    # Number of possible windows in a day
    config["N_SLOT"] = config["N_DAY_SLOT"] - (config["N_PRED"] + config["N_HIST"]) + 1

    # Load the weight and dataset
    distances = pd.read_csv("./dataset/PeMSD7_W_228.csv", header=None).values
    W = distance_to_weight(distances, gat_version=config["USE_GAT_WEIGHTS"])
    dataset = TrafficDataset(
        config, W, root="./dataset", force_reload=False, fully_connected=config["FULLY_CONNECTED"]
    )

    d_mean = dataset.mean
    d_std_dev = dataset.std_dev

    # total of 44 days in the dataset, use 34 for training, 5 for val, 5 for test
    d_train, d_val, d_test = splits.get_splits_window(dataset, config["N_SLOT"], (34, 5, 5))

    return dataset, d_mean, d_std_dev, d_train, d_val, d_test
