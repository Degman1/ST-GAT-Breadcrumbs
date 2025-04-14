import dgl
import torch
import numpy as np
import pandas as pd
import os
from dgl.data import DGLDataset
import networkx as nx
import matplotlib.pyplot as plt

from utils.math import z_score
from . import splits


class BreadcrumbsDataset(DGLDataset):
    def __init__(
        self, config, root="", force_reload=False, verbose=False, node_subset=None
    ):
        self.config = config
        self.graphs = []
        self.n_nodes = None
        self.n_edges = None
        self.mean = None
        self.std_dev = None
        self.node_subset = node_subset
        super().__init__(
            name="breadcrumbs_dataset",
            raw_dir=root,
            force_reload=force_reload,
            verbose=verbose,
        )

    def process(self):
        data_pd = pd.read_csv(self.raw_file_names[0], index_col=0, header=0)
        if self.node_subset is not None:
            selected_node_ids = [str(node_id) for node_id in self.node_subset]
            data_pd = data_pd[selected_node_ids]
        data = data_pd.values

        self.mean = np.mean(data)
        self.std_dev = np.std(data)
        data = z_score(data, self.mean, self.std_dev)

        # Compute per-node mean and sparsity
        node_means = data_pd.mean(axis=0)
        zero_percent = (data_pd == 0).sum(axis=0) / len(data_pd) * 100

        plt.figure(figsize=(4, 6))
        plt.boxplot(node_means, vert=True, patch_artist=True, showfliers=False)
        plt.title("Density Mean Distribution")
        plt.ylabel("Mean Density")
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig("output/boxplot_node_means.png", dpi=300)

        plt.figure(figsize=(4, 6))
        plt.boxplot(zero_percent, vert=True, patch_artist=True, showfliers=False)
        plt.title("Density Sparcity Distribution")
        plt.ylabel("Sparcity Percent")
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig("output/boxplot_node_sparsity.png", dpi=300)

        # Get top 60 nodes with highest mean
        top_60_mean_nodes = node_means.sort_values(ascending=False).head(60)

        # Get top 60 nodes with lowest non-zero %
        top_60_sparse_nodes = zero_percent.sort_values(ascending=True).head(60)

        plt.figure(figsize=(4, 6))
        plt.boxplot(top_60_mean_nodes, vert=True, patch_artist=True, showfliers=False)
        plt.title("Mean Density Distribution (Top-60)")
        plt.ylabel("Mean Density")
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig("output/boxplot_top60_means.png", dpi=300)

        plt.figure(figsize=(4, 6))
        plt.boxplot(top_60_sparse_nodes, vert=True, patch_artist=True, showfliers=False)
        plt.title("Density Sparcity Percent (Top-60)")
        plt.ylabel("Sparcity Percentage")
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig("output/boxplot_top60_sparsity.png", dpi=300)

        self.n_times, self.n_nodes = data.shape

        if self.node_subset is None:
            G = nx.read_adjlist(self.raw_file_names[1])
            node_ids = list(G.nodes())
            int_node_ids = [int(i) for i in node_ids]

            # Assert that the number of node IDs matches the number of nodes in the adjacency matrix
            assert len(int_node_ids) == self.n_nodes, (
                f"Mismatch between the number of node IDs ({len(int_node_ids)}) "
                f" in the DGL graph and the number of nodes ({self.n_nodes}) in the timeseries dataset."
            )

            adj_mtx = nx.to_numpy_array(G, nodelist=node_ids)

            # Get the indices of non-zero elements (edges)
            src, dst = np.nonzero(adj_mtx)
        else:
            num_selected = len(self.node_subset)
            node_ids = sorted(self.node_subset, key=int)
            int_node_ids = [int(i) for i in node_ids]
            src, dst = zip(
                *[(i, j) for i in range(num_selected) for j in range(num_selected)]
            )

        # Initialize edge index arrays based on non-zero elements
        self.n_edges = len(src)
        edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)

        # TODO Optionally if want to add edge attributes
        # edge_attr = torch.zeros((self.n_edges, 1))
        # for i in range(num_edges):
        #     edge_attr[i] = adj_mtx[src[i], dst[i]]

        self.n_pred = self.config["N_PRED"]
        self.n_hist = self.config["N_HIST"]

        # Load timestamps from CSV row headers (index)
        self.full_timestamps = pd.to_datetime(
            pd.read_csv(self.raw_file_names[0], index_col=0).index,
            format="%Y-%m-%dT-%H-%M",
        )

        # Iterate through each time window to create graphs
        n_timepoints = data.shape[0]

        self.graph_timestamps = self.full_timestamps[
            self.n_hist : n_timepoints - self.n_pred + 1
        ]

        for t in range(self.n_hist, n_timepoints - self.n_pred + 1):
            # Create a new graph based on the adjacency matrix structure
            g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=self.n_nodes)
            # Self loops are now added in
            # g = dgl.add_self_loop(g)

            # Set edge weights (optional)
            # g.edata["weight"] = torch.FloatTensor(edge_attr)

            # Create a full window of data: history + prediction
            full_window = data[
                t - self.n_hist : t + self.n_pred, :
            ]  # Shape: (n_hist + n_pred, n_nodes)
            full_window = np.swapaxes(
                full_window, 0, 1
            )  # Shape: (n_nodes, n_hist + n_pred)

            # Node features: last `n_hist` time samples (history)
            g.ndata["feat"] = torch.FloatTensor(full_window[:, : self.n_hist])

            # Node labels: next `n_pred` time samples (prediction)
            g.ndata["label"] = torch.FloatTensor(full_window[:, self.n_hist :])

            # Tag the nodes with the original POI ids
            g.ndata["id"] = torch.IntTensor(int_node_ids)

            # Assign datetime at the graph level as the time at the first ground truth prediction
            g.graph_data = {"datetime": self.full_timestamps[t]}

            # Append graph to the list
            self.graphs.append(g)

        print(
            f"Generated {len(self.graphs)} different graphs from the normalized timeseries data each "
            f"with the same {self.graphs[0].number_of_nodes()} nodes and {self.graphs[0].number_of_edges()} edges."
        )

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
        return [
            os.path.join(self.raw_dir, "pruned_clustered_3hop_timeseries.csv"),
            os.path.join(self.raw_dir, "pruned_clustered_3hop_graph.adjlist"),
        ]

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


def get_processed_dataset(config, node_subset=None):
    # Number of possible windows in a day

    dataset = BreadcrumbsDataset(
        config, root="./dataset", force_reload=True, node_subset=node_subset
    )

    d_mean = dataset.mean
    d_std_dev = dataset.std_dev

    ratios = (0.7, 0.1, 0.2)
    d_train, d_val, d_test = splits.get_splits(dataset, (0.7, 0.1, 0.2))

    return dataset, d_mean, d_std_dev, d_train, d_val, d_test
