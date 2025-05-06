import networkx as nx
import pandas as pd
import json
import matplotlib.pyplot as plt


def prune_graph_and_timeseries(
    graph_path, timeseries_path, min_degree, output_graph_path, output_timeseries_path
):
    """
    Prunes a graph by removing nodes with a degree less than `min_degree`
    and updates the timeseries data by removing the corresponding columns.

    Args:
        graph_path (str): Path to the input adjlist file.
        timeseries_path (str): Path to the timeseries CSV file.
        min_degree (int): Minimum degree to retain nodes in the graph.
        output_graph_path (str): Path to save the pruned graph in adjlist format.
        output_timeseries_path (str): Path to save the updated timeseries CSV file.
        label_mapping_path (str, optional): Path to the label remapping JSON file. If provided, apply the node ID remapping.
    """
    # Step 1: Read the graph
    G = nx.read_adjlist(graph_path)

    # Step 2: Read the timeseries data
    timeseries_df = pd.read_csv(timeseries_path, index_col=0)

    # Step 3: Ensure column IDs in the timeseries DataFrame are strings
    timeseries_df.columns = timeseries_df.columns.astype(str)

    # Step 4: Identify nodes to retain
    sparsities = []
    degrees = []
    nodes_to_keep = []
    prune_degree_count = 0
    prune_sparsity_count = 0
    for node in G.nodes:
        degrees.append(G.degree(node))
        if G.degree(node) < min_degree:
            prune_degree_count += 1
            continue  # Skip low-degree nodes

        if node in timeseries_df.columns:
            series = timeseries_df[node]
            sparsity = (series < 0.0001).sum() / len(series)
            sparsities.append(sparsity)
            if sparsity > 0.85:
                prune_sparsity_count += 1
                continue  # Skip high-sparsity nodes

            nodes_to_keep.append(node)

    # Sort the data
    degrees_sorted = sorted(degrees, reverse=True)
    sparsities_sorted = sorted(sparsities)

    # Create side-by-side subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    # Plot degree distribution
    axs[0].plot(degrees_sorted)
    axs[0].set_xlabel("Nodes (Sorted By Degree)", fontsize=18)
    axs[0].set_ylabel("Degree", fontsize=18)
    axs[0].axhline(
        y=5,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Pruning Threshold (<5)",
    )
    axs[0].legend(fontsize=18)
    axs[0].grid(True)

    # Plot sparsity distribution
    axs[1].plot(sparsities_sorted)
    axs[1].set_xlabel("Nodes (Sorted By Sparsity)", fontsize=18)
    axs[1].set_ylabel("Sparsity (Fraction of Near-Zero Values)", fontsize=18)
    axs[1].axhline(
        y=0.85,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Pruning Threshold (>0.85)",
    )
    axs[1].legend(fontsize=18)
    axs[1].grid(True)

    axs[0].tick_params(axis="both", labelsize=16)
    axs[1].tick_params(axis="both", labelsize=16)

    plt.tight_layout()
    plt.savefig("output/degree_sparsity_distribution.png", dpi=300)
    plt.close()

    print(
        f"Pruning {prune_degree_count} due to low degree and {prune_sparsity_count} due to high sparsity"
    )

    print(f"{len(nodes_to_keep)} nodes remaining after pruning.")

    # Step 5: Create a subgraph with the retained nodes
    pruned_graph = G.subgraph(nodes_to_keep).copy()

    # Step 6: Write the pruned graph to a new adjlist file
    nx.write_adjlist(pruned_graph, output_graph_path)

    # Step 7: Filter columns corresponding to retained nodes
    nodes_to_keep = list(
        map(str, nodes_to_keep)
    )  # Ensure consistent formatting with timeseries
    filtered_timeseries_df = timeseries_df[nodes_to_keep]

    # Step 8: Save the filtered timeseries to a new CSV file
    filtered_timeseries_df.to_csv(output_timeseries_path)

    print(f"Pruned graph saved to {output_graph_path}")
    print(f"Filtered timeseries data saved to {output_timeseries_path}")


def validate_graph_and_timeseries(graph_path, timeseries_path):
    """
    Validates that all node IDs in the graph are present as column IDs in the timeseries DataFrame.
    Optionally applies a label remapping if provided.

    Args:
        graph_path (str): Path to the graph adjlist file.
        timeseries_path (str): Path to the timeseries CSV file.
        label_mapping_path (str, optional): Path to the label remapping JSON file.

    Raises:
        ValueError: If there are node IDs in the graph that are not in the timeseries DataFrame columns or vice versa.
    """
    # Step 1: Load the graph
    G = nx.read_adjlist(graph_path)

    # Step 2: Load the timeseries data
    timeseries_df = pd.read_csv(timeseries_path, index_col=0)

    # Step 4: Ensure column IDs in the timeseries DataFrame are strings
    timeseries_df.columns = timeseries_df.columns.astype(str)

    # Step 5: Check if all node IDs are in the DataFrame's columns
    graph_nodes = set(G.nodes())
    timeseries_columns = set(timeseries_df.columns)

    missing_in_graph = graph_nodes - timeseries_columns
    missing_in_timeseries = timeseries_columns - graph_nodes

    if missing_in_graph:
        raise ValueError(
            f"The following node IDs from the graph are not present as columns in the timeseries DataFrame: {missing_in_graph}"
        )

    if missing_in_timeseries:
        raise ValueError(
            f"The following node IDs from the timeseries DataFrame are not present in the graph: {missing_in_timeseries}"
        )

    print(
        "Validation passed: All node IDs in the graph and timeseries DataFrame are aligned."
    )


# Example usage
if __name__ == "__main__":
    graph_path = "dataset/clustered_G3Hops.adjlist"
    # timeseries_path = "dataset/ClusterTimeseries.csv"
    timeseries_path = "dataset/pruned_clustered_timeseries_20250417.csv"

    # validate_graph_and_timeseries(graph_path, timeseries_path)

    prune_graph_and_timeseries(
        graph_path=graph_path,
        timeseries_path=timeseries_path,
        min_degree=5,
        output_graph_path="dataset/pruned_clustered_3hop_graph.adjlist",
        output_timeseries_path="dataset/pruned_clustered_3hop_timeseries.csv",
    )
