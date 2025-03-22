import networkx as nx
import pandas as pd
import json

def prune_graph_and_timeseries(graph_path, timeseries_path, min_degree, output_graph_path, output_timeseries_path):
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

    # Step 3: Identify nodes to retain
    nodes_to_keep = [node for node in G.nodes if G.degree(node) >= min_degree]

    # Step 4: Create a subgraph with the retained nodes
    pruned_graph = G.subgraph(nodes_to_keep).copy()

    # Step 5: Write the pruned graph to a new adjlist file
    nx.write_adjlist(pruned_graph, output_graph_path)

    # Step 6: Read the timeseries data
    timeseries_df = pd.read_csv(timeseries_path, index_col=0)

    # Step 7: Ensure column IDs in the timeseries DataFrame are strings
    timeseries_df.columns = timeseries_df.columns.astype(str)

    # Step 8: Filter columns corresponding to retained nodes
    nodes_to_keep = list(map(str, nodes_to_keep))  # Ensure consistent formatting with timeseries
    filtered_timeseries_df = timeseries_df[nodes_to_keep]

    # Step 9: Save the filtered timeseries to a new CSV file
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

    print("Validation passed: All node IDs in the graph and timeseries DataFrame are aligned.")

# Example usage
if __name__ == "__main__":
    graph_path = "dataset/clustered_G1Hops.adjlist"
    timeseries_path = "dataset/ClusterTimeseries.csv"

    # validate_graph_and_timeseries(graph_path, timeseries_path)
    
    prune_graph_and_timeseries(
        graph_path=graph_path,
        timeseries_path=timeseries_path,
        min_degree=5,
        output_graph_path="dataset/pruned_clustered_1hop_graph.adjlist",
        output_timeseries_path="dataset/pruned_clustered_1hop_timeseries.csv",
    )
