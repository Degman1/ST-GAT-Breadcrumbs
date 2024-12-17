import pandas as pd
from datetime import datetime, time
import numpy as np
import networkx as nx

# Helper function to categorize time into periods
def categorize_time(timestamp):
    """
    Categorize a given timestamp into one of four time periods:
    - Morning: 6:00 AM - 9:59 AM
    - Midday: 10:00 AM - 2:59 PM
    - Afternoon: 3:00 PM - 6:59 PM
    - Night: 7:00 PM - 5:59 AM
    
    Args:
        timestamp (datetime): A datetime object representing the time.
    
    Returns:
        str: The time period as a string (morning, midday, afternoon, or night).
    """
    t = timestamp.time()
    if time(6, 0) <= t <= time(9, 59):
        return "morning"
    elif time(10, 0) <= t <= time(14, 59):
        return "midday"
    elif time(15, 0) <= t <= time(18, 59):
        return "afternoon"
    else:  # 7 PM to 5:59 AM
        return "night"

# Compute node strengths and time periods from the dataframe
def compute_temporal_node_strength(df_without_time, time_periods, df):
    """
    Process the time series data by categorizing timestamps into time periods and filtering weekdays.
    
    Args:
        df (DataFrame): DataFrame containing the time series data with a 'time' column.
    
    Returns:
        DataFrame of averaged node strengths grouped by time period.
    """
    
    # Group by time periods and compute the mean for each node
    averages = df.groupby('time_period').mean().drop(columns=['time'])
    
    return averages

# Compute temporal edge strength across consecutive time steps
def compute_temporal_edge_strength(node_strengths, graph_structure, time_periods):
    """
    Compute temporal edge strength by calculating the average strength across consecutive time steps
    and grouping by time periods.
    
    Args:
        node_strengths (DataFrame): Node strengths indexed by time steps.
        graph_structure (Graph): A NetworkX graph structure with edges defined.
        time_periods (list): List of time periods corresponding to each time step.
    
    Returns:
        dict: A dictionary with time periods as keys and average edge strengths as values.
    """
    # Initialize a dictionary to store temporal edge strengths grouped by time period
    temporal_edge_strengths = {period: [] for period in set(time_periods)}
    
    # Iterate over consecutive time steps to calculate edge strengths
    for t in range(len(node_strengths) - 1):
        edge_strengths = []
        for u, v in graph_structure.edges():
            strength_t = abs(node_strengths.loc[t, str(u)] - node_strengths.loc[t, str(v)])
            strength_t1 = abs(node_strengths.loc[t+1, str(u)] - node_strengths.loc[t+1, str(v)])
            avg_strength = (strength_t + strength_t1) / 2  # Average across t and t+1
            edge_strengths.append(avg_strength)
        # Group edge strength by time period
        temporal_edge_strengths[time_periods[t]].append(np.mean(edge_strengths))
    
    # Average edge strengths for each time period
    averaged_edge_strengths = {
        period: np.mean(strengths) if strengths else 0
        for period, strengths in temporal_edge_strengths.items()
    }
    
    return averaged_edge_strengths

import folium
import matplotlib.pyplot as plt

# Function to visualize the graph with node and edge strength
def visualize_temporal_graph(graph, node_positions, edge_strength, node_strength, time_period):
    """
    Visualize the graph on a map using Folium, coloring nodes and edges based on strengths.
    
    Args:
        graph (Graph): The NetworkX graph.
        node_positions (dict): A dictionary mapping node IDs to (longitude, latitude) positions.
        edge_strength (dict): Temporal edge strengths grouped by time period.
        node_strength (DataFrame): Node strengths averaged by time period.
        time_period (str): The current time period being visualized.
    """
    # Create a map centered at the average location of all nodes
    avg_long = np.mean([pos[0] for pos in node_positions.values()])
    avg_lat = np.mean([pos[1] for pos in node_positions.values()])
    map_center = folium.Map(location=[avg_lat, avg_long], zoom_start=10)
    
    # Add nodes to the map with colored markers based on their strength
    for node, pos in node_positions.items():
        folium.CircleMarker(
            location=pos,
            radius=10,
            color='blue',
            fill=True,
            fill_color=plt.cm.viridis(node_strength.loc[time_period, str(node)] / node_strength.loc[time_period].max()),
            fill_opacity=0.8
        ).add_to(map_center)
    
    # Add edges to the map with colored lines based on their strength
    for u, v in graph.edges():
        avg_edge_strength = edge_strength[time_period].get((u, v), 0)  # Get strength if edge exists
        folium.PolyLine(
            locations=[node_positions[u], node_positions[v]],
            color=plt.cm.plasma(avg_edge_strength),
            weight=3,
            opacity=0.8
        ).add_to(map_center)
    
    # Save map to HTML file
    map_center.save(f"temporal_graph_{time_period}.html")
    print(f"Map for time period '{time_period}' saved to 'temporal_graph_{time_period}.html'.")

# Main script
if __name__ == "__main__":
    # Paths to input files
    csv_path = "ClusterTimeseries.csv"
    graph_path = "dataset/G.adjlist"
    
    # TODO Load the graph structure and node positions
    graph_structure = nx.read_adjlist(graph_path, nodetype=int)
    node_positions = pd.read_csv("node_locations.csv", index_col=0).to_dict('index')
    
    # Read time series data
    print("\nReading time series data...")
    df = pd.read_csv(csv_path)
    
    print("\nProcessing time series data...")
    # Convert time column to datetime format
    df['time'] = pd.to_datetime(df['time'], format="%Y-%m-%dT-%H-%M")
    
    # Filter weekdays only (0 = Monday, 4 = Friday)
    df = df[df['time'].dt.weekday < 5]
    
    # Categorize times into the defined periods
    df['time_period'] = df['time'].apply(categorize_time)
    
    # Extract time periods and drop the 'time' column
    time_periods = df['time_period'].tolist()
    df_without_time = df.drop(columns=['time', 'time_period'])
    
    # Process time series data
    print("\nComputing temporal node strength")
    temporal_node_strength = compute_temporal_node_strength(time_periods, df_without_time, df)
    
    # Compute temporal edge strength
    print("\nComputing temporal edge strength...")
    temporal_edge_strength = compute_temporal_edge_strength(df_without_time, graph_structure, time_periods)
    
    # Visualize graph for each time period
    for time_period in temporal_edge_strength.keys():
        print(f"\nVisualizing graph for time period: {time_period}")
        visualize_temporal_graph(graph_structure, node_positions, temporal_edge_strength, temporal_node_strength, time_period)