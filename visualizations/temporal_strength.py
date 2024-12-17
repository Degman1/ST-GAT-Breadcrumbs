import pandas as pd
from datetime import datetime, time
import numpy as np

# Helper function to categorize time into periods
def categorize_time(timestamp):
    t = timestamp.time()
    if time(6, 0) <= t <= time(9, 59):
        return "morning"
    elif time(10, 0) <= t <= time(14, 59):
        return "midday"
    elif time(15, 0) <= t <= time(18, 59):
        return "afternoon"
    else:  # 7 PM to 5:59 AM
        return "night"

# Load the time series data
def process_time_series(csv_path):
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Assume first column is the time index, named 'time'
    df.rename(columns={df.columns[0]: 'time'}, inplace=True)
    
    # Convert time column to datetime format
    df['time'] = pd.to_datetime(df['time'], format="%Y-%m-%dT-%H-%M")
    
    # Filter weekdays only (0 = Monday, 4 = Friday)
    df = df[df['time'].dt.weekday < 5]
    
    # Categorize times into the defined periods
    df['time_period'] = df['time'].apply(categorize_time)
    
    # Drop the 'time' column as it's no longer needed
    df = df.drop(columns=['time'])
    
    # Group by time periods and compute the mean for each node
    averages = df.groupby('time_period').mean()
    
    return averages

if __name__ == "__main__":
    # Path to the input CSV file
    csv_path = "../dataset/ClusterTimeseries.csv"
    
    # Process the time series data
    result = process_time_series(csv_path)
    
    # Display the resulting averages
    print("Average node values for each time period:")
    print(result)
    
    # Optionally save to a new CSV file
    result.to_csv("TimePeriodAverages.csv")
