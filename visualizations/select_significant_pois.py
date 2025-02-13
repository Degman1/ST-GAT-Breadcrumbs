import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
import torch

def get_significant_pois(attention_matrix, customNodeIds=None, top_k=30, plot=True):
    significance_scores = attention_matrix.sum(axis=1)
    sorted_indices = np.argsort(significance_scores)[::-1]
    significant_pois = sorted_indices[:top_k]

    # Plot the scores
    if plot:
        sorted_scores = significance_scores[sorted_indices]
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(sorted_scores) + 1), sorted_scores, marker='o', label='Significance Scores')
        plt.axvline(x=top_k, color='r', linestyle='--', label=f'Top {top_k} POIs')
        plt.xlabel("Indices of Ranked POIs")
        plt.ylabel("Significance Score")
        plt.title("Top-K Method for Selecting Significant POIs")
        plt.legend()
        plt.savefig("visualizations/poi_significance.png", bbox_inches="tight")
        print("POI significance chart saved.")

    return np.array(customNodeIds)[significant_pois] if customNodeIds is not None else significant_pois
