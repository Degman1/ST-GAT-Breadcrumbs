import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
import torch


def get_significant_pois(
    attention_matrix, customNodeIds=None, top_k=100, plot=True, transform=np.log1p
):
    significance_scores = attention_matrix.sum(axis=0)
    significance_scores = transform(significance_scores)
    sorted_indices = np.argsort(significance_scores)[::-1]
    sorted_scores = significance_scores[sorted_indices]
    significant_pois = sorted_indices[:top_k]

    # Plot the scores
    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(
            sorted_scores,
        )
        plt.grid(True)
        # plt.axvline(x=top_k, color="r", linestyle="--", label=f"Top {top_k} POIs")
        plt.xlabel("Index of Ranked POI")
        plt.ylabel("Significance Score")
        plt.title("Ranking POIs with Attention-Based Significance Scores")
        location = "./output/poi_significance.png"
        plt.savefig(location, bbox_inches="tight")
        print(f"POI significance chart saved at {location}")

    return (
        (
            np.array(customNodeIds)[significant_pois]
            if customNodeIds is not None
            else significant_pois
        ),
        sorted_scores[:top_k],
        sorted_indices,
    )
