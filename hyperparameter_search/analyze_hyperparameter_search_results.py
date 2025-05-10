import json
from collections import defaultdict
from random import sample


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def compute_weighted_mae(results):
    param_mae_sums = defaultdict(
        lambda: {
            "weighted_sum": 0,
            "weighted_average_mae": 0,
            "mae@50%": 0,
            "mae@60%": 0,
            "mae@70%": 0,
        }
    )

    for entry in results:
        params_key = tuple(
            sorted(entry["params"].items())
        )  # Unique key for each param set
        train_percent = entry["train_percent"]
        mae = entry["MAE/val"]

        if train_percent == 0.5:
            param_mae_sums[params_key]["weighted_average_mae"] += mae * 0.2
            param_mae_sums[params_key]["mae@50%"] = mae
            param_mae_sums[params_key]["epochs@50%"] = entry["epochs"]
            param_mae_sums[params_key]["weighted_sum"] += 0.2
        elif train_percent == 0.6:
            param_mae_sums[params_key]["weighted_average_mae"] += mae * 0.3
            param_mae_sums[params_key]["mae@60%"] = mae
            param_mae_sums[params_key]["epochs@60%"] = entry["epochs"]
            param_mae_sums[params_key]["weighted_sum"] += 0.3
        elif train_percent == 0.7:
            param_mae_sums[params_key]["weighted_average_mae"] += mae * 0.5
            param_mae_sums[params_key]["mae@70%"] = mae
            param_mae_sums[params_key]["epochs@70%"] = entry["epochs"]
            param_mae_sums[params_key]["weighted_sum"] += 0.5

    return param_mae_sums


def find_best_hyperparams(file_path, top_5=0, filter_param=None, filter_value=None):
    data = load_json(file_path)
    results = data["Results"]

    # Apply filtering if specified
    if filter_param is not None and filter_value is not None:
        results = [
            entry
            for entry in results
            if entry["params"].get(filter_param) == filter_value
        ]

    weighted_mae = compute_weighted_mae(results)
    sorted_params = sorted(
        weighted_mae.items(), key=lambda x: x[1]["mae@70%"]
    )  # Sort by lowest MAE first

    if top_5 == 0:
        print("Top 5 Hyperparameters (Lowest to Highest Weighted MAE):")
        for i, (params, mae) in enumerate(sorted_params[:5], 1):
            print(f"{i}. Params: {dict(params)}, Weighted MAE: {mae}")
    elif top_5 == 1:
        # Get best-performing config
        best_config = sorted_params[0]

        # Filter out the best config and select 4 more diverse ones
        remaining_configs = sorted_params[1:]

        # Simple diversity: sample 4 spaced-out entries from the rest
        spread_indices = [
            int(len(remaining_configs) * i / 5) for i in range(1, 5)
        ]  # Spaced across the list
        diverse_configs = [remaining_configs[i] for i in spread_indices]

        print("Top Hyperparameter (Best Weighted MAE):")
        print(f"Params: {dict(best_config[0])}, Weighted MAE: {best_config[1]}")

        print("\nFour Additional Diverse Hyperparameter Configs:")
        for i, (params, mae) in enumerate(diverse_configs, 1):
            print(f"{i}. Params: {dict(params)}, Weighted MAE: {mae}")
    else:
        best_params = sorted_params[0]  # Lowest weighted MAE
        print("Best Hyperparameters (Lowest Weighted MAE):")
        print(dict(best_params[0]))
        print(f"Weighted MAE: {best_params[1]:.6f}")


if __name__ == "__main__":
    file_path = "results.json"  # Change this to your actual file
    find_best_hyperparams(file_path, top_5=0)
