import json
from collections import defaultdict

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def compute_weighted_mae(results):
    param_mae_sums = defaultdict(lambda: {"weighted_sum": 0, "weight_sum": 0})
    
    for entry in results:
        params_key = tuple(sorted(entry["params"].items()))  # Unique key for each param set
        train_percent = entry["train_percent"]
        mae = entry["mae"]
        
        param_mae_sums[params_key]["weighted_sum"] += mae * train_percent
        param_mae_sums[params_key]["weight_sum"] += train_percent
    
    # Compute final weighted MAE
    param_weighted_mae = {
        params: values["weighted_sum"] / values["weight_sum"]
        for params, values in param_mae_sums.items()
    }
    
    return param_weighted_mae

def find_best_hyperparams(file_path, top_5=False, filter_param=None, filter_value=None):
    data = load_json(file_path)
    results = data["Results"]
    
    # Apply filtering if specified
    if filter_param is not None and filter_value is not None:
        results = [entry for entry in results if entry["params"].get(filter_param) == filter_value]
    
    weighted_mae = compute_weighted_mae(results)
    sorted_params = sorted(weighted_mae.items(), key=lambda x: x[1])  # Sort by lowest MAE first
    
    if top_5:
        print("Top 5 Hyperparameters (Lowest to Highest Weighted MAE):")
        for i, (params, mae) in enumerate(sorted_params[:5], 1):
            print(f"{i}. Params: {dict(params)}, Weighted MAE: {mae:.6f}")
    else:
        best_params = sorted_params[0]  # Lowest weighted MAE
        print("Best Hyperparameters (Lowest Weighted MAE):")
        print(dict(best_params[0]))
        print(f"Weighted MAE: {best_params[1]:.6f}")

if __name__ == "__main__":
    file_path = "results.json"  # Change this to your actual file
    find_best_hyperparams(file_path, top_5=True, filter_param="N_HIST", filter_value=24)
