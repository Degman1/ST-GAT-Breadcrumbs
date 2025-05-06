import os
import glob
from typing import List
import numpy as np
from collections import defaultdict
from tensorboard.backend.event_processing import event_accumulator

# ----------- Settings -------------
EVENTS_DIR = "../runs/arch_tuning"  # Path to your folder with .tfevents.* files
SMOOTH_WEIGHT = (
    0.7  # Weight for exponential smoothing (0 = no smoothing, 1 = max smoothing)
)
TARGET_METRIC = "MAE/val"
METRICS_TO_REPORT = [
    "MAE/train",
    "RMSE/train",
    "WMAPE/train",
    "MAE/val",
    "RMSE/val",
    "WMAPE/val",
]
# ----------------------------------


def smooth(scalars, weight):
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def extract_metrics(ea, tag):
    if tag not in ea.Tags()["scalars"]:
        return [], []
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def load_event_file(path):
    ea = event_accumulator.EventAccumulator(path, size_guidance={"scalars": 0})
    ea.Reload()
    return ea


def find_best_epoch(smoothed_vals, original_steps):
    min_idx = np.argmin(smoothed_vals)
    best_step = original_steps[min_idx]
    return best_step


def main():
    event_files = glob.glob(
        os.path.join(EVENTS_DIR, "**", "events.out.tfevents.*"), recursive=True
    )
    if not event_files:
        print("No event files found.")
        return

    for event_file in event_files:
        print(f"\n--- Processing {event_file} ---")
        ea = load_event_file(event_file)

        # 1. Extract validation MAE
        val_steps, val_mae = extract_metrics(ea, TARGET_METRIC)
        if not val_mae:
            print(f"{TARGET_METRIC} not found in log.")
            continue

        # 2. Smooth validation MAE
        smoothed_mae = smooth(val_mae, SMOOTH_WEIGHT)

        # 3. Find best epoch
        best_epoch = find_best_epoch(smoothed_mae, val_steps)
        print(f"Selected epoch: {best_epoch} (based on smoothed {TARGET_METRIC})")

        # 4. Report all metrics at that epoch
        results = {}
        for metric in METRICS_TO_REPORT:
            steps, values = extract_metrics(ea, metric)
            if not steps:
                results[metric] = "N/A"
                continue

            # Find value at best epoch (or closest prior step)
            valid_indices = [i for i, s in enumerate(steps) if s <= best_epoch]
            if valid_indices:
                idx = valid_indices[-1]
                results[metric] = values[idx]
            else:
                results[metric] = "N/A"

        # 5. Print results
        print("Metric values at selected epoch:")
        for k, v in results.items():
            print(f"{k:15s}: {round(v, 3):.3f}")


if __name__ == "__main__":
    main()
