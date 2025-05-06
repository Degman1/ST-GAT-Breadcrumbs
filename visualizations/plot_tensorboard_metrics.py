import os
import glob
import matplotlib.pyplot as plt
from typing import List
from tensorboard.backend.event_processing import event_accumulator
import itertools

# ----------- Settings -------------
EVENTS_DIR = "../runs/arch_tuning"  # Root directory with .tfevents.* files
OUTPUT_DIR = "./arch_tuning_plots"  # Where to save plots
SMOOTH_WEIGHT = 0.7  # Weight for exponential smoothing (0–1)

# Only include event files that contain any of these substrings in their filenames
INCLUDE_FILENAMES = [
    "Default Architecture",
    "16-64 LSTM Hidden Sizes",
    "64-256 LSTM Hidden Sizes",
]

METRICS = ["Loss/val"]
# ----------------------------------


def smooth(scalars: List[float], weight: float) -> List[float]:
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


def file_matches(filename, include_keys):
    return any(key in filename for key in include_keys)


def main():
    event_files = glob.glob(
        os.path.join(EVENTS_DIR, "**", "events.out.tfevents.*"), recursive=True
    )
    filtered_event_files = [
        f for f in event_files if file_matches(f, INCLUDE_FILENAMES)
    ]

    if not filtered_event_files:
        print("No matching event files found.")
        return

    color_cycle = itertools.cycle(plt.cm.tab10.colors)

    for metric in METRICS:
        plt.figure(figsize=(10, 6))
        for event_file in filtered_event_files:
            run_name = os.path.basename(os.path.dirname(event_file))
            print(f"Processing: {run_name}")

            ea = event_accumulator.EventAccumulator(
                event_file, size_guidance={"scalars": 0}
            )
            ea.Reload()

            steps, raw_vals = extract_metrics(ea, metric)
            if not raw_vals:
                print(f"Skipping missing metric: {metric}")
                continue

            smoothed_vals = smooth(raw_vals, SMOOTH_WEIGHT)
            color = next(color_cycle)

            plt.plot(
                steps,
                raw_vals,
                label=f"{run_name}",
                alpha=0.4,
                color=color,
            )
            plt.plot(
                steps,
                smoothed_vals,
                label=f"{run_name} (smoothed)",
                color=color,
                linewidth=2,
            )

        plt.title(f"Impact of LSTM Hidden Sizes on Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filename = os.path.join(OUTPUT_DIR, f"all_runs_{metric.replace('/', '_')}.png")
        plt.tight_layout()
        plt.savefig(filename, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {filename}")


if __name__ == "__main__":
    main()
