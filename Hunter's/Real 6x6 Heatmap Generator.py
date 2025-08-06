#!/usr/bin/env python3
"""
Script to generate 6×6 heatmap images for each row in `p.trialX.csv` files.
For trials 1 through 10, each row (after stripping the date column) is reshaped
row-first into a 6×6 matrix (so values 0–5 fill the first row, 6–11 the second, etc.),
and saved as a PNG heatmap.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

# Configuration
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
TRIAL_RANGE  = range(1, 11)  # trials 1 through 10
COLORMAP     = "viridis"


def ensure_output_folder(path):
    os.makedirs(path, exist_ok=True)


def load_all_matrices(csv_path):
    try:
        # Read CSV, skip header row, treat all remaining rows as data
        df = pd.read_csv(csv_path, skiprows=1, header=None)
    except Exception as e:
        raise RuntimeError(f"Error reading '{csv_path}': {e}")

    # Expect at least one date column + 36 data columns
    if df.shape[1] < 37:
        raise ValueError(
            f"Expected at least 37 columns (date + 36 data), got {df.shape[1]}"
        )

    # Drop the first (date) column and select the next 36 columns
    data_df = df.iloc[:, 1:37]

    # Ensure all remaining values are numeric
    try:
        numeric = data_df.apply(pd.to_numeric, errors='raise').values
    except Exception as e:
        raise RuntimeError(f"Error converting data to numeric: {e}")

    return numeric


def generate_heatmap(matrix, save_path):
    fig, ax = plt.subplots(figsize=(2, 2), dpi=32)
    ax.imshow(
        matrix,
        cmap=COLORMAP,
        interpolation='nearest'
    )
    ax.axis('off')
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def process_trial(trial_num):
    input_file = os.path.join(SCRIPT_DIR, f"p.trial{trial_num}.csv")
    output_folder = os.path.join(SCRIPT_DIR, f"p.trial{trial_num}_heatmaps")
    ensure_output_folder(output_folder)

    if not os.path.exists(input_file):
        print(f"Warning: '{input_file}' not found. Skipping trial {trial_num}.")
        return

    try:
        data = load_all_matrices(input_file)
    except Exception as e:
        print(f"Error loading trial {trial_num}: {e}")
        return

    for idx, row in enumerate(data, start=1):
        # Reshape row-first: 0–5 → row 0, 6–11 → row 1, etc.
        matrix = row.reshape((6, 6), order='C')
        filename = f"entry_{idx:03d}.png"
        out_path = os.path.join(output_folder, filename)
        generate_heatmap(matrix, out_path)
        print(f"Saved heatmap for trial {trial_num}, entry {idx} to '{out_path}'")


def main():
    for trial in TRIAL_RANGE:
        process_trial(trial)


if __name__ == "__main__":
    main()
