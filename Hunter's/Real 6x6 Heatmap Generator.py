#!/usr/bin/env python3
"""
Script to generate 6×6 heatmap images for each row in `extracted_6x6_grids.csv`.
Each row (after stripping the date column) is multiplied by 100, reshaped
column-first into a 6×6 matrix, and saved as a PNG heatmap.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "p.trial5.csv")
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, ".Real6x6_heatmaps")
COLORMAP = "viridis"
VMIN = 0
VMAX = 30


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
        interpolation='nearest',
        vmin=VMIN,
        vmax=VMAX
    )
    ax.axis('off')
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def main():
    ensure_output_folder(OUTPUT_FOLDER)

    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        sys.exit(1)

    try:
        data = load_all_matrices(INPUT_FILE)
    except Exception as e:
        print(e)
        sys.exit(1)

    # Multiply all values by 100
    data *= 10

    for idx, row in enumerate(data, start=1):
        # Reshape column-first so that heatmap[0, :] == [row[0], row[6], ...]
        matrix = row.reshape((6, 6), order='F')
        filename = f"entry_{idx:03d}.png"
        out_path = os.path.join(OUTPUT_FOLDER, filename)
        generate_heatmap(matrix, out_path)
        print(f"Saved heatmap for entry {idx} to '{out_path}'")


if __name__ == "__main__":
    main()
