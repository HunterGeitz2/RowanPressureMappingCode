#!/usr/bin/env python3
"""
Script to generate 6×6 heatmap images for each row in `extracted_6x6_grids.csv`.
Each row (36 values) is reshaped into a 6×6 matrix and saved as a PNG heatmap.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

# Configuration
INPUT_FILE = "extracted_6x6_grids.csv"
OUTPUT_FOLDER = "6x6_heatmaps"
COLORMAP = "viridis"
VMIN = 0
VMAX = 30


def ensure_output_folder(path):
    os.makedirs(path, exist_ok=True)


def load_all_matrices(csv_path):
    try:
        data = np.loadtxt(csv_path, delimiter=',')
    except Exception as e:
        raise RuntimeError(f"Error loading '{csv_path}': {e}")
    if data.ndim != 2 or data.shape[1] != 36:
        raise ValueError(f"Expected CSV shape (n, 36), got {data.shape}")
    return data


def generate_heatmap(matrix, save_path):
    fig, ax = plt.subplots(figsize=(2, 2), dpi=32)
    ax.imshow(matrix, cmap=COLORMAP, interpolation='nearest', vmin=VMIN, vmax=VMAX)
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

    for idx, row in enumerate(data, start=1):
        matrix = row.reshape((6, 6))
        filename = f"entry_{idx:03d}.png"
        out_path = os.path.join(OUTPUT_FOLDER, filename)
        generate_heatmap(matrix, out_path)
        print(f"Saved heatmap for entry {idx} to '{out_path}'")


if __name__ == "__main__":
    main()
