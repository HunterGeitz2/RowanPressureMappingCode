#!/usr/bin/env python3
"""
Script to generate 32×32 heatmap images for each row in `f.trial6.csv`,
ignoring the first column of the CSV and skipping any malformed rows.
Each remaining 1024 values is reshaped into a 32×32 matrix and saved as a PNG heatmap,
auto‐scaled per‐image (unless you set VMIN/VMAX manually). Input and output
paths are located relative to the script’s directory.  The output PNGs
will be exactly 32×32 pixels.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

# Determine the directory this script lives in
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration (relative to SCRIPT_DIR)
INPUT_FILE = os.path.join(SCRIPT_DIR, "f.trial5.csv")
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "RealActual32x32_heatmaps")
COLORMAP = "viridis"
# Set to None to auto-scale per image; otherwise use fixed bounds
VMIN = None
VMAX = None


def ensure_output_folder(path):
    os.makedirs(path, exist_ok=True)


def load_all_matrices(csv_path):
    matrices = []
    with open(csv_path, 'r') as f:
        # Skip header (remove this line if no header)
        next(f)
        for lineno, line in enumerate(f, start=2):
            parts = line.rstrip('\n').split(',')
            if len(parts) < 1025:
                print(f"Warning: skipping malformed row {lineno} ({len(parts)} cols)")
                continue
            try:
                # ignore first column, parse cols 1–1024
                row = [float(x) for x in parts[1:1025]]
            except ValueError as e:
                print(f"Warning: non-numeric at row {lineno}, skipping: {e}")
                continue
            matrices.append(row)

    if not matrices:
        raise RuntimeError(f"No valid 32×32 rows found in '{csv_path}'")
    return np.array(matrices)


def generate_heatmap(matrix, save_path):
    # auto-scale unless VMIN/VMAX set
    vmin = VMIN if VMIN is not None else matrix.min()
    vmax = VMAX if VMAX is not None else matrix.max()

    # figsize=(1,1) at dpi=32 → 32×32 px output
    fig, ax = plt.subplots(figsize=(1, 1), dpi=32)
    ax.imshow(matrix, cmap=COLORMAP,
              interpolation='nearest',
              vmin=vmin, vmax=vmax)
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
        matrix = row.reshape((32, 32))
        filename = f"entry_{idx:04d}.png"
        out_path = os.path.join(OUTPUT_FOLDER, filename)
        generate_heatmap(matrix, out_path)
        print(f"Saved 32×32-px heatmap for entry {idx} to '{out_path}'")


if __name__ == "__main__":
    main()
