#!/usr/bin/env python3
"""
Script to generate:
  • 32×32 heatmap images for each row in `32x32SyntheticData.csv`,
    using only the last 1024 columns of each row.
  • 6×6 heatmap images (values ×10) by block-averaging each 32×32 grid
    into 6×6, and saving with figsize=(2,2), dpi=32.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Paths & config ---
SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE        = os.path.join(SCRIPT_DIR, "32x32SyntheticData.csv")
OUTPUT_FOLDER     = os.path.join(SCRIPT_DIR, "TestRealActual32x32_heatmaps")
SMALL_OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "TestRealActual6x6_heatmaps")

COLORMAP = "viridis"
VMIN, VMAX = None, None

# small‐heatmap rendering settings (your example)
SMALL_FIGSIZE = (2, 2)
SMALL_DPI     = 32

# --- Helpers ---
def ensure_folder(path):
    os.makedirs(path, exist_ok=True)

def load_all_matrices(csv_path):
    mats = []
    with open(csv_path, 'r') as f:
        next(f)  # skip header
        for ln, line in enumerate(f, start=2):
            parts = line.rstrip('\n').split(',')
            if len(parts) < 1024:
                print(f"Warning: row {ln} has {len(parts)} cols, skipping")
                continue
            try:
                vals = list(map(float, parts[-1024:]))
            except ValueError:
                print(f"Warning: non-numeric on row {ln}, skipping")
                continue
            mats.append(vals)
    if not mats:
        raise RuntimeError(f"No valid rows in '{csv_path}'")
    return np.array(mats)

def generate_heatmap(matrix, save_path, pixels=32):
    vmin = VMIN if VMIN is not None else matrix.min()
    vmax = VMAX if VMAX is not None else matrix.max()
    fig, ax = plt.subplots(figsize=(1,1), dpi=pixels)
    ax.imshow(matrix, cmap=COLORMAP,
              interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.axis('off')
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def downsample_to_6x6(matrix):
    """Block-average 32×32 → 6×6 so we can’t hit a reshape error."""
    rows, cols = matrix.shape
    # edges that split 0→32 into 6 roughly equal blocks
    row_edges = np.linspace(0, rows, 7, dtype=int)
    col_edges = np.linspace(0, cols, 7, dtype=int)
    small = np.zeros((6,6))
    for i in range(6):
        for j in range(6):
            r0, r1 = row_edges[i],   row_edges[i+1]
            c0, c1 = col_edges[j],   col_edges[j+1]
            block = matrix[r0:r1, c0:c1]
            small[i,j] = block.mean()
    return small

def generate_small_heatmap(mat6, save_path):
    """Use figsize=(2,2),dpi=32 just like your example."""
    vmin = VMIN if VMIN is not None else mat6.min()
    vmax = VMAX if VMAX is not None else mat6.max()
    fig, ax = plt.subplots(figsize=SMALL_FIGSIZE, dpi=SMALL_DPI)
    ax.imshow(mat6, cmap=COLORMAP,
              interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.axis('off')
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def main():
    ensure_folder(OUTPUT_FOLDER)
    ensure_folder(SMALL_OUTPUT_FOLDER)

    if not os.path.exists(INPUT_FILE):
        print(f"Error: '{INPUT_FILE}' not found."); sys.exit(1)

    data = load_all_matrices(INPUT_FILE)
    for idx, row in enumerate(data, start=1):
        mat32 = row.reshape((32,32))

        # 32×32 output
        out32 = os.path.join(OUTPUT_FOLDER, f"entry_{idx:04d}.png")
        generate_heatmap(mat32, out32)
        print(f"Saved 32×32 heatmap #{idx} → '{out32}'")

        # 6×6 output: downsample + scale ×10
        small_mat = downsample_to_6x6(mat32) * 10
        out6 = os.path.join(SMALL_OUTPUT_FOLDER, f"entry_{idx:04d}.png")
        generate_small_heatmap(small_mat, out6)
        print(f"Saved 6×6 (×10 scaled) heatmap #{idx} → '{out6}'")

if __name__ == "__main__":
    main()
