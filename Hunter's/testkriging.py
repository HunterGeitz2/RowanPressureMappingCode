import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging

# --- DETERMINE SCRIPT DIRECTORY -------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- CONFIGURATION --------------------------------------------------------
# Input CSV (32×32 data) alongside this script
FILE_PATH = os.path.join(SCRIPT_DIR, '32x32SyntheticData.csv')

# Full sensor grid dimensions
SENSOR_ROWS, SENSOR_COLS = 32, 32

# Coarse grid dimensions (6×6)
COARSE_ROWS, COARSE_COLS = 6, 6

# Downsampling method: 'average' to average each block, 'nearest' to pick a representative point
DOWNSAMPLE_METHOD = 'average'  # or 'nearest'

# Kriging variogram parameters
VARIOGRAM_MODEL = 'gaussian'
VARIO_PARAMS = {
    'range': 2.0,
    'sill': 1.0,
    'nugget': 0.1
}

# Output directory for kriged heatmaps
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'Test6x6_krigedheatmaps')
NEW_SIZE = 100  # resolution of the kriged grid
# ----------------------------------------------------------------------------

# Load only the 32×32 sensor columns (expected to be named '0' through '1023')
try:
    df = pd.read_csv(FILE_PATH)
    sensor_cols = [str(i) for i in range(SENSOR_ROWS * SENSOR_COLS)]
    data = df[sensor_cols].astype(float)
    print(f"Loaded {data.shape[0]} frames, each with {data.shape[1]} sensor values.")
except Exception as e:
    raise RuntimeError(f"Could not read '{FILE_PATH}' or extract sensor columns: {e}")


def downsample_to_coarse(full_matrix, coarse_rows=COARSE_ROWS, coarse_cols=COARSE_COLS, method=DOWNSAMPLE_METHOD):
    """
    Downsample a full_matrix (32×32) to a coarse_rows×coarse_cols grid by averaging or nearest sampling.
    """
    rows, cols = full_matrix.shape
    # Compute equally spaced bin edges
    row_edges = np.linspace(0, rows, coarse_rows + 1)
    col_edges = np.linspace(0, cols, coarse_cols + 1)
    coarse = np.zeros((coarse_rows, coarse_cols))

    for i in range(coarse_rows):
        for j in range(coarse_cols):
            r0 = int(np.floor(row_edges[i]))
            r1 = int(np.floor(row_edges[i + 1]))
            c0 = int(np.floor(col_edges[j]))
            c1 = int(np.floor(col_edges[j + 1]))

            if method == 'average':
                block = full_matrix[r0:r1, c0:c1]
                coarse[i, j] = block.mean() if block.size > 0 else np.nan
            elif method == 'nearest':
                # pick the center of the bin
                ri = int(round((row_edges[i] + row_edges[i + 1]) / 2))
                ci = int(round((col_edges[j] + col_edges[j + 1]) / 2))
                ri = min(rows - 1, ri)
                ci = min(cols - 1, ci)
                coarse[i, j] = full_matrix[ri, ci]
            else:
                raise ValueError("Unknown downsampling method: choose 'average' or 'nearest'.")
    return coarse


def apply_kriging(matrix6x6, new_size=NEW_SIZE):
    """Runs ordinary kriging on a 6×6 numpy array, returns a smooth grid of size new_size."""
    rows, cols = matrix6x6.shape
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    values = matrix6x6.flatten()
    X_flat, Y_flat = X.flatten(), Y.flatten()

    xi = np.linspace(0, cols - 1, new_size)
    yi = np.linspace(0, rows - 1, new_size)

    OK = OrdinaryKriging(
        X_flat, Y_flat, values,
        variogram_model=VARIOGRAM_MODEL,
        variogram_parameters=VARIO_PARAMS,
        verbose=False,
        enable_plotting=False
    )
    z, ss = OK.execute('grid', xi, yi)
    return z


def plot_kriging(frame_index, new_size=NEW_SIZE):
    """Show the downsampled 6×6 grid and its kriged interpolation side-by-side for one frame."""
    if not (0 <= frame_index < data.shape[0]):
        raise IndexError(f"frame_index must be between 0 and {data.shape[0]-1}")

    full = data.iloc[frame_index].values.reshape((SENSOR_ROWS, SENSOR_COLS))
    coarse = downsample_to_coarse(full)
    z = apply_kriging(coarse, new_size)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(coarse, origin='lower', cmap='viridis')
    plt.title(f'{COARSE_ROWS}×{COARSE_COLS} Coarse ({DOWNSAMPLE_METHOD})')
    plt.colorbar(shrink=0.7)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(z, origin='lower', cmap='viridis')
    plt.title(f'Kriged {new_size}×{new_size} Grid')
    plt.colorbar(shrink=0.7)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def save_all_kriged(output_dir=OUTPUT_DIR, new_size=NEW_SIZE):
    """Generate and save kriged PNGs for every frame, using the chosen downsampling method."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving kriged maps to '{output_dir}' with '{DOWNSAMPLE_METHOD}' downsampling...")

    for idx in range(data.shape[0]):
        full = data.iloc[idx].values.reshape((SENSOR_ROWS, SENSOR_COLS))
        coarse = downsample_to_coarse(full)
        z = apply_kriging(coarse, new_size)

        plt.figure(figsize=(4, 4))
        plt.imshow(z, origin='lower', cmap='viridis')
        plt.axis('off')
        filename = os.path.join(output_dir, f'kriged_{idx}.png')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved {filename}")

if __name__ == '__main__':
    try:
        save_all_kriged()
        print("All kriged maps have been saved successfully.")
    except Exception as e:
        print(f"Error during kriging: {e}")
