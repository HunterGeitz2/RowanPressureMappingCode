import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
import os

# Path to your CSV file
file_path = 'extracted_6x6_grids.csv'
# Read the CSV into a DataFrame
try:
    data = pd.read_csv(file_path)
    print(f"Loaded CSV with {data.shape[0]} rows and {data.shape[1]} columns.")
except Exception as e:
    raise RuntimeError(f"Failed to read '{file_path}': {e}")

# Dimensions of the coarse grid
rows, cols = 6, 6

# Default variogram parameters (can be tuned)
VARIOGRAM_PARAMS = {
    'variogram_model': 'gaussian',
    'variogram_parameters': {
        'range': 2.0,
        'sill': 1.0,
        'nugget': 0.1
    }
}

def apply_kriging(matrix, new_size=100, variogram_model='gaussian', variogram_parameters=None):
    """
    Apply ordinary kriging to a coarse matrix to produce a smooth high-resolution grid.
    """
    if variogram_parameters is None:
        variogram_parameters = VARIOGRAM_PARAMS['variogram_parameters']

    # Original grid coordinates
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)

    # Flatten for input to PyKrige
    values = matrix.flatten()
    X_flat = X.flatten()
    Y_flat = Y.flatten()

    # New grid
    xi = np.linspace(0, cols - 1, new_size)
    yi = np.linspace(0, rows - 1, new_size)

    # Run ordinary kriging
    OK = OrdinaryKriging(
        X_flat, Y_flat, values,
        variogram_model=variogram_model,
        variogram_parameters=variogram_parameters,
        verbose=False,
        enable_plotting=False
    )
    z, ss = OK.execute('grid', xi, yi)
    return z


def plot_kriging(time_index, new_size=100):
    """
    Plot the original coarse grid and its kriged interpolation for a given row of the CSV.
    """
    if time_index < 0 or time_index >= data.shape[0]:
        raise IndexError(f"time_index must be between 0 and {data.shape[0]-1}")

    row = data.iloc[time_index].values
    if len(row) != rows * cols:
        raise ValueError(f"Expected {rows*cols} values but got {len(row)}")

    matrix = row.reshape((rows, cols))
    z = apply_kriging(matrix, new_size, **VARIOGRAM_PARAMS)

    plt.figure(figsize=(10, 4))
    # Original
    plt.subplot(1, 2, 1)
    plt.imshow(matrix, origin='lower', cmap='viridis')
    plt.title(f'Original {rows}x{cols} Grid (Row={time_index})')
    plt.colorbar(shrink=0.7)
    plt.axis('off')

    # Kriged
    plt.subplot(1, 2, 2)
    plt.imshow(z, origin='lower', cmap='viridis')
    plt.title(f'Kriged {new_size}x{new_size} Grid')
    plt.colorbar(shrink=0.7)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def save_all_kriged(output_dir='6x6_krigedheatmaps', new_size=100):
    """
    Generate and save kriged maps for all rows in the CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving kriged maps to '{output_dir}'...")

    for idx in range(data.shape[0]):
        row = data.iloc[idx].values
        if len(row) != rows * cols:
            print(f"Skipping row {idx}: expected {rows*cols} values, got {len(row)}")
            continue

        matrix = row.reshape((rows, cols))
        z = apply_kriging(matrix, new_size, **VARIOGRAM_PARAMS)

        plt.figure(figsize=(4, 4))
        plt.imshow(z, origin='lower', cmap='viridis')
        plt.axis('off')
        filename = os.path.join(output_dir, f'kriged_{idx}.png')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved {filename}")

if __name__ == '__main__':
    # Automatically run saving when executed as a script
    try:
        save_all_kriged()
        print("All kriged maps have been saved successfully.")
    except Exception as e:
        print(f"Error during kriging: {e}")

# Example usage from interactive session:
# plot_kriging(time_index=0, new_size=100)
