import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pykrige.ok import OrdinaryKriging
import os

# Load the CSV file
file_path = "soren.csv"  # Ensure this path is correct soren.csv
data = pd.read_csv(file_path)
data.drop(['Frame', 'Timestamp', 'Average Pressure (mmHg)', 'Minimum Pressure (mmHg)', 
           'Maximum Pressure (mmHg)', 'Standard Pressure Deviation (mmHg)', 
           'Median Pressure (mmHg)', 'Contact Area (m²)', 'Total Area (m²)', 
           'Estimated Force (N)', 'Range Min (mmHg)', 'Range Max (mmHg)'], axis=1, inplace=True)

# Placeholder for coords (black squares on a chessboard)
# coords = []
# for row in range(32):
#     for col in range(32):
#         if (row + col) % 2 == 1:
#             coords.append([row, col])

coords = [
    [0, 0], [0, 2], [0, 4], [0, 6], [0, 8], [0, 10], [0, 12], [0, 14], [0, 16], [0, 18], [0, 20], [0, 22], [0, 24], [0, 26], [0, 28], [0, 30],
    [2, 0], [2, 2], [2, 4], [2, 6], [2, 8], [2, 10], [2, 12], [2, 14], [2, 16], [2, 18], [2, 20], [2, 22], [2, 24], [2, 26], [2, 28], [2, 30],
    [4, 0], [4, 2], [4, 4], [4, 6], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 18], [4, 20], [4, 22], [4, 24], [4, 26], [4, 28], [4, 30],
    [6, 0], [6, 2], [6, 4], [6, 6], [6, 8], [6, 10], [6, 12], [6, 14], [6, 16], [6, 18], [6, 20], [6, 22], [6, 24], [6, 26], [6, 28], [6, 30],
    [8, 0], [8, 2], [8, 4], [8, 6], [8, 8], [8, 10], [8, 12], [8, 14], [8, 16], [8, 18], [8, 20], [8, 22], [8, 24], [8, 26], [8, 28], [8, 30],
    [10, 0], [10, 2], [10, 4], [10, 6], [10, 8], [10, 10], [10, 12], [10, 14], [10, 16], [10, 18], [10, 20], [10, 22], [10, 24], [10, 26], [10, 28], [10, 30],
    [12, 0], [12, 2], [12, 4], [12, 6], [12, 8], [12, 10], [12, 12], [12, 14], [12, 16], [12, 18], [12, 20], [12, 22], [12, 24], [12, 26], [12, 28], [12, 30],
    [14, 0], [14, 2], [14, 4], [14, 6], [14, 8], [14, 10], [14, 12], [14, 14], [14, 16], [14, 18], [14, 20], [14, 22], [14, 24], [14, 26], [14, 28], [14, 30],
    [16, 0], [16, 2], [16, 4], [16, 6], [16, 8], [16, 10], [16, 12], [16, 14], [16, 16], [16, 18], [16, 20], [16, 22], [16, 24], [16, 26], [16, 28], [16, 30],
    [18, 0], [18, 2], [18, 4], [18, 6], [18, 8], [18, 10], [18, 12], [18, 14], [18, 16], [18, 18], [18, 20], [18, 22], [18, 24], [18, 26], [18, 28], [18, 30],
    [20, 0], [20, 2], [20, 4], [20, 6], [20, 8], [20, 10], [20, 12], [20, 14], [20, 16], [20, 18], [20, 20], [20, 22], [20, 24], [20, 26], [20, 28], [20, 30],
    [22, 0], [22, 2], [22, 4], [22, 6], [22, 8], [22, 10], [22, 12], [22, 14], [22, 16], [22, 18], [22, 20], [22, 22], [22, 24], [22, 26], [22, 28], [22, 30],
    [24, 0], [24, 2], [24, 4], [24, 6], [24, 8], [24, 10], [24, 12], [24, 14], [24, 16], [24, 18], [24, 20], [24, 22], [24, 24], [24, 26], [24, 28], [24, 30],
    [26, 0], [26, 2], [26, 4], [26, 6], [26, 8], [26, 10], [26, 12], [26, 14], [26, 16], [26, 18], [26, 20], [26, 22], [26, 24], [26, 26], [26, 28], [26, 30],
    [28, 0], [28, 2], [28, 4], [28, 6], [28, 8], [28, 10], [28, 12], [28, 14], [28, 16], [28, 18], [28, 20], [28, 22], [28, 24], [28, 26], [28, 28], [28, 30],
    [30, 0], [30, 2], [30, 4], [30, 6], [30, 8], [30, 10], [30, 12], [30, 14], [30, 16], [30, 18], [30, 20], [30, 22], [30, 24], [30, 26], [30, 28], [30, 30]
]

def apply_kriging_hr(matrix, new_size=100):
    """ Apply Kriging interpolation including zero values for smooth interpolation. """
    
    # Generate grid for the original matrix
    x = np.arange(matrix.shape[1])
    y = np.arange(matrix.shape[0])
    X, Y = np.meshgrid(x, y)

    # Flatten the arrays for kriging input
    values = matrix.flatten()
    X_flat = X.flatten()
    Y_flat = Y.flatten()

    # Create new grid for interpolation
    new_x = np.linspace(0, matrix.shape[1] - 1, new_size)
    new_y = np.linspace(0, matrix.shape[0] - 1, new_size)
    new_X, new_Y = np.meshgrid(new_x, new_y)

    # Apply ordinary kriging with an appropriate variogram model
    kriging = OrdinaryKriging(X_flat, Y_flat, values, 
                              variogram_model='gaussian', 
                              variogram_parameters={'range': 10, 'sill': 2, 'nugget': 0.5},
                              verbose=False, enable_plotting=False)
    
    # Interpolate the values on the new grid
    z_new, _ = kriging.execute('grid', new_x, new_y)

    return z_new

def plot_pressure_map_with_kriging(time, rows=32, cols=32, krig_size=256):
    ''' Convert 1024 values from the specified row into a 32x32 array, apply kriging, and plot the pressure maps. '''
    
    # Select the row corresponding to the specified time (row index)
    if time >= data.shape[0]:
        print(f"Invalid time value. The dataset only has {data.shape[0]} rows.")
        return

    row_data = data.iloc[time].values  # Extract the specified row

    if len(row_data) != 1024:
        print(f"Unexpected data length. Expected 1024 values, but got {len(row_data)}.")
        return

    # Reshape the 1024 values into a 32x32 matrix
    pressure_matrix = row_data.reshape((rows, cols))

    # Clip the matrix values to the expected pressure range
    fsr_range = (0, 150)
    pressure_matrix = np.clip(pressure_matrix, fsr_range[0], fsr_range[1])

    # Create a new matrix to store values at the specified coordinates
    reduced_matrix = np.zeros_like(pressure_matrix)  # Initialize with zeros
    
    # Populate reduced_matrix with values from pressure_matrix at specified coordinates
    for coord in coords:
        reduced_matrix[coord[0], coord[1]] = pressure_matrix[coord[0], coord[1]]

    # Apply kriging to the original pressure matrix, including zero values
    kriged_pressure_matrix = apply_kriging_hr(pressure_matrix, krig_size)

    # Prepare values from the reduced matrix only for the specified coordinates
    x_reduced = [coord[1] for coord in coords]  # Column indices
    y_reduced = [coord[0] for coord in coords]  # Row indices
    values_reduced = reduced_matrix.flatten()[[coord[0] * cols + coord[1] for coord in coords]]

    # Apply Kriging only for the reduced matrix using the specific coordinates
    kriged_reduced_matrix = OrdinaryKriging(
        x_reduced,
        y_reduced,
        values_reduced,
        variogram_model='gaussian',
        variogram_parameters={'range': 10, 'sill': 2, 'nugget': 0.5},
        verbose=False,
        enable_plotting=False
    ).execute('grid', np.linspace(0, 31, krig_size), np.linspace(0, 31, krig_size))[0]

    # Calculate RMSE between kriged matrices
    residuals = kriged_reduced_matrix - kriged_pressure_matrix
    squared_residuals = residuals ** 2
    mean_squared_residuals = np.mean(squared_residuals)
    rmse = np.sqrt(mean_squared_residuals)

    print(f"RMSE between Kriged Reduced and Kriged Original Pressure Maps: {rmse}")

    # Plot the original, reduced, and kriged pressure maps
    plt.figure(figsize=(14, 12))

    # Plot original pressure map
    plt.subplot(2, 2, 1)
    sns.heatmap(pressure_matrix, cmap="viridis", cbar=True, vmin=0, vmax=150)
    plt.title(f"Original Pressure Map at Time {time}")
    plt.axis('off')

    # Plot reduced pressure map
    plt.subplot(2, 2, 2)
    sns.heatmap(reduced_matrix, cmap="viridis", cbar=True, vmin=0, vmax=150)
    plt.title(f"Reduced Pressure Map (Selected Squares) at Time {time}")
    plt.axis('off')

    # Plot kriged original pressure map (256x256)
    plt.subplot(2, 2, 3)
    sns.heatmap(kriged_pressure_matrix, cmap="viridis", cbar=True, vmin=0, vmax=150)
    plt.title(f"Kriged Original Pressure Map (256x256)")
    plt.axis('off')

    # Plot kriged reduced pressure map (256x256)
    plt.subplot(2, 2, 4)
    sns.heatmap(kriged_reduced_matrix, cmap="viridis", cbar=True, vmin=0, vmax=150)
    plt.title(f"Kriged Reduced Pressure Map (256x256)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage - call the function with the appropriate time and data
plot_pressure_map_with_kriging(677)


def plot_and_save_pressure_maps(data, rows=32, cols=32, krig_size=256):
    ''' 
    Convert 1024 values from each row into a 32x32 array, apply kriging, and save the pressure maps 
    into specified folders for high-resolution (original) and low-resolution (reduced) kriged maps.
    '''
    # Create folders if they don't exist
    os.makedirs("kriging_high", exist_ok=True)
    os.makedirs("kriging_low", exist_ok=True)
    
    for time in range(data.shape[0]):
        # Select the row corresponding to the specified time (row index)
        row_data = data.iloc[time].values  # Extract the specified row

        if len(row_data) != 1024:
            print(f"Unexpected data length at time {time}. Expected 1024 values, but got {len(row_data)}.")
            continue

        # Reshape the 1024 values into a 32x32 matrix
        pressure_matrix = row_data.reshape((rows, cols))

        # Clip the matrix values to the expected pressure range
        fsr_range = (0, 150)
        pressure_matrix = np.clip(pressure_matrix, fsr_range[0], fsr_range[1])

        # Create a new matrix to store values at the specified coordinates
        reduced_matrix = np.zeros_like(pressure_matrix)  # Initialize with zeros
        
        # Populate reduced_matrix with values from pressure_matrix at specified coordinates
        for coord in coords:
            reduced_matrix[coord[0], coord[1]] = pressure_matrix[coord[0], coord[1]]

        # Apply kriging to the original pressure matrix, including zero values
        kriged_pressure_matrix = apply_kriging_hr(pressure_matrix, krig_size)

        # Prepare values from the reduced matrix only for the specified coordinates
        x_reduced = [coord[1] for coord in coords]  # Column indices
        y_reduced = [coord[0] for coord in coords]  # Row indices
        values_reduced = reduced_matrix.flatten()[[coord[0] * cols + coord[1] for coord in coords]]

        # Apply Kriging only for the reduced matrix using the specific coordinates
        kriged_reduced_matrix = OrdinaryKriging(
            x_reduced,
            y_reduced,
            values_reduced,
            variogram_model='gaussian',
            variogram_parameters={'range': 10, 'sill': 2, 'nugget': 0.5},
            verbose=False,
            enable_plotting=False
        ).execute('grid', np.linspace(0, 31, krig_size), np.linspace(0, 31, krig_size))[0]

        # Save the kriged high-resolution (original) map
        plt.figure(figsize=(7, 6))
        sns.heatmap(kriged_pressure_matrix, cmap="viridis", cbar=True, vmin=0, vmax=150)
        plt.title(f"Kriged Original Pressure Map at Time {time}")
        plt.axis('off')
        plt.savefig(f"kriging_high/hr_{time + 1}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        # Save the kriged low-resolution (reduced) map
        plt.figure(figsize=(7, 6))
        sns.heatmap(kriged_reduced_matrix, cmap="viridis", cbar=True, vmin=0, vmax=150)
        plt.title(f"Kriged Reduced Pressure Map at Time {time}")
        plt.axis('off')
        plt.savefig(f"kriging_low/lr_{time + 1}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f"Saved hr_{time + 1}.png and lr_{time + 1}.png")

# Call the function with the loaded data
plot_and_save_pressure_maps(data)

