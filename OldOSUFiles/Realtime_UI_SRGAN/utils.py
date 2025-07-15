import os
import numpy as np

def read_latest_sensor_data(path="6x6_data.csv"):
    try:
        full_path = os.path.abspath(path)  # Ensures full path is resolved
        data = np.loadtxt(full_path, delimiter=",")
        if data.shape != (6, 6):
            raise ValueError("Data is not in 6x6 shape")
        return data
    except Exception as e:
        print(f"[ERROR] Couldn't read sensor data: {e}")
        return np.zeros((6, 6))
