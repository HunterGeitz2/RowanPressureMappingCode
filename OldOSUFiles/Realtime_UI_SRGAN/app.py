from flask import Flask, jsonify, send_file
from flask_cors import CORS
import numpy as np

import matplotlib
matplotlib.use('Agg')  # ✅ Use non-GUI backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from model_inference import upscale_image
from utils import read_latest_sensor_data
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

STATIC_FOLDER = "static"
os.makedirs(STATIC_FOLDER, exist_ok=True)

@app.route("/predict", methods=["GET"])
def predict():
    # Step 1: Get 6x6 matrix from hardware
    matrix = read_latest_sensor_data("6x6_data.csv")

    # Step 2: Debug info
    print("Real-time 6x6 Matrix:")
    print(matrix)
    if np.all(matrix == 0):
        print("⚠️  Warning: All values are zero! Please check sensor connections or CSV data.")

    # Step 3: Create 6x6 heatmap image using safe Matplotlib backend
    input_path = os.path.join(STATIC_FOLDER, "input_6x6.png")
    #norm_matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix) + 1e-8)

    fig, ax = plt.subplots(figsize=(2, 2), dpi=32)
    #ax.imshow(norm_matrix, cmap="viridis", interpolation="nearest")
    ax.imshow(matrix, cmap="viridis", interpolation="nearest", vmin=0,vmax=30)
    ax.axis("off")

    canvas = FigureCanvas(fig)
    canvas.print_png(input_path)
    plt.close(fig)

    # Step 4: Run SRGAN on 6x6 image
    output_img = upscale_image(input_path)
    output_path = os.path.join(STATIC_FOLDER, "output_srgan_32x32.png")
    output_img.resize((300, 300), Image.BICUBIC).save(output_path)

    # Step 5: Extract statistics
    max_force = float(np.max(matrix))
    total_force = float(np.sum(matrix))

    # Step 6: Return image URLs and stats
    return jsonify({
        "input_image": "http://127.0.0.1:5000/static/input_6x6.png",
        "output_image": "http://127.0.0.1:5000/static/output_srgan_32x32.png",
        "max_force": max_force,
        "total_force": total_force
    })

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_file(os.path.join(STATIC_FOLDER, filename), mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True)
