from flask import Flask, jsonify, send_file
from flask_cors import CORS
import numpy as np

import matplotlib
#matplotlib.use('Agg')  # ✅ Use non-GUI backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from model_inference import upscale_image
from utils import read_latest_sensor_data
from PIL import Image
import os

matrix = read_latest_sensor_data("6x6_data.csv")
#matrix = matrix/9
#fig, ax = plt.subplots(figsize=(2, 2), dpi=32)
plt.imshow(matrix, cmap="jet", vmin=0,vmax=25)
#plt.imshow(matrix, cmap="hsv", vmin=0,vmax=25)
plt.show()
#ax.axis("off")