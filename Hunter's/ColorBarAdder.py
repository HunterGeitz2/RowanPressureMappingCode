#!/usr/bin/env python3
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ─── CONFIGURE THESE TWO VARIABLES ──────────────────────────────
INPUT_FOLDER  = "TestRealActual32x32_heatmaps"               # set your source folder name here
OUTPUT_FOLDER = "TestRealActual32x32_heatmaps_with_smallcolorbar"  # set your destination folder name here
# ─────────────────────────────────────────────────────────────────

# Resolve paths relative to this script’s location
BASE_DIR    = os.path.abspath(os.path.dirname(__file__))
INPUT_PATH  = os.path.join(BASE_DIR, INPUT_FOLDER)
OUTPUT_PATH = os.path.join(BASE_DIR, OUTPUT_FOLDER)

def add_colorbar_to_folder(input_path, output_path, vmin=0, vmax=150, cmap='viridis'):
    """
    For each PNG in input_path, load it, append a colorbar matching vmin–vmax
    and cmap (with zero pad), and save into output_path under the same filename.
    Uses an AxesDivider to place the bar flush against the image.
    """
    if not os.path.isdir(input_path):
        print(f"Error: input folder '{input_path}' not found.")
        return

    os.makedirs(output_path, exist_ok=True)
    png_files = sorted(glob.glob(os.path.join(input_path, '*.png')))
    if not png_files:
        print(f"No PNG files found in '{input_path}'.")
        return

    for filepath in png_files:
        # Load the image
        img = mpimg.imread(filepath)

        # Create a figure with a single image axis
        fig, ax_img = plt.subplots(figsize=(8, 4))
        ax_img.imshow(img, origin='upper')
        ax_img.axis('off')

        # Use AxesDivider to append a colorbar axis with zero pad
        divider = make_axes_locatable(ax_img)
        cax = divider.append_axes("right", size="5%", pad=0.0)

        # Build a dummy mappable to drive the colorbar
        norm = Normalize(vmin=vmin, vmax=vmax)
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])

        # Draw the colorbar into that axis
        cbar = fig.colorbar(mappable, cax=cax)
        cax.yaxis.tick_right()

        # Save into the output folder with no padding
        filename = os.path.basename(filepath)
        out_path = os.path.join(output_path, filename)
        fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        print(f"Saved: {out_path}")

if __name__ == '__main__':
    add_colorbar_to_folder(INPUT_PATH, OUTPUT_PATH)
