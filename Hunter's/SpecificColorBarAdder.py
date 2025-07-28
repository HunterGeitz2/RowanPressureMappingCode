import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ─── CONFIG ─────────────────────────────────────────────────────
INPUT_FOLDER  = "TestRealActual32x32_heatmaps"
OUTPUT_FOLDER = "TestRealActual32x32_heatmaps_with_specificsmallcolorbar"
CSV_FILENAME  = "32x32SyntheticData.csv"  # CSV in same directory as this script
# ────────────────────────────────────────────────────────────────

# Resolve paths relative to this script’s location
BASE_DIR    = os.path.abspath(os.path.dirname(__file__))
INPUT_PATH  = os.path.join(BASE_DIR, INPUT_FOLDER)
OUTPUT_PATH = os.path.join(BASE_DIR, OUTPUT_FOLDER)
CSV_PATH    = os.path.join(BASE_DIR, CSV_FILENAME)

def add_colorbar_from_csv(input_path, output_path, csv_path, cmap='viridis'):
    # 1) Load the CSV; expect a 'Frame' column and your min/max cols
    df = pd.read_csv(csv_path)
    df.set_index('Frame', inplace=True)

    os.makedirs(output_path, exist_ok=True)
    for filepath in sorted(glob.glob(os.path.join(input_path, '*.png'))):
        filename = os.path.basename(filepath)
        stem = os.path.splitext(filename)[0]

        # 2) Extract numeric frame from filename (e.g. 'entry_0001' -> 1)
        m = re.search(r'(\d+)', stem)
        if not m:
            print(f"⚠️  Could not parse frame number from '{filename}'; skipping.")
            continue
        frame_num = int(m.group(1))

        if frame_num not in df.index:
            print(f"⚠️  No CSV row for frame {frame_num}; skipping '{filename}'")
            continue

        # 3) Lookup your desired min/max
        row      = df.loc[frame_num]
        vmin_csv = row['Minimum Pressure (mmHg)']
        vmax_csv = row['Maximum Pressure (mmHg)']

        # 4) Plot the image with those exact bounds
        img = mpimg.imread(filepath)
        fig, ax = plt.subplots(figsize=(8,4))
        im = ax.imshow(img,
                       origin='upper',
                       cmap=cmap,
                       vmin=vmin_csv,
                       vmax=vmax_csv)
        ax.axis('off')

        # 5) Append a tight colorbar and restrict ticks to endpoints
        divider = make_axes_locatable(ax)
        cax     = divider.append_axes("right", size="5%", pad=0.0)
        cbar    = fig.colorbar(im, cax=cax)
        cbar.set_ticks([vmin_csv, vmax_csv])
        cax.yaxis.tick_right()

        # 6) Save
        out_path = os.path.join(output_path, filename)
        fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"Saved: {out_path}")

if __name__ == '__main__':
    add_colorbar_from_csv(INPUT_PATH, OUTPUT_PATH, CSV_PATH)
