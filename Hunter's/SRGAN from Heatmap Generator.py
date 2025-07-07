#!/usr/bin/env python3
"""
Batch Super-Resolution Script

This script finds all 6×6 PNG heatmaps in the "6x6_heatmaps" folder,
applies the SRGAN generator defined in model_inference.py to each one,
and saves the resulting 32×32 images to a folder named "32x32_heatmaps".
"""
import os
import sys
from PIL import Image

# Import the upscale function and script directory from the existing model_inference module
try:
    from model_inference import upscale_image, this_dir
except ImportError:
    print("Error: Could not import upscale_image from model_inference.py. Ensure it's in the same directory.")
    sys.exit(1)

# Configuration
INPUT_FOLDER = os.path.join(this_dir, "Test6x6_krigedheatmaps")
OUTPUT_FOLDER = os.path.join(this_dir, "TestKrigedSRGANReal32x32_heatmaps")


def ensure_folder(path):
    os.makedirs(path, exist_ok=True)


def main():
    # Create output directory
    ensure_folder(OUTPUT_FOLDER)

    # Verify input directory exists
    if not os.path.isdir(INPUT_FOLDER):
        print(f"Error: Input folder '{INPUT_FOLDER}' not found.")
        sys.exit(1)

    # List PNG files
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith('.png')]
    if not files:
        print(f"No PNG files found in '{INPUT_FOLDER}'")
        return

    # Process each file
    for filename in sorted(files):
        input_path = os.path.join(INPUT_FOLDER, filename)
        # Build output filename (e.g. entry_001.png -> entry_001_32x32.png)
        base, ext = os.path.splitext(filename)
        output_filename = f"{base}_32x32{ext}"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        try:
            # Perform super-resolution
            output_img = upscale_image(input_path)
            # Save the upscaled image
            output_img.save(output_path)
            print(f"Saved upscaled image: {output_path}")
        except Exception as e:
            print(f"Failed to process '{input_path}': {e}")


if __name__ == "__main__":
    main()
