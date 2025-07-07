#!/usr/bin/env python3
import os
import sys
import argparse
from PIL import Image

def make_comparisons(folder1, folder2, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    files1 = sorted(f for f in os.listdir(folder1) if f.lower().endswith('.png'))
    files2 = sorted(f for f in os.listdir(folder2) if f.lower().endswith('.png'))
    if len(files1) != len(files2):
        raise ValueError(f"❌ Number of PNGs differs: {len(files1)} vs {len(files2)}")

    for f1, f2 in zip(files1, files2):
        img1 = Image.open(os.path.join(folder1, f1))
        img2 = Image.open(os.path.join(folder2, f2))

        # --- resize the smaller image so both match img1's dimensions ---
        if img2.size != img1.size:
            img2 = img2.resize(img1.size, resample=Image.LANCZOS)

        # create a new blank canvas wide enough for both
        w, h = img1.width + img2.width, img1.height
        new_img = Image.new('RGBA', (w, h), (255, 255, 255, 0))
        new_img.paste(img1, (0, 0))
        new_img.paste(img2, (img1.width, 0))

        out_path = os.path.join(output_folder, f1)
        new_img.save(out_path)
        print(f"✔️ Saved {out_path}")

def resolve_and_check(path):
    base = os.path.dirname(os.path.abspath(__file__))
    p = os.path.abspath(path) if os.path.isabs(path) else os.path.join(base, path)
    if not os.path.isdir(p):
        print(f"❌ Folder not found: {p}")
        print("Contents of", os.path.dirname(p), ":")
        for item in sorted(os.listdir(os.path.dirname(p))):
            print("  ", item)
        sys.exit(1)
    return p

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine matching PNGs from two folders side-by-side"
    )
    parser.add_argument("folder1",
                        nargs="?",
                        default="TestRealActual6x6_heatmaps",
                        help="6x6 heatmaps folder (default: TestRealActual6x6_heatmaps)")
    parser.add_argument("folder2",
                        nargs="?",
                        default="TestSRGANReal32x32_heatmaps",
                        help="32x32 heatmaps folder (default: TestSRGANReal32x32_heatmaps)")
    parser.add_argument("-o", "--output",
                        default="Testcomparison",
                        help="output folder (default: Testcomparison)")

    args = parser.parse_args()

    f1 = resolve_and_check(args.folder1)
    f2 = resolve_and_check(args.folder2)
    out = args.output if os.path.isabs(args.output) else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), args.output
    )

    make_comparisons(f1, f2, out)
