#!/usr/bin/env python3
import os
import sys
import argparse
import re
from PIL import Image

def make_comparisons(folder1, folder2, folder3, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    def get_pngs(folder):
        # list and sort PNGs by the numeric index in their filenames
        return sorted(
            [f for f in os.listdir(folder) if f.lower().endswith('.png')],
            key=lambda f: int(re.search(r"(\d+)", f).group(1))
        )

    files1 = get_pngs(folder1)
    files2 = get_pngs(folder2)
    files3 = get_pngs(folder3)

    # check that all folders have the same number of PNGs
    if not (len(files1) == len(files2) == len(files3)):
        raise ValueError(
            f"❌ Number of PNGs differs: {len(files1)} vs {len(files2)} vs {len(files3)}"
        )

    for f1, f2, f3 in zip(files1, files2, files3):
        img1 = Image.open(os.path.join(folder1, f1))
        img2 = Image.open(os.path.join(folder2, f2))
        img3 = Image.open(os.path.join(folder3, f3))

        # resize any smaller images to match img1's dimensions
        if img2.size != img1.size:
            img2 = img2.resize(img1.size, resample=Image.LANCZOS)
        if img3.size != img1.size:
            img3 = img3.resize(img1.size, resample=Image.LANCZOS)

        # create a new blank canvas wide enough for all three
        total_width = img1.width + img2.width + img3.width
        height = img1.height
        new_img = Image.new('RGBA', (total_width, height), (255, 255, 255, 0))

        # paste images side by side
        offset_x = 0
        for img in (img1, img2, img3):
            new_img.paste(img, (offset_x, 0))
            offset_x += img.width

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
        description="Combine matching PNGs from three folders side-by-side"
    )
    parser.add_argument(
        "folder1",
        nargs="?",
        default="TestRealActual6x6_heatmaps",
        help="1st heatmaps folder (default: TestRealActual6x6_heatmaps)"
    )
    parser.add_argument(
        "folder2",
        nargs="?",
        default="Test6x6_krigedheatmaps",
        help="2nd heatmaps folder (default: Test6x6_krigedheatmaps)"
    )
    parser.add_argument(
        "folder3",
        nargs="?",
        default="TestKrigedSRGANReal32x32_heatmaps",
        help="3rd heatmaps folder (default: TestKrigedSRGANReal32x32_heatmaps)"
    )
    parser.add_argument(
        "-o", "--output",
        default="TestKrigedcomparison",
        help="output folder (default: TestKrigedcomparison)"
    )

    args = parser.parse_args()

    f1 = resolve_and_check(args.folder1)
    f2 = resolve_and_check(args.folder2)
    f3 = resolve_and_check(args.folder3)
    out = args.output if os.path.isabs(args.output) else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), args.output
    )

    make_comparisons(f1, f2, f3, out)
