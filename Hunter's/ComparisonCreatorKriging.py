#!/usr/bin/env python3
import os
import sys
import argparse
from PIL import Image

def make_comparisons(folder1, folder2, folder3, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    files1 = sorted(f for f in os.listdir(folder1) if f.lower().endswith('.png'))
    files2 = sorted(f for f in os.listdir(folder2) if f.lower().endswith('.png'))
    files3 = sorted(f for f in os.listdir(folder3) if f.lower().endswith('.png'))

    if not (len(files1) == len(files2) == len(files3)):
        raise ValueError(
            f"❌ Number of PNGs differs: "
            f"{len(files1)} (folder1) vs {len(files2)} (folder2) vs {len(files3)} (folder3)"
        )

    for f1, f2, f3 in zip(files1, files2, files3):
        img1 = Image.open(os.path.join(folder1, f1))
        img2 = Image.open(os.path.join(folder2, f2))
        img3 = Image.open(os.path.join(folder3, f3))

        # resize to match img1's dimensions if needed
        target_size = img1.size
        if img2.size != target_size:
            img2 = img2.resize(target_size, resample=Image.LANCZOS)
        if img3.size != target_size:
            img3 = img3.resize(target_size, resample=Image.LANCZOS)

        # create canvas wide enough for three images
        w = img1.width + img2.width + img3.width
        h = img1.height
        new_img = Image.new('RGBA', (w, h), (255, 255, 255, 0))

        # paste in order: raw heatmap, kriged 6x6, kriged 32x32
        new_img.paste(img1, (0, 0))
        new_img.paste(img2, (img1.width, 0))
        new_img.paste(img3, (img1.width + img2.width, 0))

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
    parser.add_argument("folder1",
                        nargs="?",
                        default="6x6_heatmaps",
                        help="6x6 heatmaps folder (default: 6x6_heatmaps)")
    parser.add_argument("folder2",
                        nargs="?",
                        default="6x6_krigedheatmaps",
                        help="6x6 kriged heatmaps folder (default: 6x6_krigedheatmaps)")
    parser.add_argument("folder3",
                        nargs="?",
                        default="32x32_krigedheatmaps",
                        help="32x32 kriged heatmaps folder (default: 32x32_krigedheatmaps)")
    parser.add_argument("-o", "--output",
                        default="comparisonkriged",
                        help="output folder (default: comparisonkriged)")

    args = parser.parse_args()

    f1 = resolve_and_check(args.folder1)
    f2 = resolve_and_check(args.folder2)
    f3 = resolve_and_check(args.folder3)
    out = args.output if os.path.isabs(args.output) else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), args.output
    )

    make_comparisons(f1, f2, f3, out)
