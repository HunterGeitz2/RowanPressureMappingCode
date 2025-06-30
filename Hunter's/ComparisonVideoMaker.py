#!/usr/bin/env python3
import os
import sys
import argparse

try:
    import cv2
except ImportError:
    print("❌ opencv-python is not installed. Run:\n    pip install opencv-python", file=sys.stderr)
    sys.exit(1)

def resolve_folder(path):
    # if absolute, use it; otherwise interpret relative to script’s directory
    if os.path.isabs(path):
        return path
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, path)

def make_video(folder, output, fps=30):
    folder = resolve_folder(folder)

    if not os.path.isdir(folder):
        print(f"❌ Folder not found: {folder}", file=sys.stderr)
        sys.exit(1)

    images = sorted(f for f in os.listdir(folder) if f.lower().endswith('.png'))
    if not images:
        print(f"❌ No PNGs in {folder}", file=sys.stderr)
        sys.exit(1)

    first = cv2.imread(os.path.join(folder, images[0]))
    if first is None:
        print(f"❌ Failed to load {images[0]}", file=sys.stderr)
        sys.exit(1)
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output, fourcc, fps, (w, h))

    for fname in images:
        path = os.path.join(folder, fname)
        frame = cv2.imread(path)
        if frame is None:
            print(f"⚠️  Skipping unreadable image: {fname}", file=sys.stderr)
            continue
        if frame.shape[0:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(frame)

    writer.release()
    print(f"✔️ Saved video: {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a video from PNGs using OpenCV"
    )
    parser.add_argument("folder", nargs="?", default="comparisonkriged",
                        help="Folder of PNGs (default: comparisonkriged)")
    parser.add_argument("-o", "--output", default="output.mp4",
                        help="Output video file (default: output.mp4)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second (default: 30)")
    args = parser.parse_args()

    make_video(args.folder, args.output, args.fps)
