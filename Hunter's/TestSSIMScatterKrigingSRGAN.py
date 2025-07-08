import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim  # pip install scikit-image

# 1. Folders (next to this script)
GT_FOLDER      = "TestRealActual32x32_heatmaps"
KRIGING_FOLDER = "TestKrigedSRGANReal32x32_heatmaps"
SRGAN_FOLDER   = "TestSRGANReal32x32_heatmaps"

# 2. Extensions and paths
EXTENSIONS = ('*.png','*.jpg','*.jpeg','*.tif','*.tiff')
base_dir   = os.path.dirname(os.path.abspath(__file__))
GT_DIR     = os.path.join(base_dir, GT_FOLDER)
KRIG_DIR   = os.path.join(base_dir, KRIGING_FOLDER)
SRGAN_DIR  = os.path.join(base_dir, SRGAN_FOLDER)

# 3. Gather files
def all_files(folder):
    out = []
    for ext in EXTENSIONS:
        out += glob.glob(os.path.join(folder, ext))
    return sorted(out)

gt_files    = all_files(GT_DIR)
krig_files  = all_files(KRIG_DIR)
srgan_files = all_files(SRGAN_DIR)

assert len(gt_files)==len(krig_files)==len(srgan_files), \
    f"Count mismatch: {len(gt_files)}, {len(krig_files)}, {len(srgan_files)}"

# 4. Load & metrics
def load_gray(path):
    im = Image.open(path).convert("L")
    return np.asarray(im, dtype=np.float32) / 255.0

def align_to(gt, pred):
    h,w = gt.shape
    if pred.shape != gt.shape:
        im = Image.fromarray((pred*255).astype(np.uint8))
        im = im.resize((w, h), resample=Image.BILINEAR)
        pred = np.asarray(im, dtype=np.float32)/255.0
    return pred

def mse(gt, p): return np.mean((gt-p)**2)
def rmse(gt,p): return np.sqrt(mse(gt,p))
def mae(gt,p):  return np.mean(np.abs(gt-p))
def maxe(gt,p): return np.max(np.abs(gt-p))

def ssim_index(gt, p):
    h,w = gt.shape
    win = min(7, h, w)
    if win%2==0: win-=1
    win = max(win, 3)
    return ssim(gt, p,
                data_range=1.0,
                win_size=win,
                channel_axis=None)

# 5. Compute metrics (with image_idx)
records = []
for idx, (g,k,s) in enumerate(zip(gt_files, krig_files, srgan_files)):
    gt = load_gray(g)
    kr = align_to(gt, load_gray(k))
    sr = align_to(gt, load_gray(s))
    for name,img in (("Kriging + SRGAN", kr), ("Only SRGAN", sr)):
        records.append({
            "image_idx": idx,
            "image":     os.path.basename(g),
            "method":    name,
            "MSE":       mse(gt,img),
            "RMSE":      rmse(gt,img),
            "MAE":       mae(gt,img),
            "MaxErr":    maxe(gt,img),
            "SSIM":      ssim_index(gt,img)
        })

df = pd.DataFrame(records)

# … (other plots unchanged) …

# → New SSIM scatter: x = image_idx, y = SSIM
plt.figure(figsize=(8,4))
for method, group in df.groupby("method"):
    plt.scatter(group["image_idx"], group["SSIM"], label=method, alpha=0.7)
plt.xlabel("Image Index")
plt.ylabel("SSIM")
plt.title("Per-Image SSIM by Method")
plt.legend()
plt.tight_layout()
plt.show()
