import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# 1. User configuration: just folder names in the same directory as this script
GT_FOLDER      = "TestRealActual32x32_heatmaps"           # your ground-truth images
KRIGING_FOLDER = "TestKrigedSRGANReal32x32_heatmaps"      # your kriging outputs
SRGAN_FOLDER   = "TestSRGANReal32x32_heatmaps"            # your SRGAN outputs

# File extensions to look for
EXTENSIONS     = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff')

# 2. Determine absolute paths relative to the script location
base_dir      = os.path.dirname(os.path.abspath(__file__))
GT_DIR        = os.path.join(base_dir, GT_FOLDER)
KRIGING_DIR   = os.path.join(base_dir, KRIGING_FOLDER)
SRGAN_DIR     = os.path.join(base_dir, SRGAN_FOLDER)

# 3. Helper to gather image file paths
def all_files(folder):
    files = []
    for ext in EXTENSIONS:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)

gt_files      = all_files(GT_DIR)
kriging_files = all_files(KRIGING_DIR)
srgan_files   = all_files(SRGAN_DIR)

assert len(gt_files) == len(kriging_files) == len(srgan_files), (
    f"Counts mismatch: ground_truth({len(gt_files)}), "
    f"kriging({len(kriging_files)}), srgan({len(srgan_files)})"
)

# 4. Metric functions
def load_float(path):
    """Load image as a float32 array normalized to [0,1]."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr

def align_to(gt, pred):
    """Resize pred to match gt’s H×W if needed (bilinear interpolation)."""
    if pred.shape[:2] != gt.shape[:2]:
        h, w = gt.shape[:2]
        pil = Image.fromarray((pred * 255).astype(np.uint8))
        pil = pil.resize((w, h), resample=Image.BILINEAR)
        pred = np.asarray(pil, dtype=np.float32) / 255.0
    return pred

def mse(gt, pred):
    return np.mean((gt - pred) ** 2)

def rmse(gt, pred):
    return np.sqrt(mse(gt, pred))

def mae(gt, pred):
    return np.mean(np.abs(gt - pred))

def max_error(gt, pred):
    return np.max(np.abs(gt - pred))

# 5. Compute error metrics per image
records = []
for gt_path, kr_path, sr_path in zip(gt_files, kriging_files, srgan_files):
    gt = load_float(gt_path)
    kr = load_float(kr_path)
    sr = load_float(sr_path)
    
    # align sizes
    kr = align_to(gt, kr)
    sr = align_to(gt, sr)
    
    for method, img in (("Kriging and SRGAN", kr), ("Only SRGAN", sr)):
        records.append({
            "image":    os.path.basename(gt_path),
            "method":   method,
            "MSE":      mse(gt, img),
            "RMSE":     rmse(gt, img),
            "MAE":      mae(gt, img),
            "MaxError": max_error(gt, img)
        })

df = pd.DataFrame(records)

# 6. Bar chart: mean RMSE & MAE by method
agg = df.groupby("method")[["RMSE", "MAE"]].mean().reset_index()
x     = np.arange(len(agg))
width = 0.35

plt.figure(figsize=(6,4))
plt.bar(x - width/2, agg["RMSE"], width, label="RMSE")
plt.bar(x + width/2, agg["MAE"],  width, label="MAE")
plt.xticks(x, agg["method"])
plt.ylabel("Error")
plt.title("Mean Pixel-wise Errors (aligned to ground truth size)")
plt.legend()
plt.tight_layout()
plt.show()

# 7. Boxplot: per-image RMSE distributions
plt.figure(figsize=(6,4))
df.boxplot(column="RMSE", by="method")
plt.suptitle("")  # remove automatic title
plt.title("RMSE Distribution per Method")
plt.ylabel("RMSE")
plt.tight_layout()
plt.show()
