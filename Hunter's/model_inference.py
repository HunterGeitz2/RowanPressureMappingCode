# Author: Bhavitha Singamsetty
# model_inference.py

import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

# ===========================
# Generator Model Definition
# ===========================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Final upsampling to reach 32×32
            nn.Upsample(scale_factor=4/3, mode="bilinear", align_corners=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# ===========================
# Load Pre-trained Generator
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)

# Determine the directory where this script resides
this_dir = os.path.dirname(os.path.abspath(__file__))
# Load generator.pth from the same folder
generator_path = os.path.join(this_dir, "generator.pth")
if not os.path.exists(generator_path):
    raise FileNotFoundError(f"Generator weights not found: {generator_path}")
generator.load_state_dict(torch.load(generator_path, map_location=device))
generator.eval()

# ===========================
# Image Transforms
# ===========================
transform = transforms.Compose([
    transforms.Resize((6, 6)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

inv_transform = transforms.Compose([
    # Undo normalization from [-1,1] to [0,1]
    transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
    transforms.ToPILImage()
])

# ===========================
# Inference Function
# ===========================
def upscale_image(image_path):
    # Load and preprocess image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)  # Shape: [1,3,6,6]

    with torch.no_grad():
        output_tensor = generator(input_tensor).squeeze(0).cpu()  # Shape: [3,32,32]

    # Post-process and convert to PIL image
    output_tensor = output_tensor.clamp(-1, 1)
    output_image = inv_transform(output_tensor)
    return output_image

# ===========================
# Example Usage
# ===========================
if __name__ == "__main__":
    # Use script directory for relative paths
    input_filename = "pressure_image_6x6_122.png"
    input_path = os.path.join(this_dir, input_filename)
    output_filename = "output_32x32.png"
    output_path = os.path.join(this_dir, output_filename)

    # Check input exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Run inference and save output
    output_img = upscale_image(input_path)
    output_img.save(output_path)
    print(f"✅ Output saved to {output_path}")
