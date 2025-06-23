#Author:Bhavitha Singamsetty
# model_inference.py
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os

# ===========================
# Generator Model
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
            nn.Upsample(scale_factor=4/3, mode="bilinear", align_corners=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# ===========================
# Load Generator Model
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
generator.load_state_dict(torch.load("../models/generator.pth", map_location=device))
generator.eval()

# ===========================
# Transform Function
# ===========================
transform = transforms.Compose([
    transforms.Resize((6, 6)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

inv_transform = transforms.Compose([
    transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
    transforms.ToPILImage()
])

# ===========================
# Inference Function
# ===========================
def upscale_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)  # Shape: [1, 3, 6, 6]

    with torch.no_grad():
        output_tensor = generator(input_tensor).squeeze(0).cpu()  # Shape: [3, 32, 32]

    output_image = inv_transform(output_tensor.clamp(-1, 1))
    return output_image  # PIL.Image object

# ===========================
# Example usage
# ===========================
if __name__ == "__main__":
    input_path = "../images_6x6/pressure_image_6x6_122.png"
    output = upscale_image(input_path)
    output.save("output_32x32.png")
    print("✅ Output saved to output_32x32.png")
