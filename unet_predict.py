# prediction.py
# Standalone script to use the trained UNet model for prediction on a new GeoTIFF

# Import necessary libraries
import torch
import torch.nn as nn
import rasterio
from rasterio.windows import Window
import numpy as np
from skimage.transform import resize
from tqdm import tqdm

# DoubleConv Module: Two consecutive convolutional layers with batch normalization and ReLU activation
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# Down Module: Max pooling followed by a DoubleConv block for downsampling
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# Up Module: Upsampling followed by concatenation with skip connection and DoubleConv
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# OutConv Module: Final 1x1 convolution to produce output channels
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# UNet Model: Full U-Net architecture for image segmentation
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# Function to predict on a new GeoTIFF and save the mask
def predict_on_new_image(model_path, new_image_path, output_path, tile_size=256, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    with rasterio.open(new_image_path) as src:
        num_channels = src.count  # Assume same number of bands as training
    model = UNet(n_channels=num_channels, n_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully.")

    # Open the new GeoTIFF to get metadata and generate windows
    with rasterio.open(new_image_path) as src:
        height, width = src.shape
        meta = src.meta  # Metadata for output (transform, crs, etc.)
        bands = list(range(1, src.count + 1))  # All bands
        windows = []
        for i in range(0, height, tile_size):
            for j in range(0, width, tile_size):
                w = min(tile_size, width - j)
                h = min(tile_size, height - i)
                if w == tile_size and h == tile_size:  # Only full tiles (add padding if needed for edges)
                    windows.append(Window(j, i, w, h))

    if len(windows) == 0:
        raise ValueError("No full tiles available. Image may be smaller than tile size or not a multiple of it.")

    # Prepare full mask array (single band, same height/width as input)
    full_mask = np.zeros((height, width), dtype=np.uint8)

    # Process each tile
    with torch.no_grad():
        for window in tqdm(windows, desc="Predicting tiles"):
            with rasterio.open(new_image_path) as src:
                image = src.read(bands, window=window)  # Read tile (C, H, W)
                height_win, width_win = image.shape[1], image.shape[2]

            # Normalize and resize (like in dataset)
            image = resize(image.transpose(1, 2, 0), (tile_size, tile_size, num_channels), mode='reflect', anti_aliasing=True)
            image = image.transpose(2, 0, 1).astype(np.float32)
            for c in range(image.shape[0]):
                channel = image[c]
                min_val = channel.min()
                max_val = channel.max()
                if max_val - min_val > 1e-6:
                    image[c] = (channel - min_val) / (max_val - min_val)
                else:
                    image[c] = 0

            # Convert to tensor and predict
            image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)  # (1, C, H, W)
            output = model(image_tensor)
            pred = torch.sigmoid(output) > 0.35  # Binary mask
            pred = pred.cpu().numpy().squeeze(0).squeeze(0)  # (H, W)
            pred = resize(pred, (height_win, width_win), order=0, anti_aliasing=False).astype(np.uint8)  # Resize back if needed

            # Write to full mask
            full_mask[window.row_off:window.row_off + window.height, window.col_off:window.col_off + window.width] = pred

    # Save the mask as GeoTIFF (update metadata for single band)
    meta.update(count=1, dtype='uint8', nodata=0)
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(full_mask, 1)  # Write single band

    print(f"Prediction complete. Output saved to {output_path}")

# Example usage
if __name__ == "__main__":
    model_path = "Export_Model\\unet_model.pth"  # Path to saved model
    new_image_path = r"Raster_Classified\T46QGM_20250315T040611\T46QGM_20250315T040611.tif"  # Replace with your new GeoTIFF path
    output_path = r"Wildfire_Polygon\\T46QGM_20250315T040611_predicted_mask.tif"  # Desired output path
    predict_on_new_image(model_path, new_image_path, output_path)