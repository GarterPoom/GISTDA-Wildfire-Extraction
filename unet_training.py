# Import necessary libraries for deep learning, geospatial processing, and visualization
import torch  # PyTorch library for tensor operations and neural networks
import torch.nn as nn  # PyTorch module for neural network layers and functions
import torch.optim as optim  # PyTorch module for optimization algorithms
from torch.utils.data import Dataset, DataLoader  # PyTorch utilities for dataset handling and batching
import rasterio  # Library for reading and writing geospatial raster data (e.g., GeoTIFF)
from rasterio.windows import Window  # Utility for defining windowed reads from raster data
from rasterio.features import rasterize  # Function to convert vector geometries to raster format
import geopandas as gpd  # Library for handling geospatial vector data (e.g., shapefiles)
from shapely.geometry import mapping  # Function to convert geometries to GeoJSON-like format
import numpy as np  # Library for numerical operations on arrays
from skimage.transform import resize  # Function for resizing images or arrays
from tqdm import tqdm  # Progress bar for iterating over loops
import pandas as pd  # Library for data manipulation and analysis
import seaborn as sns  # Visualization library for creating heatmaps and plots
import matplotlib.pyplot as plt  # Plotting library for visualizations
from sklearn.metrics import classification_report, confusion_matrix  # Metrics for model evaluation
from sklearn.model_selection import train_test_split  # Utility to split data into training and validation sets

# DoubleConv Module: Two consecutive convolutional layers with batch normalization and ReLU activation
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        # Initialize the parent class (nn.Module)
        super().__init__()
        # Define a sequential block with two Conv2d -> BatchNorm -> ReLU layers
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # 3x3 convolution preserving input size
            nn.BatchNorm2d(out_channels),  # Normalize the output to stabilize training
            nn.ReLU(inplace=True),  # Apply ReLU activation in-place for memory efficiency
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # Second 3x3 convolution
            nn.BatchNorm2d(out_channels),  # Second normalization
            nn.ReLU(inplace=True)  # Second ReLU activation
        )

    def forward(self, x):
        # Forward pass: apply the double convolution block to the input tensor
        return self.double_conv(x)

# Down Module: Max pooling followed by a DoubleConv block for downsampling
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        # Initialize the parent class (nn.Module)
        super().__init__()
        # Define a sequential block with max pooling and DoubleConv
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # 2x2 max pooling to reduce spatial dimensions by half
            DoubleConv(in_channels, out_channels)  # Apply DoubleConv after pooling
        )

    def forward(self, x):
        # Forward pass: apply max pooling and DoubleConv
        return self.maxpool_conv(x)

# Up Module: Upsampling followed by concatenation with skip connection and DoubleConv
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        # Initialize the parent class (nn.Module)
        super().__init__()
        # Define transposed convolution for upsampling
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)  # Upsample by factor of 2
        # Define DoubleConv for processing concatenated feature maps
        self.conv = DoubleConv(in_channels, out_channels)  # Input channels include skip connection

    def forward(self, x1, x2):
        # Forward pass
        x1 = self.up(x1)  # Upsample the input tensor
        # Calculate padding to match x2's dimensions (skip connection)
        diffY = x2.size()[2] - x1.size()[2]  # Difference in height
        diffX = x2.size()[3] - x1.size()[3]  # Difference in width
        # Pad x1 to match x2's spatial dimensions
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # Concatenate skip connection (x2) with upsampled input (x1) along channel dimension
        x = torch.cat([x2, x1], dim=1)
        # Apply DoubleConv to concatenated tensor
        return self.conv(x)

# OutConv Module: Final 1x1 convolution to produce output channels
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        # Initialize the parent class (nn.Module)
        super().__init__()
        # Define 1x1 convolution to map to desired number of output channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Forward pass: apply 1x1 convolution
        return self.conv(x)

# UNet Model: Full U-Net architecture for image segmentation
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        # Initialize the parent class (nn.Module)
        super(UNet, self).__init__()
        # Store input and output channel counts
        self.n_channels = n_channels  # Number of input channels (e.g., image bands)
        self.n_classes = n_classes  # Number of output classes (e.g., 1 for binary segmentation)
        # Define encoder (contracting path)
        self.inc = DoubleConv(n_channels, 64)  # Initial DoubleConv block
        self.down1 = Down(64, 128)  # First downsampling block
        self.down2 = Down(128, 256)  # Second downsampling block
        self.down3 = Down(256, 512)  # Third downsampling block
        self.down4 = Down(512, 1024)  # Fourth downsampling block (bottleneck)
        # Define decoder (expansive path)
        self.up1 = Up(1024, 512)  # First upsampling block
        self.up2 = Up(512, 256)  # Second upsampling block
        self.up3 = Up(256, 128)  # Third upsampling block
        self.up4 = Up(128, 64)  # Fourth upsampling block
        # Define output layer
        self.outc = OutConv(64, n_classes)  # Final 1x1 convolution to produce segmentation map

    def forward(self, x):
        # Forward pass through encoder
        x1 = self.inc(x)  # Initial convolution
        x2 = self.down1(x1)  # First downsampling
        x3 = self.down2(x2)  # Second downsampling
        x4 = self.down3(x3)  # Third downsampling
        x5 = self.down4(x4)  # Bottleneck
        # Forward pass through decoder with skip connections
        x = self.up1(x5, x4)  # First upsampling with skip from x4
        x = self.up2(x, x3)  # Second upsampling with skip from x3
        x = self.up3(x, x2)  # Third upsampling with skip from x2
        x = self.up4(x, x1)  # Fourth upsampling with skip from x1
        # Produce final segmentation map
        logits = self.outc(x)
        return logits

# SegmentationDataset: Custom dataset for loading tiled GeoTIFF images and shapefile labels
class SegmentationDataset(Dataset):
    def __init__(self, image_path, label_path, windows, target_size=(256, 256), bands=None):
        # Initialize dataset parameters
        self.image_path = image_path  # Path to GeoTIFF image
        self.label_path = label_path  # Path to shapefile with labels
        self.windows = windows  # List of rasterio Window objects for tiling
        self.target_size = target_size  # Desired size for resized images/masks
        # Open GeoTIFF to get metadata
        with rasterio.open(image_path) as src:
            self.crs = src.crs  # Coordinate reference system of the image
            self.bands = bands if bands else list(range(1, src.count + 1))  # Select bands (default: all)
            self.num_bands = len(self.bands)  # Number of selected bands
        # Load shapefile and ensure CRS matches image
        self.gdf = gpd.read_file(label_path)  # Read shapefile with geopandas
        if self.gdf.crs != self.crs:
            self.gdf = self.gdf.to_crs(self.crs)  # Reproject shapefile to match image CRS

    def __len__(self):
        # Return the number of tiles (windows)
        return len(self.windows)

    def __getitem__(self, idx):
        # Get a single tile (image and mask) by index
        window = self.windows[idx]  # Select the window for this tile
        # Read image tile from GeoTIFF
        with rasterio.open(self.image_path) as src:
            image = src.read(self.bands, window=window)  # Read specified bands for the window
            tile_transform = src.window_transform(window)  # Get affine transform for the window
            height = window.height  # Height of the tile
            width = window.width  # Width of the tile
            bounds = rasterio.windows.bounds(window, src.transform)  # Get geographic bounds of the tile

        # Clip shapefile geometries to tile bounds
        clipped_gdf = self.gdf.clip(bounds)  # Clip shapefile to tile's geographic extent
        # Convert geometries to raster mask
        shapes = [(mapping(geom), 1) for geom in clipped_gdf.geometry if not geom.is_empty]  # Map valid geometries
        mask = rasterize(
            shapes,  # Geometries to rasterize
            out_shape=(height, width),  # Output shape of the mask
            transform=tile_transform,  # Affine transform for the tile
            fill=0,  # Unburn value (0 for non-labeled areas)
            all_touched=True,  # Include all pixels touched by geometries
            dtype=np.uint8  # 8-bit unsigned integer for binary mask
        )

        # Resize image and mask to target size
        image = resize(image.transpose(1, 2, 0), self.target_size + (self.num_bands,), mode='reflect', anti_aliasing=True)  # Resize image
        image = image.transpose(2, 0, 1).astype(np.float32)  # Transpose back to (C, H, W) and convert to float32
        # Normalize each channel of the image
        for c in range(image.shape[0]):
            channel = image[c]  # Select channel
            min_val = channel.min()  # Minimum value in channel
            max_val = channel.max()  # Maximum value in channel
            if max_val - min_val > 1e-6:  # Avoid division by zero
                image[c] = (channel - min_val) / (max_val - min_val)  # Normalize to [0, 1]
            else:
                image[c] = 0  # Set to 0 if channel has no variation

        # Resize mask (no interpolation for binary data)
        mask = resize(mask, self.target_size, order=0, anti_aliasing=False).astype(np.uint8)  # Resize mask
        mask = mask[None, :, :]  # Add channel dimension (1, H, W)

        # Return image and mask as PyTorch tensors
        return torch.from_numpy(image), torch.from_numpy(mask)

# evaluate_model: Function to evaluate the model and generate classification metrics
def evaluate_model(model, dataloader, device):
    # Set model to evaluation mode
    model.eval()
    all_preds = []  # List to store predictions
    all_labels = []  # List to store true labels

    # Disable gradient computation for evaluation
    with torch.no_grad():
        # Iterate over validation dataloader with progress bar
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)  # Move images to device (CPU/GPU)
            masks = masks.to(device).float()  # Move masks to device and convert to float

            # Forward pass
            outputs = model(images)  # Get model predictions
            preds = torch.sigmoid(outputs) > 0.5  # Apply sigmoid and threshold for binary classification
            preds = preds.cpu().numpy().flatten()  # Move to CPU and flatten
            masks = masks.cpu().numpy().flatten()  # Move to CPU and flatten

            # Collect predictions and labels
            all_preds.extend(preds)
            all_labels.extend(masks)

    # Generate classification report as a DataFrame
    report = classification_report(all_labels, all_preds, target_names=['Unburn', 'Burn'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()  # Convert report to DataFrame
    print("\nClassification Report:")  # Print header
    print(report_df)  # Display classification report

    # Generate and plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)  # Compute confusion matrix
    plt.figure(figsize=(8, 6))  # Create figure for confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Unburn', 'Burn'], yticklabels=['Unburn', 'Burn'])  # Plot heatmap
    plt.title('Confusion Matrix')  # Set title
    plt.ylabel('True Label')  # Set y-axis label
    plt.xlabel('Predicted Label')  # Set x-axis label
    plt.show()  # Display plot

    # Return classification report and confusion matrix
    return report_df, cm

# Main Training Script
if __name__ == "__main__":
    # Set up device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check for GPU availability
    print(f"Using device: {device}")  # Print device being used

    # Define paths to image and label files
    image_path = r"Raster_Classified\T47QLA_20250401T035601\T47QLA_20250401T035601.tif"  # Path to Sentinel-2 GeoTIFF
    label_path = r"Wildfire_Polygon\T47QLA_20250401T035601_masked_Burn_classified\T47QLA_20250401T035601_masked_Burn_classified_Burn_classified.shp"  # Path to shapefile

    # Define tile size (must match dataset target_size)
    tile_size = 256  # Size of tiles (256x256 pixels)

    # Generate windows for tiling the image
    with rasterio.open(image_path) as src:
        height, width = src.shape  # Get image dimensions (rows, cols)
        bands = list(range(1, src.count + 1))  # Use all available bands
        num_channels = len(bands)  # Number of bands (input channels)
        windows = []  # List to store windows
        # Iterate over image to create tiles
        for i in range(0, height, tile_size):
            for j in range(0, width, tile_size):
                w = min(tile_size, height - i)  # Width of tile (handle edge cases)
                h = min(tile_size, width - j)  # Height of tile (handle edge cases)
                if w == tile_size and h == tile_size:  # Only include full-sized tiles
                    windows.append(Window(j, i, tile_size, tile_size))  # Add window to list

    # Check if any valid tiles were generated
    if len(windows) == 0:
        print("No full tiles available. Image may be smaller than tile size.")  # Error message
        exit(1)  # Exit if no tiles are available

    # Split windows into training and validation sets
    train_windows, val_windows = train_test_split(windows, test_size=0.2, random_state=42)  # 80-20 split

    # Fallback if validation set is empty
    if len(val_windows) == 0:
        print("Validation set is empty. Consider adding more data or reducing test_size.")  # Warning message
        val_windows = train_windows  # Use training windows for validation (not ideal)

    # Create datasets for training and validation
    train_dataset = SegmentationDataset(image_path, label_path, train_windows, bands=bands)  # Training dataset
    val_dataset = SegmentationDataset(image_path, label_path, val_windows, bands=bands)  # Validation dataset
    # Create dataloaders for batching
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)  # Training dataloader
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)  # Validation dataloader

    # Initialize U-Net model
    model = UNet(n_channels=num_channels, n_classes=1).to(device)  # Create model and move to device

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss with logits
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Adam optimizer with learning rate 1e-4

    # Training loop
    num_epochs = 20  # Number of training epochs
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0  # Track total loss for the epoch
        # Create progress bar for training
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        # Iterate over training batches
        for images, masks in progress_bar:
            images = images.to(device)  # Move images to device
            masks = masks.to(device).float()  # Move masks to device and convert to float

            optimizer.zero_grad()  # Clear accumulated gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, masks)  # Compute loss
            loss.backward()  # Backpropagate gradients
            optimizer.step()  # Update model parameters

            running_loss += loss.item()  # Accumulate batch loss
            progress_bar.set_postfix(loss=loss.item())  # Update progress bar with current loss

        # Compute and print average loss for the epoch
        avg_loss = running_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Evaluate model on validation set
    report_df, cm = evaluate_model(model, val_dataloader, device)  # Generate metrics and plots

    # Save trained model
    torch.save(model.state_dict(), "Export_Model\\unet_model.pth")  # Save model weights
    print("Training complete. Model saved as unet_model.pth")  # Confirm model saved