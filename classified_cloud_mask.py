import os
import logging
import typing
import numpy as np
import rasterio
from osgeo import gdal
from rasterio.transform import Affine
from rasterio.crs import CRS

class SentinelCloudMasker:
    """
    A class to handle cloud masking for Sentinel-2 imagery
    """
    # Cloud classes from Sentinel-2 Scene Classification Layer (SCL)
    CLOUD_CLASSES = {
        3: "Cloud Shadow",
        6: "Water",
        8: "Cloud medium probability",
        9: "Cloud high probability", 
        10: "Thin cirrus"
    }

    def __init__(self, 
                 scl_dir: str, 
                 band_dir: str, 
                 output_dir: str, 
                 log_level: int = logging.INFO):
        """
        Initialize the cloud masking processor

        Args:
            scl_dir (str): Directory containing SCL files
            band_dir (str): Directory containing band files
            output_dir (str): Directory to save masked files
            log_level (int, optional): Logging level. Defaults to logging.INFO.
        """
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Validate and set directories
        self._validate_directories(scl_dir, band_dir, output_dir)
        
        self.scl_dir = scl_dir
        self.band_dir = band_dir
        self.output_dir = output_dir

        # Prepare file mappings
        self.scl_files = self._get_file_mapping(scl_dir, '_SCL.tif')
        self.band_files = self._get_file_mapping(band_dir, '.tif')

    def _validate_directories(self, *dirs: str) -> None:
        """
        Validate input directories exist and are accessible

        Args:
            dirs (str): Directories to validate

        Raises:
            ValueError: If a directory does not exist or is not accessible
        """
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                raise ValueError(f"Directory does not exist: {dir_path}")
            if not os.path.isdir(dir_path):
                raise ValueError(f"Not a directory: {dir_path}")
            if not os.access(dir_path, os.R_OK):
                raise ValueError(f"Directory not readable: {dir_path}")

    def _get_file_mapping(self, directory: str, suffix: str) -> dict:
        """
        Create a mapping of file identifiers to full file paths, including subfolders.

        Args:
            directory (str): Directory to search for files
            suffix (str): File suffix to filter

        Returns:
            dict: Mapping of file identifiers to full paths
        """
        file_mapping = {}
        for root, _, files in os.walk(directory):  # Recursively walk through subfolders
            for file in files:
                if file.endswith(suffix):
                    identifier = os.path.basename(file).split(suffix)[0]
                    file_mapping[identifier] = os.path.join(root, file)
        return file_mapping

    def _load_scl_layer(self, file_path: str) -> typing.Tuple[np.ndarray, Affine, CRS]:
        """
        Load Sentinel-2 Scene Classification Layer (SCL)

        Args:
            file_path (str): Path to SCL file

        Returns:
            Tuple containing SCL data, geotransform, and coordinate reference system
        """
        try:
            with rasterio.open(file_path) as dataset:
                scl_data = dataset.read(1)
                return scl_data, dataset.transform, dataset.crs
        except rasterio.RasterioIOError as e:
            self.logger.error(f"Error reading SCL file {file_path}: {e}")
            raise

    def create_cloud_mask(self, scl_data: np.ndarray) -> np.ndarray:
        """
        Create a cloud mask from SCL layer

        Args:
            scl_data (np.ndarray): Scene Classification Layer data

        Returns:
            np.ndarray: Boolean mask where True indicates non-cloud pixels
        """
        return ~np.isin(scl_data, list(self.CLOUD_CLASSES.keys()))

    def process_files(self) -> None:
        """
        Process files by matching SCL and band files, applying cloud masking,
        and exporting results as GeoTIFF files to the output directory.
        """
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Match and process files
        processed_count = 0
        for identifier, scl_path in self.scl_files.items():
            if identifier not in self.band_files:
                self.logger.warning(f"No matching band file for {identifier}")
                continue

            try:
                band_path = self.band_files[identifier]
                self.logger.info(f"Processing: {identifier}")

                # Load SCL and create mask
                scl_data, transform, crs = self._load_scl_layer(scl_path)
                cloud_mask = self.create_cloud_mask(scl_data)

                # Load band data (all bands)
                with rasterio.open(band_path) as band_dataset:
                    band_data = band_dataset.read()  # Read all bands (shape: [bands, height, width])
                    band_metadata = band_dataset.meta.copy()

                # Apply cloud mask to each band individually
                masked_band = np.zeros_like(band_data, dtype=np.float32)  # Ensure dtype is float32
                for i in range(band_data.shape[0]):
                    # Apply mask: set pixels in CLOUD_CLASSES to NaN
                    masked_band[i] = np.where(cloud_mask, band_data[i], np.nan)

                # Prepare output metadata
                band_metadata.update({
                    "driver": "GTiff",
                    "dtype": "float32",
                    "count": masked_band.shape[0],
                    "compress": "LZW"
                    # Do not set "nodata" explicitly, as NaN is inherently handled
                })
                
                # Save masked bands to GeoTIFF
                output_path = os.path.join(self.output_dir, f"{identifier}_masked.tif")
                with rasterio.open(output_path, "w", **band_metadata) as dest:
                    for i in range(masked_band.shape[0]):
                        dest.write(masked_band[i], indexes=i + 1)  # Write each band

                processed_count += 1
                self.logger.info(f"Saved masked GeoTIFF: {output_path}")

            except Exception as e:
                self.logger.error(f"Error processing {identifier}: {e}")

        self.logger.info(f"Processed {processed_count} files")

def main():
    """
    Main function to execute cloud masking process
    """
    try:
        # Configurable directories
        scl_dir = "SCL Classified"
        band_dir = "Raster Classified"
        output_dir = "Raster Classified Cloud Mask"

        # Create and run cloud masker
        cloud_masker = SentinelCloudMasker(
            scl_dir, 
            band_dir, 
            output_dir, 
            log_level=logging.DEBUG
        )
        cloud_masker.process_files()

    except Exception as e:
        logging.error(f"Unexpected error in main process: {e}")

if __name__ == "__main__":
    main()
