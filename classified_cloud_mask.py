import os
import logging
import typing
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
from osgeo import gdal

class SentinelCloudMasker:
    """
    A class to handle cloud masking for Sentinel-2 imagery.
    """
    CLOUD_CLASSES = {
        3: "Cloud Shadow",
        6: "Water",
        8: "Cloud medium probability",
        9: "Cloud high probability", 
        10: "Thin cirrus"
    }

    def __init__(self, scl_dir: str, band_dir: str, output_dir: str, log_level: int = logging.INFO):
        """
        Initialize the cloud masking processor.
        """
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Validate directories
        self._validate_directories(scl_dir, band_dir, output_dir)
        
        self.scl_dir = scl_dir
        self.band_dir = band_dir
        self.output_dir = output_dir

        # Prepare file mappings
        self.scl_files = self._get_file_mapping(scl_dir, '_SCL.tif')
        self.band_files = self._get_file_mapping(band_dir, '.tif')

    def _validate_directories(self, *dirs: str) -> None:
        """
        Validate a list of directories.

        Ensure that each directory exists, is a directory, and is readable.

        Args:
            *dirs (str): List of directories to validate

        Raises:
            ValueError: If any of the directories do not exist, are not directories, or are not readable.
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
        Create a mapping of file identifiers to full file paths for a given directory and suffix.

        Args:
            directory (str): Directory to traverse for files.
            suffix (str): File suffix to match (e.g. '.tif').

        Returns:
            dict: Mapping of file identifiers to full file paths.
        """
        file_mapping = {}
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(suffix):
                    identifier = os.path.basename(file).split(suffix)[0]
                    file_mapping[identifier] = os.path.join(root, file)
        return file_mapping

    def _load_scl_layer(self, file_path: str) -> typing.Tuple[np.ndarray, Affine, CRS]:
        """
        Load a Sentinel-2 Scene Classification Layer (SCL) file and return its data, transform and CRS.

        Args:
            file_path (str): Path to the SCL file

        Returns:
            typing.Tuple[np.ndarray, Affine, CRS]: A tuple containing the SCL data, its transform and CRS

        Raises:
            rasterio.RasterioIOError: If there is an error reading the SCL file
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
        Create a cloud mask from Sentinel-2 Scene Classification Layer (SCL) data.

        This function generates a binary cloud mask by marking pixels as cloud-free 
        when they do not belong to the predefined cloud-related classes.

        Args:
            scl_data (np.ndarray): Scene Classification Layer data as a NumPy array.

        Returns:
            np.ndarray: A binary mask where True indicates cloud-free pixels and 
                        False indicates cloud-affected pixels based on the SCL data.
        """

        return ~np.isin(scl_data, list(self.CLOUD_CLASSES.keys()))

    def process_files(self) -> None:
        """
        Process all files in the SCL and band directories.

        This function will loop through all SCL files and:

        1. Load the SCL file and create a cloud mask
        2. Load the corresponding band file
        3. Mask the band data using the cloud mask
        4. Save the masked band data as an uncompressed GeoTIFF
        5. Compress the masked GeoTIFF using GDAL Translate
        6. Remove the intermediate uncompressed GeoTIFF

        :return: None
        """
        os.makedirs(self.output_dir, exist_ok=True)

        processed_count = 0
        for identifier, scl_path in self.scl_files.items():
            if identifier not in self.band_files:
                self.logger.warning(f"No matching band file for {identifier}")
                continue

            try:
                self.logger.info(f"Processing: {identifier}")

                # Load SCL and create mask
                scl_data, transform, crs = self._load_scl_layer(scl_path)
                cloud_mask = self.create_cloud_mask(scl_data)

                # Load band data
                band_path = self.band_files[identifier]
                with rasterio.open(band_path) as band_dataset:
                    band_data = band_dataset.read()  # Shape: [bands, height, width]
                    band_metadata = band_dataset.meta.copy()

                # Initialize masked bands
                masked_band = np.full_like(band_data, np.nan, dtype=np.float32)

                # Apply mask and validate bands
                for i in range(band_data.shape[0]):
                    self.logger.debug(f"Band {i + 1} original stats: min={np.min(band_data[i])}, max={np.max(band_data[i])}")
                    masked_band[i] = np.where(cloud_mask, band_data[i], np.nan)
                    self.logger.debug(f"Band {i + 1} masked stats: min={np.nanmin(masked_band[i])}, max={np.nanmax(masked_band[i])}")

                # Save intermediate file (uncompressed)
                temp_output_path = os.path.join(self.output_dir, f"{identifier}_masked_uncompressed.tif")
                band_metadata.update({
                    "driver": "GTiff",
                    "dtype": "float32",
                    "count": band_data.shape[0],
                    "transform": transform,
                    "crs": crs
                })

                with rasterio.open(temp_output_path, "w", **band_metadata) as dest:
                    for idx in range(band_data.shape[0]):
                        dest.write(masked_band[idx], indexes=idx + 1)

                # Compress using GDAL Translate with options
                output_path = os.path.join(self.output_dir, f"{identifier}_masked.tif")
                # Compress using GDAL Translate with enhanced options
                gdal.Translate(
                    destName=output_path,
                    srcDS=temp_output_path,
                    options=gdal.TranslateOptions(
                        format="GTiff",
                        creationOptions=[
                        "COMPRESS=LZW",        # LZW Compression
                        "PREDICTOR=3",         # Predictor for float32
                        "TILED=YES",           # Enable tiling
                        "BLOCKXSIZE=256",      # Tile width
                        "BLOCKYSIZE=256",
                        "BIGTIFF=YES"          # Tile height
                    ]
                )
            )
                
                # Remove intermediate file
                os.remove(temp_output_path)

                self.logger.info(f"Saved compressed masked GeoTIFF: {output_path}")
                processed_count += 1

            except Exception as e:
                self.logger.error(f"Error processing {identifier}: {e}")

        self.logger.info(f"Processed {processed_count} files")

def main():
    """
    Main function to perform cloud masking on Sentinel-2 imagery.

    This function initializes the SentinelCloudMasker with the specified
    directories for SCL and band files, as well as the output directory for
    processed files. It then proceeds to process the files while logging
    detailed debug information.

    Directories:
    - 'SCL_Classified': Directory containing SCL files for cloud masking.
    - 'Raster_Classified': Directory containing band files to be masked.
    - 'Raster_Classified_Cloud_Mask': Output directory for masked GeoTIFF files.

    Raises:
    - Logs an error message if an unexpected exception occurs during processing.
    """

    try:
        scl_dir = 'SCL_Classified'
        band_dir = 'Raster_Classified'
        output_dir = "Raster_Classified_Cloud_Mask"

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