import os
import logging
import typing
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
from rasterio.windows import Window
from rasterio.enums import Resampling
from osgeo import gdal
import gc

class SentinelCloudMasker:
    """
    A memory-optimized class to handle cloud masking for Sentinel-2 imagery.
    Processes large files in chunks to prevent memory overload.
    """
    CLOUD_CLASSES = {
        3: "Cloud Shadow",
        6: "Water",
        8: "Cloud medium probability",
        9: "Cloud high probability", 
        10: "Thin cirrus"
    }
    
    # Define chunk size for processing (adjust based on available memory)
    CHUNK_SIZE = 2048  # Process 1024x1024 pixel chunks at a time

    def __init__(self, scl_dir: str, band_dir: str, output_dir: str, log_level: int = logging.INFO):
        """
        Initialize the cloud masking processor with chunked processing capabilities.
        """
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Validate directories
        self._validate_directories(scl_dir, band_dir, output_dir)
        
        self.scl_dir = Path(scl_dir)
        self.band_dir = Path(band_dir)
        self.output_dir = Path(output_dir)

        # Prepare file mappings
        self.scl_files = self._get_file_mapping(self.scl_dir, '_SCL.tif')
        self.band_files = self._get_band_file_mapping(self.band_dir)

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
            path = Path(dir_path)
            if not path.exists():
                raise ValueError(f"Directory does not exist: {dir_path}")
            if not path.is_dir():
                raise ValueError(f"Not a directory: {dir_path}")
            if not os.access(path, os.R_OK):
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
        for file_path in directory.rglob(f"*{suffix}"):
            # Extract tile ID and date from filename (e.g., T47PPT_20241221T034059)
            identifier = file_path.stem.split('_SCL')[0]
            file_mapping[identifier] = str(file_path)
            self.logger.debug(f"Found SCL file: {identifier} -> {file_path}")
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

        # Print summary of found files
        self.logger.info(f"Found {len(self.scl_files)} SCL files and {len(self.band_files)} matching band files")
        
        for identifier, scl_path in self.scl_files.items():
            if identifier not in self.band_files:
                self.logger.warning(f"No matching band file for {identifier}")
                continue

            try:
                self.logger.info(f"Processing: {identifier}")
                self.logger.debug(f"SCL file: {scl_path}")
                self.logger.debug(f"Band file: {self.band_files[identifier]}")
                
                with rasterio.open(scl_path) as scl_dataset, \
                     rasterio.open(self.band_files[identifier]) as band_dataset:
                    
                    # Create temporary output file
                    temp_output_path = str(self.output_dir / f"{identifier}_masked_temp.tif")
                    output_path = str(self.output_dir / f"{identifier}_masked.tif")
                    
                    with self.setup_output_dataset(band_dataset, temp_output_path) as dest:
                        # Process by chunks
                        total_chunks = sum(1 for _ in self._get_chunk_windows(band_dataset.width, band_dataset.height))
                        processed_chunks = 0
                        
                        for window in self._get_chunk_windows(band_dataset.width, band_dataset.height):
                            # Read data chunks
                            scl_chunk = scl_dataset.read(1, window=window)
                            band_chunk = band_dataset.read(window=window)
                            
                            # Create mask and process chunk
                            cloud_mask = self.create_cloud_mask(scl_chunk)
                            masked_chunk = self.process_chunk(band_chunk, cloud_mask)
                            
                            # Write chunk
                            for idx in range(masked_chunk.shape[0]):
                                dest.write(masked_chunk[idx], indexes=idx + 1, window=window)
                            
                            # Clear memory
                            del scl_chunk, band_chunk, cloud_mask, masked_chunk
                            gc.collect()
                            
                            # Update progress
                            processed_chunks += 1
                            if processed_chunks % 10 == 0:  # Log every 10 chunks
                                self.logger.debug(f"Processed {processed_chunks}/{total_chunks} chunks")

                    # Compress final output using GDAL
                    gdal.Translate(
                        destName=output_path,
                        srcDS=temp_output_path,
                        options=gdal.TranslateOptions(
                            format="GTiff",
                            creationOptions=[
                                "COMPRESS=LZW",
                                "PREDICTOR=3",
                                "TILED=YES",
                                "BLOCKXSIZE=256",
                                "BLOCKYSIZE=256",
                                "BIGTIFF=YES"
                            ]
                        )
                    )
                    
                    # Clean up temporary file
                    os.remove(temp_output_path)
                    
                    self.logger.info(f"Saved compressed masked GeoTIFF: {output_path}")
                    processed_count += 1
                    
                    # Force garbage collection
                    gc.collect()

            except Exception as e:
                self.logger.error(f"Error processing {identifier}: {e}")
                continue

        self.logger.info(f"Successfully processed {processed_count} files")

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