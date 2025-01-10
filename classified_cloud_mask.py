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
        """Validate directory existence and permissions."""
        for dir_path in dirs:
            path = Path(dir_path)
            if not path.exists():
                raise ValueError(f"Directory does not exist: {dir_path}")
            if not path.is_dir():
                raise ValueError(f"Not a directory: {dir_path}")
            if not os.access(path, os.R_OK):
                raise ValueError(f"Directory not readable: {dir_path}")

    def _get_file_mapping(self, directory: Path, suffix: str) -> dict:
        """Create a mapping of identifiers to file paths for SCL files."""
        file_mapping = {}
        for file_path in directory.rglob(f"*{suffix}"):
            # Extract tile ID and date from filename (e.g., T47PPT_20241221T034059)
            identifier = file_path.stem.split('_SCL')[0]
            file_mapping[identifier] = str(file_path)
            self.logger.debug(f"Found SCL file: {identifier} -> {file_path}")
        return file_mapping

    def _get_band_file_mapping(self, directory: Path) -> dict:
        """Create a mapping of identifiers to file paths for band files, handling subfolders."""
        file_mapping = {}
        # Walk through all subfolders
        for file_path in directory.rglob("*.tif"):
            if '_SCL.tif' not in str(file_path):  # Skip SCL files if they exist in band directory
                # Extract the base identifier (tile ID and date) from the filename
                filename = file_path.stem
                # Look for patterns like T47PPT_20241221T034059
                for scl_id in self.scl_files.keys():
                    if scl_id in filename:
                        file_mapping[scl_id] = str(file_path)
                        self.logger.debug(f"Found matching band file: {scl_id} -> {file_path}")
                        break
        return file_mapping

    def _get_chunk_windows(self, width: int, height: int) -> typing.Iterator[Window]:
        """Generate processing windows for chunked reading."""
        for y in range(0, height, self.CHUNK_SIZE):
            for x in range(0, width, self.CHUNK_SIZE):
                chunk_width = min(self.CHUNK_SIZE, width - x)
                chunk_height = min(self.CHUNK_SIZE, height - y)
                yield Window(x, y, chunk_width, chunk_height)

    def create_cloud_mask(self, scl_data: np.ndarray) -> np.ndarray:
        """Create a boolean mask identifying non-cloud pixels."""
        mask = ~np.isin(scl_data, list(self.CLOUD_CLASSES.keys()))
        return mask

    def process_chunk(self, 
                     band_data: np.ndarray, 
                     cloud_mask: np.ndarray) -> np.ndarray:
        """Process a single chunk of data."""
        masked_chunk = np.full_like(band_data, np.nan, dtype=np.float32)
        for i in range(band_data.shape[0]):
            masked_chunk[i] = np.where(cloud_mask, band_data[i], np.nan)
        return masked_chunk

    def setup_output_dataset(self, 
                           template_dataset: rasterio.io.DatasetReader, 
                           output_path: str) -> rasterio.io.DatasetWriter:
        """Set up the output dataset with appropriate metadata."""
        metadata = template_dataset.meta.copy()
        metadata.update({
            "driver": "GTiff",
            "dtype": "float32",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "compress": 'lzw',
            "predictor": 3,
            "interleave": 'band'
        })
        return rasterio.open(output_path, "w", **metadata)

    def process_files(self) -> None:
        """Process files in chunks to minimize memory usage."""
        self.output_dir.mkdir(exist_ok=True)
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