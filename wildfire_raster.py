import rasterio
import numpy as np
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from rasterio.windows import Window
from shapely.geometry import box
import logging
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentinelProcessor:
    def __init__(self, root_dir, chunk_size=1024):
        """
        Initialize the Sentinel processor with root directory.

        Args:
            root_dir (str): Root directory for processing
            chunk_size (int): Size of chunks for processing (default: 1024)
        """
        self.root_dir = Path(root_dir).resolve()
        self.chunk_size = chunk_size
        self.input_dir, self.output_dir = self._setup_directories()

    def _setup_directories(self):
        """
        Set up input and output directories.

        Returns:
            tuple: (input_dir, output_dir) as Path objects
        """
        input_dir = self.root_dir / 'input'
        output_dir = self.root_dir / 'output'
        
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return input_dir, output_dir

    @contextmanager
    def _open_rasters(self, pre_path, post_path):
        """
        Context manager to safely open and close raster files.
        
        Args:
            pre_path (str): Path to pre-fire image
            post_path (str): Path to post-fire image
            
        Yields:
            tuple: (pre_src, post_src) rasterio dataset objects
        """
        pre_src = post_src = None
        try:
            pre_src = rasterio.open(pre_path)
            post_src = rasterio.open(post_path)
            yield pre_src, post_src
        finally:
            if pre_src is not None:
                pre_src.close()
            if post_src is not None:
                post_src.close()

    @staticmethod
    def calculate_indices(pre_bands, post_bands):
        """
        Calculate various spectral indices.

        Args:
            pre_bands (dict): Dictionary of pre-fire bands
            post_bands (dict): Dictionary of post-fire bands

        Returns:
            tuple: dNBR, NDWI, and NDVI indices
        """

        # Calculate indices
        nbr_pre = (pre_bands['B8A'] - pre_bands['B12']) / (pre_bands['B8A'] + pre_bands['B12'])
        nbr_post = (post_bands['B8A'] - post_bands['B12']) / (post_bands['B8A'] + post_bands['B12'])
        dnbr = nbr_pre - nbr_post

        ndwi = (post_bands['B03'] - post_bands['B08']) / (post_bands['B03'] + post_bands['B08'])
        ndvi = (post_bands['B08'] - post_bands['B04']) / (post_bands['B08'] + post_bands['B04'])
        
        return dnbr, ndwi, ndvi

    @staticmethod
    def create_burn_label(dnbr, ndwi, ndvi, b08):
        """
        Create burn label mask based on spectral indices.

        Args:
            dnbr (np.ndarray): Difference in Normalized Burn Ratio
            ndwi (np.ndarray): Normalized Difference Water Index
            ndvi (np.ndarray): Normalized Difference Vegetation Index
            b08 (np.ndarray): Band 8 (NIR) data

        Returns:
            np.ndarray: Burn label mask (0: no burn, 1: burn)
        """
        result = ((dnbr > 0.27) & (ndwi < 0) & (ndvi < 0.14) & (b08 < 2500)).astype(np.float32)
        result[~np.isfinite(dnbr) | ~np.isfinite(ndwi) | ~np.isfinite(ndvi) | ~np.isfinite(b08)] = 0
        
        return result

    def process_chunk(self, pre_src, post_src, window):
        """
        Process a single chunk of the image.
        
        Args:
            pre_src: Pre-fire rasterio dataset
            post_src: Post-fire rasterio dataset
            window: rasterio.windows.Window object
            
        Returns:
            dict: Processed data for the chunk
        """
        # Read bands for the chunk
        band_names = [
            'B01', 'B02', 'B03', 'B04', 'B05', 'B06',
            'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'
        ]
        
        pre_bands = {
            band_name: pre_src.read(i + 1, window=window).astype(np.float32)
            for i, band_name in enumerate(band_names)
        }
        
        post_bands = {
            band_name: post_src.read(i + 1, window=window).astype(np.float32)
            for i, band_name in enumerate(band_names)
        }

        # Calculate indices for the chunk
        dnbr, ndwi, ndvi = self.calculate_indices(pre_bands, post_bands)
        
        # Calculate burn label for the chunk
        burn_label = self.create_burn_label(dnbr, ndwi, ndvi, post_bands['B08'])

        return {
            **post_bands,
            'dNBR': dnbr,
            'NDVI': ndvi,
            'NDWI': ndwi,
            'Burn_Label': burn_label
        }

    def process_tile_pair(self, tile_id, paths):
        """
        Process a pair of pre/post-fire images in chunks but save as a single file.
        
        Args:
            tile_id (str): Unique identifier for the tile
            paths (dict): Dictionary with 'pre' and 'post' paths
        """
        if not paths['pre'] or not paths['post']:
            logger.warning(f"Missing pre or post image for tile {tile_id}")
            return

        try:
            with self._open_rasters(paths['pre'], paths['post']) as (pre_src, post_src):
                # Get dimensions
                height = post_src.height
                width = post_src.width

                # Create output profile
                output_profile = post_src.profile.copy()
                # Update the output profile to enable BigTIFF
                output_profile.update({
                    'count': 16,  # 12 bands + dNBR, NDVI, NDWI, Burn_Label
                    'dtype': 'float32',
                    'nodata': np.nan,
                    'BIGTIFF': 'YES',  # Enable BigTIFF to handle large files
                    'compress': 'LZW'  # or 'LZW'
                })

                # Generate output path and filename
                post_filename = Path(paths['post']).stem
                tile_date = post_filename.split('_')[1]
                output_filename = f"{tile_id}_{tile_date}_processed.tif"
                output_path = self.output_dir / output_filename

                # Create output file
                with rasterio.open(output_path, 'w', **output_profile) as dst:
                    # Process in chunks
                    for y in range(0, height, self.chunk_size):
                        for x in range(0, width, self.chunk_size):
                            window = Window(
                                x, y,
                                min(self.chunk_size, width - x),
                                min(self.chunk_size, height - y)
                            )

                            # Process chunk
                            chunk_data = self.process_chunk(pre_src, post_src, window)

                            # Write chunk data to the appropriate location
                            for i, (band_name, data) in enumerate(chunk_data.items()):
                                dst.write(data, i + 1, window=window)
                                # Set band description only once per band
                                if x == 0 and y == 0:
                                    dst.set_band_description(i + 1, band_name)

                            logger.debug(f"Processed chunk at {x},{y} for tile {tile_id}")

                logger.info(f"Processed and saved tile {tile_id} to {output_path}")

        except Exception as e:
            logger.error(f"Error processing tile pair {tile_id}: {str(e)}")
            raise

    def get_tile_pairs(self):
        """
        Get pairs of pre/post-fire images from input directory.

        Returns:
            dict: Dictionary mapping tile IDs to pre/post image paths
        """
        tile_pairs = {}

        try:
            for file_path in self.input_dir.rglob('*.tif'):
                tile_id = file_path.stem.split('_')[0]
                if tile_id not in tile_pairs:
                    tile_pairs[tile_id] = {'pre': None, 'post': None}
                
                if 'pre' in file_path.stem.lower():
                    tile_pairs[tile_id]['pre'] = str(file_path)
                elif 'post' in file_path.stem.lower():
                    tile_pairs[tile_id]['post'] = str(file_path)
            
            return tile_pairs
            
        except Exception as e:
            logger.error(f"Error getting tile pairs: {str(e)}")
            raise

    def process_all(self, max_workers=None):
        """
        Process all tile pairs in parallel.

        Args:
            max_workers (int, optional): Maximum number of worker processes
        """
        try:
            tile_pairs = self.get_tile_pairs()

            if not tile_pairs:
                logger.warning("No tile pairs found in input directory")
                return

            logger.info(f"Found {len(tile_pairs)} tile pairs to process")

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self.process_tile_pair, tile_id, paths)
                    for tile_id, paths in tile_pairs.items()
                ]

                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error in worker process: {str(e)}")

        except Exception as e:
            logger.error(f"Error in process_all: {str(e)}")
            raise

def main():
    """Main entry point for the script."""
    try:
        # Use absolute path for root directory
        root_dir = Path("Raster").resolve()
        
        # Initialize processor with memory-efficient chunk size
        processor = SentinelProcessor(
            root_dir=root_dir,
            chunk_size=1024  # Process in 1024x1024 chunks
        )
        
        logger.info(f"Processing Sentinel-2 images in {root_dir}...")
        processor.process_all()
        logger.info("Processing completed successfully")
        
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()