import rasterio
import numpy as np
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from rasterio.windows import Window, get_data_window
from shapely.geometry import box
import logging
from contextlib import contextmanager
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentinelProcessor:
    def __init__(self, root_dir, chunk_size=1024, tile_size=1024):
        """
        Initialize the Sentinel processor with root directory.

        Args:
            root_dir (str): Root directory for processing
            chunk_size (int): Size of chunks for processing (default: 1024)
            tile_size (int): Size of output tiles (default: 1024)
        """
        self.root_dir = Path(root_dir).resolve()
        self.chunk_size = chunk_size
        self.tile_size = tile_size
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
    def get_optimal_chunk_size(total_size, mem_limit_mb=200):
        """
        Calculate optimal chunk size based on memory limit.
        
        Args:
            total_size (int): Total size of the image dimension
            mem_limit_mb (int): Memory limit in megabytes per chunk
            
        Returns:
            int: Optimal chunk size
        """
        mem_bytes = mem_limit_mb * 1024 * 1024  # Convert MB to bytes
        pixel_bytes = 4  # float32 = 4 bytes
        bands = 16  # Number of output bands
        
        # Calculate maximum pixels that fit in memory limit
        max_pixels = mem_bytes / (pixel_bytes * bands)
        
        # Calculate chunk size (square root of max pixels)
        chunk_size = int(math.sqrt(max_pixels))
        
        # Round down to nearest multiple of 256 for efficiency
        chunk_size = (chunk_size // 256) * 256
        
        # Ensure minimum chunk size of 256
        return max(256, min(chunk_size, total_size))

    @staticmethod
    def calculate_indices(pre_bands, post_bands):
        """
        Calculate various spectral indices.
        
        Args:
            pre_bands (dict): Dictionary of pre-fire band data
            post_bands (dict): Dictionary of post-fire band data
            
        Returns:
            tuple: (dnbr, ndwi, ndvi) calculated indices
        """
        with np.errstate(divide='ignore', invalid='ignore'):
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
            dnbr (np.array): Differenced Normalized Burn Ratio
            ndwi (np.array): Normalized Difference Water Index
            ndvi (np.array): Normalized Difference Vegetation Index
            b08 (np.array): Band 8 data
            
        Returns:
            np.array: Binary burn label mask
        """
        burn_label = np.where(
            (dnbr > 0.27) & (ndwi < 0) & (ndvi < 0.14) & (b08 < 2500),
            1,
            0
        ).astype(np.float32)

        burn_label[~np.isfinite(dnbr) | ~np.isfinite(ndwi) | 
                  ~np.isfinite(ndvi) | ~np.isfinite(b08)] = 0
        
        return burn_label

    def process_chunk(self, pre_src, post_src, window):
        """
        Process a single chunk of the image.
        
        Args:
            pre_src: Pre-fire rasterio dataset
            post_src: Post-fire rasterio dataset
            window: Rasterio window object defining the chunk
            
        Returns:
            dict: Processed band data and indices
        """
        band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06',
                     'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
        
        # Read chunks with proper masks
        pre_bands = {}
        post_bands = {}
        
        for i, band_name in enumerate(band_names):
            pre_data = pre_src.read(i + 1, window=window, masked=True)
            post_data = post_src.read(i + 1, window=window, masked=True)
            
            # Convert to float32 and handle masked values
            pre_bands[band_name] = pre_data.filled(np.nan).astype(np.float32)
            post_bands[band_name] = post_data.filled(np.nan).astype(np.float32)

        # Calculate indices
        dnbr, ndwi, ndvi = self.calculate_indices(pre_bands, post_bands)
        
        # Create burn label
        burn_label = self.create_burn_label(dnbr, ndwi, ndvi, post_bands['B08'])

        return {
            **post_bands,
            'dNBR': dnbr,
            'NDVI': ndvi,
            'NDWI': ndwi,
            'Burn_Label': burn_label
        }

    def _save_chunk(self, chunk_data, meta, window, original_filename, output_dir):
        """
        Save processed chunk data.
        
        Args:
            chunk_data (dict): Processed chunk data
            meta (dict): Rasterio metadata
            window: Rasterio window object
            original_filename (str): Base filename for output
            output_dir (Path): Directory to save chunk
        """
        chunk_meta = meta.copy()
        chunk_meta.update({
            'height': window.height,
            'width': window.width,
            'transform': rasterio.transform.from_bounds(
                *self._get_tile_bounds(meta['transform'], window),
                window.width,
                window.height
            )
        })
        
        chunk_name = f"{original_filename}_chunk_{window.row_off}_{window.col_off}.tif"
        chunk_path = output_dir / chunk_name
        
        with rasterio.open(chunk_path, 'w', **chunk_meta) as dst:
            for i, (band_name, data) in enumerate(chunk_data.items()):
                dst.write(data, i + 1)
                dst.set_band_description(i + 1, band_name)
        
        logger.info(f"Saved chunk: {chunk_path}")

    @staticmethod
    def _get_tile_bounds(transform, window):
        """
        Calculate bounds for a tile.
        
        Args:
            transform: Rasterio transform
            window: Rasterio window object
            
        Returns:
            tuple: (left, bottom, right, top) bounds
        """
        left = transform[2] + window.col_off * transform[0]
        top = transform[5] + window.row_off * transform[4]
        right = left + window.width * transform[0]
        bottom = top + window.height * transform[4]
        return left, bottom, right, top

    def process_tile_pair(self, tile_id, paths):
        """
        Process a pair of pre/post-fire images using chunked processing.
        
        Args:
            tile_id (str): Identifier for the tile pair
            paths (dict): Dictionary containing paths to pre and post images
        """
        if not paths['pre'] or not paths['post']:
            logger.warning(f"Missing pre or post image for tile {tile_id}")
            return

        try:
            with rasterio.open(paths['post']) as post_src:
                # Calculate optimal chunk size based on image dimensions
                img_size = max(post_src.width, post_src.height)
                chunk_size = self.get_optimal_chunk_size(img_size)
                
                output_profile = post_src.profile.copy()
                output_profile.update({
                    'count': 16,
                    'dtype': 'float32',
                    'nodata': np.nan,
                })

                # Generate output directory
                post_filename = Path(paths['post']).stem
                tile_date = post_filename.split('_')[1]
                tile_output_dir = self.output_dir / f"{tile_id}_{tile_date}"
                tile_output_dir.mkdir(parents=True, exist_ok=True)

                # Process image in chunks
                for y in range(0, post_src.height, chunk_size):
                    for x in range(0, post_src.width, chunk_size):
                        # Calculate actual chunk size (handling edges)
                        effective_width = min(chunk_size, post_src.width - x)
                        effective_height = min(chunk_size, post_src.height - y)
                        
                        window = Window(x, y, effective_width, effective_height)
                        
                        # Process chunk
                        with rasterio.open(paths['pre']) as pre_src:
                            chunk_data = self.process_chunk(pre_src, post_src, window)
                        
                        # Save chunk as a tile
                        chunk_output_dir = tile_output_dir / f"chunk_{y}_{x}"
                        chunk_output_dir.mkdir(parents=True, exist_ok=True)
                        
                        self._save_chunk(
                            chunk_data,
                            output_profile,
                            window,
                            f"{tile_id}_{tile_date}",
                            chunk_output_dir
                        )

                logger.info(f"Processed {tile_id} in chunks")

        except Exception as e:
            logger.error(f"Error processing tile pair {tile_id}: {str(e)}")
            raise

    def get_tile_pairs(self):
        """
        Get pairs of pre/post-fire images from input directory.
        
        Returns:
            dict: Dictionary of tile pairs with pre/post image paths
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

    def process_all(self, max_workers=2):
        """
        Process all tile pairs in parallel.
        
        Args:
            max_workers (int): Number of parallel processes to use
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
        root_dir = Path("Raster").resolve()
        
        processor = SentinelProcessor(
            root_dir=root_dir,
            chunk_size=1024,
            tile_size=1024
        )
        
        logger.info(f"Processing Sentinel-2 images in {root_dir}...")
        processor.process_all(max_workers=2)
        logger.info("Processing completed successfully")
        
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()