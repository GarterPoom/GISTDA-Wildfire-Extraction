import rasterio
import numpy as np
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from rasterio.windows import Window
from rasterio.enums import Compression
from shapely.geometry import box
import logging
from contextlib import contextmanager

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
    def calculate_indices(pre_bands, post_bands):
        """Calculate various spectral indices."""
        nbr_pre = (pre_bands['B8A'] - pre_bands['B12']) / (pre_bands['B8A'] + pre_bands['B12'])
        nbr_post = (post_bands['B8A'] - post_bands['B12']) / (post_bands['B8A'] + post_bands['B12'])
        dnbr = nbr_pre - nbr_post

        ndwi = (post_bands['B03'] - post_bands['B08']) / (post_bands['B03'] + post_bands['B08'])
        ndvi = (post_bands['B08'] - post_bands['B04']) / (post_bands['B08'] + post_bands['B04'])
        
        return dnbr, ndwi, ndvi

    @staticmethod
    def create_burn_label(dnbr, ndwi, ndvi, b08):
        """Create burn label mask based on spectral indices."""
        result = np.where(
            (dnbr > 0.27) & (ndwi < 0) & (ndvi < 0.14) & (b08 < 2500), 
            1,  # Burn label
            0   # Non-burn label
        ).astype(np.float32)

        result[~np.isfinite(dnbr) | ~np.isfinite(ndwi) | ~np.isfinite(ndvi) | ~np.isfinite(b08)] = 0
        return result

    def process_chunk(self, pre_src, post_src, window):
        """Process a single chunk of the image."""
        band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06',
                     'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
        
        pre_bands = {
            band_name: pre_src.read(i + 1, window=window).astype(np.float32)
            for i, band_name in enumerate(band_names)
        }
        
        post_bands = {
            band_name: post_src.read(i + 1, window=window).astype(np.float32)
            for i, band_name in enumerate(band_names)
        }

        dnbr, ndwi, ndvi = self.calculate_indices(pre_bands, post_bands)
        burn_label = self.create_burn_label(dnbr, ndwi, ndvi, post_bands['B08'])

        return {
            **post_bands,
            'dNBR': dnbr,
            'NDVI': ndvi,
            'NDWI': ndwi,
            'Burn_Label': burn_label
        }

    def save_tile_with_compression(self, tile_data, meta, window, output_path):
        """Save tile with compression."""
        tile_meta = meta.copy()
        tile_meta.update({
            "height": window.height,
            "width": window.width,
            "transform": rasterio.windows.transform(window, meta['transform']),
            "compress": Compression.lzw,
            "tiled": True,
            "BIGTIFF": "YES"  # Enable BIGTIFF to support large files
        })
        
        with rasterio.open(output_path, "w", **tile_meta) as dst:
            for i, (band_name, data) in enumerate(tile_data.items()):
                dst.write(data, i + 1)
                dst.set_band_description(i + 1, band_name)
        
        logger.info(f"Saved compressed tile: {output_path}")

    def process_tile_pair(self, tile_id, paths):
        """Process a pair of pre/post-fire images and save as tiles."""
        if not paths['pre'] or not paths['post']:
            logger.warning(f"Missing pre or post image for tile {tile_id}")
            return

        try:
            with self._open_rasters(paths['pre'], paths['post']) as (pre_src, post_src):
                output_profile = post_src.profile.copy()
                output_profile.update({
                    'count': 16,
                    'dtype': 'float32',
                    'nodata': np.nan,
                })

                # Process the entire image in memory
                window = Window(0, 0, post_src.width, post_src.height)
                processed_data = self.process_chunk(pre_src, post_src, window)

                # Generate output path for compressed file
                post_filename = Path(paths['post']).stem
                output_file = self.output_dir / f"{post_filename}_train.tif"
                
                # Save the file with compression
                self.save_tile_with_compression(processed_data, output_profile, window, output_file)

                logger.info(f"Processed and saved {tile_id} with compression")

        except Exception as e:
            logger.error(f"Error processing tile pair {tile_id}: {str(e)}")
            raise

    def get_tile_pairs(self):
        """Get pairs of pre/post-fire images from input directory."""
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
        """Process all tile pairs in parallel."""
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
            chunk_size=1024,  # Process in 1024x1024 chunks
            tile_size=1024    # Output tile size
        )
        
        logger.info(f"Processing and tiling Sentinel-2 images in {root_dir}...")
        processor.process_all()
        logger.info("Processing completed successfully")
        
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
