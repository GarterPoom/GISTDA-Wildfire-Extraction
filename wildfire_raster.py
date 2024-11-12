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

    def chop_and_save_tiles(self, raster_data, meta, original_filename, tile_output_dir):
        """
        Chop processed raster into tiles and save them, grouping into 10 areas with complete tiles.

        Args:
            raster_data (dict): Dictionary containing band data
            meta (dict): Raster metadata
            original_filename (str): Original filename for naming tiles
            tile_output_dir (Path): Directory to save tiles
        """
        try:
            height = meta['height']
            width = meta['width']

            # Calculate number of tiles (original size)
            tile_width = (width + self.tile_size - 1) // self.tile_size
            tile_height = (height + self.tile_size - 1) // self.tile_size

            # Calculate the total number of tiles
            total_tiles = tile_width * tile_height

            # We want to limit the number of folders to 10, calculate the number of tiles per area
            max_areas = 10
            tiles_per_area = total_tiles // max_areas  # How many tiles per area

            # Determine how many tiles should be in each folder, trying to balance rows and columns
            area_rows = int(np.ceil(np.sqrt(tiles_per_area)))
            area_cols = int(np.ceil(tiles_per_area / area_rows))

            # Group tiles into fewer areas (max 10 areas)
            for area_row in range(max_areas):
                for area_col in range(max_areas):
                    # Determine the window bounds for this area
                    row_start = area_row * self.tile_size * area_rows
                    row_end = min((area_row + 1) * self.tile_size * area_rows, height)

                    col_start = area_col * self.tile_size * area_cols
                    col_end = min((area_col + 1) * self.tile_size * area_cols, width)

                    # Create the window for this area
                    window = Window(
                        col_start,
                        row_start,
                        col_end - col_start,
                        row_end - row_start
                    )

                    # Slice the data from the raster
                    tile_data = {
                        band_name: band_data[
                            window.row_off:window.row_off + window.height,
                            window.col_off:window.col_off + window.width
                        ]
                        for band_name, band_data in raster_data.items()
                    }

                    if any(data.any() for data in tile_data.values()):
                        # Grouping tiles into an area folder (AreaRowCol)
                        area_folder = f"Area{area_row}_{area_col}"
                        area_folder_path = tile_output_dir / area_folder
                        area_folder_path.mkdir(parents=True, exist_ok=True)

                        # Save the grouped tiles for this area
                        self._save_tile(tile_data, meta, window, 
                                    original_filename, area_folder_path)

        except Exception as e:
            logger.error(f"Failed to save tiles for {original_filename}: {e}")
            raise

    def _save_single_tile(self, raster_data, meta, original_filename, output_dir):
        """Save entire raster as a single tile."""
        tile_meta = meta.copy()
        tile_name = f"{original_filename}.tif"
        tile_path = output_dir / tile_name
        
        with rasterio.open(tile_path, "w", **tile_meta) as dst:
            for i, (band_name, data) in enumerate(raster_data.items()):
                dst.write(data, i + 1)
                dst.set_band_description(i + 1, band_name)
        
        logger.info(f"Saved complete image as single tile: {tile_path}")

    def _save_tile(self, tile_data, meta, window, original_filename, output_dir):
        """Save individual tile data."""
        tile_meta = meta.copy()
        tile_meta.update({
            "height": window.height,
            "width": window.width,
            "transform": rasterio.transform.from_bounds(
                *self._get_tile_bounds(meta['transform'], window),
                window.width,
                window.height
            )
        })
        
        tile_name = f"{original_filename}_tile_{window.row_off}_{window.col_off}.tif"
        tile_path = output_dir / tile_name
        
        with rasterio.open(tile_path, "w", **tile_meta) as dst:
            for i, (band_name, data) in enumerate(tile_data.items()):
                dst.write(data, i + 1)
                dst.set_band_description(i + 1, band_name)
        
        logger.info(f"Saved tile: {tile_path}")

    @staticmethod
    def _get_tile_bounds(transform, window):
        """Calculate bounds for a tile."""
        left = transform[2] + window.col_off * transform[0]
        top = transform[5] + window.row_off * transform[4]
        right = left + window.width * transform[0]
        bottom = top + window.height * transform[4]
        return left, bottom, right, top

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

                # Generate output directory for tiles
                post_filename = Path(paths['post']).stem
                tile_date = post_filename.split('_')[1]
                tile_output_dir = self.output_dir / f"{tile_id}_{tile_date}"
                tile_output_dir.mkdir(parents=True, exist_ok=True)

                # Chop and save tiles
                self.chop_and_save_tiles(
                    processed_data,
                    output_profile,
                    f"{tile_id}_{tile_date}",
                    tile_output_dir
                )

                logger.info(f"Processed and tiled {tile_id}")

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

    def process_all(self, max_workers=None):
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