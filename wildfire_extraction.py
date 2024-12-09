import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from rasterio.windows import Window
import logging
from contextlib import contextmanager
import math
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentinelProcessor:
    def __init__(self, root_dir, chunk_size, tile_size):
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
            1, 0)

        burn_label[~np.isfinite(dnbr) | ~np.isfinite(ndwi) | 
                  ~np.isfinite(ndvi) | ~np.isfinite(b08)] = 0
        
        return burn_label

    def process_chunk(self, pre_src, post_src, window):
        """
        Process a single chunk of the image.
        """
        band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06',
                     'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
        
        pre_bands = {}
        post_bands = {}
        
        # Process and store output bands in desired order
        output_data = {}
        
        # First process the B01-B12 bands
        for band_name in band_names:
            idx = band_names.index(band_name) + 1
            pre_data = pre_src.read(idx, window=window, masked=True)
            post_data = post_src.read(idx, window=window, masked=True)
            
            pre_bands[band_name] = pre_data.filled(np.nan).astype(np.float32)
            post_bands[band_name] = post_data.filled(np.nan).astype(np.float32)
            
            # Store normalized post bands
            output_data[band_name] = post_bands[band_name]

        # Calculate indices
        dnbr, ndwi, ndvi = self.calculate_indices(pre_bands, post_bands)
        
        # Add normalized indices
        output_data['dNBR'] = dnbr
        output_data['NDVI'] = ndvi
        output_data['NDWI'] = ndwi
        
        # Add burn label
        output_data['Burn_Label'] = self.create_burn_label(
            dnbr, ndwi, ndvi, post_bands['B08']
        )
        
        return output_data

    def _save_chunk(self, chunk_data, meta, window, original_filename, output_dir):
        """
        Save processed chunk data with specified band order.
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
        
        # Define band order
        band_order = [
            'B01', 'B02', 'B03', 'B04', 'B05', 'B06',
            'B07', 'B08', 'B8A', 'B09', 'B11', 'B12',
            'dNBR', 'NDVI', 'NDWI', 'Burn_Label'
        ]
        
        with rasterio.open(chunk_path, 'w', **chunk_meta) as dst:
            for i, band_name in enumerate(band_order):
                dst.write(chunk_data[band_name], i + 1)
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
        """
        if not paths['pre'] or not paths['post']:
            logger.warning(f"Missing pre or post image for tile {tile_id}")
            return

        try:
            with rasterio.open(paths['post']) as post_src:
                # Calculate optimal chunk size
                img_size = max(post_src.width, post_src.height)
                chunk_size = self.get_optimal_chunk_size(img_size)
                
                # Set number of output bands to exactly 16 (12 original + 4 indices)
                num_output_bands = 16
                
                # Update output profile
                output_profile = post_src.profile.copy()
                output_profile.update({
                    'count': num_output_bands,
                    'dtype': 'float32',
                    'nodata': np.nan
                })

                # Generate output directory
                post_filename = Path(paths['post']).stem
                tile_date = post_filename.split('_')[1]
                tile_output_dir = self.output_dir / f"{tile_id}_{tile_date}"
                tile_output_dir.mkdir(parents=True, exist_ok=True)

                # Process image in chunks
                for y in range(0, post_src.height, chunk_size):
                    for x in range(0, post_src.width, chunk_size):
                        effective_width = min(chunk_size, post_src.width - x)
                        effective_height = min(chunk_size, post_src.height - y)
                        
                        window = Window(x, y, effective_width, effective_height)
                        
                        with rasterio.open(paths['pre']) as pre_src:
                            chunk_data = self.process_chunk(pre_src, post_src, window)
                        
                        chunk_output_dir = tile_output_dir / f"chunk_{y}_{x}"
                        chunk_output_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Save chunk with ordered band names
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

    def preview_processed_tile(self, tile_id, chunk_coords=None, sample_size=5):
            """
            Preview processed raster data (output GeoTIFF with calculated indices).
        
            Args:
                tile_id (str): Identifier for the processed tile
                chunk_coords (tuple): Optional (y, x) coordinates of specific chunk to preview
                sample_size (int): Number of pixels to show for each dimension
            
            Returns:
                pd.DataFrame: DataFrame containing the preview of processed data
            """
            # Find the processed tile directory
            processed_tiles = list(self.output_dir.glob(f"{tile_id}*"))
            if not processed_tiles:
                logger.error(f"No processed data found for tile {tile_id}")
                return None
            
            tile_dir = processed_tiles[0]
        
            # If chunk coordinates are not specified, get the first chunk
            if chunk_coords is None:
                chunks = list(tile_dir.rglob("*.tif"))
                if not chunks:
                    logger.error(f"No chunks found in {tile_dir}")
                    return None
                chunk_path = chunks[0]
            else:
                chunk_y, chunk_x = chunk_coords
                chunk_path = tile_dir / f"chunk_{chunk_y}_{chunk_x}" / f"{tile_dir.name}_chunk_{chunk_y}_{chunk_x}.tif"
                if not chunk_path.exists():
                    logger.error(f"Chunk not found at {chunk_path}")
                    return None

            # Read and preview the processed data
            with rasterio.open(chunk_path) as src:
                # Get band descriptions (including calculated indices)
                band_descriptions = [src.descriptions[i-1] or f'Band_{i}' for i in range(1, src.count + 1)]
            
                # Read a small window of data
                window_data = {}
                for i in range(src.count):
                    band_data = src.read(i + 1, window=Window(0, 0, sample_size, sample_size))
                    band_name = band_descriptions[i]
                    window_data[band_name] = band_data.flatten()
            
                # Create DataFrame
                df = pd.DataFrame(window_data)
            
                # Add pixel coordinates
                y_coords, x_coords = np.meshgrid(range(sample_size), range(sample_size))
                df['y_coord'] = y_coords.flatten()
                df['x_coord'] = x_coords.flatten()
            
                # Reorder columns to show coordinates first, then indices, then bands
                index_cols = ['dNBR', 'NDVI', 'NDWI', 'Burn_Label']
                band_cols = [col for col in df.columns if col not in ['y_coord', 'x_coord'] + index_cols]
                cols = ['y_coord', 'x_coord'] + index_cols + band_cols
                df = df[cols]
            
                # Add basic metadata
                df.attrs['crs'] = src.crs.to_string()
                df.attrs['transform'] = src.transform
                df.attrs['resolution'] = src.res
            
                # Add basic statistics for indices
                stats = {}
                for index_name in index_cols:
                    if index_name in df.columns:
                        valid_data = df[index_name].dropna()
                        stats[index_name] = {
                            'mean': valid_data.mean(),
                            'std': valid_data.std(),
                            'min': valid_data.min(),
                            'max': valid_data.max()
                        }
                df.attrs['index_stats'] = stats
            
                return df
            
    def move_burn_priority_files(self, destination_dir, max_size_gb):
        """
        Randomly selects TIFF files with priority given to those with the highest burn area, 
        and moves them to the specified destination directory.
    
        Args:
            destination_dir (str): Path to the destination directory.
            max_size_gb (int): Maximum total size of selected files in gigabytes (default: 5).
        """
        destination_dir = Path(destination_dir).resolve()
        destination_dir.mkdir(parents=True, exist_ok=True)
    
        tiff_files = list(self.output_dir.rglob("*.tif"))
        if not tiff_files:
            logger.warning("No TIFF files found in output directory.")
            return
    
        # Calculate burn area for each file
        file_burn_areas = []
        for file_path in tiff_files:
            try:
                with rasterio.open(file_path) as src:
                    burn_label_band = src.read(src.count)  # Burn_Label is the last band
                    burn_area = np.sum(burn_label_band == 1)  # Count burn pixels
                    file_size = file_path.stat().st_size
                    file_burn_areas.append((file_path, burn_area, file_size))
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {str(e)}")
    
        # Sort files by burn area (descending)
        file_burn_areas.sort(key=lambda x: x[1], reverse=True)
    
        # Select files until size limit is reached
        selected_files = []
        total_size = 0
        max_size_bytes = max_size_gb * 1024 * 1024 * 1024  # Convert GB to bytes
    
        for file_path, burn_area, file_size in file_burn_areas:
            if total_size + file_size > max_size_bytes:
                break
            selected_files.append(file_path)
            total_size += file_size
    
        logger.info(f"Selected {len(selected_files)} files with total size {total_size / (1024**3):.2f} GB")
    
        # Move files to the destination directory
        for file_path in selected_files:
            destination_path = destination_dir / file_path.name
            shutil.move(str(file_path), str(destination_path))
            logger.info(f"Moved file: {file_path} -> {destination_path}")
    
        logger.info(f"All selected files moved to {destination_dir}")

def main():
    """Main entry point for the script."""
    try:
        root_dir = Path("Raster").resolve()
        train_dir = Path("Raster_Train").resolve()
        
        processor = SentinelProcessor(
            root_dir=root_dir,
            chunk_size=1024, # Adjust following your hardware specification
            tile_size=1024 # Adjust following your hardware specification
        )

        # First, process all tiles
        logger.info(f"Processing Sentinel-2 images in {root_dir}...")
        processor.process_all(max_workers=2)
        logger.info("Processing completed successfully")

        # Randomly select files with burn priority and move to Raster_Train
        logger.info("Selecting and moving TIFF files by burn priority to Raster_Train...")
        processor.move_burn_priority_files(train_dir, max_size_gb=3)
        logger.info("File selection and movement completed.")

        # Now check for processed tiles
        processed_tiles = [d.name.split('_')[0] for d in processor.output_dir.iterdir() if d.is_dir()]
        
        if not processed_tiles:
            logger.warning("No processed tiles found in output directory")
            return
            
        # Preview the first processed tile
        first_tile = processed_tiles[0]
        logger.info(f"\nPreviewing processed data for tile: {first_tile}")
        
        df = processor.preview_processed_tile(first_tile)
        if df is not None:
            logger.info("\nProcessed data preview:")
            logger.info(df.head())
            
            logger.info("\nIndex Statistics:")
            for index_name, stats in df.attrs['index_stats'].items():
                logger.info(f"\n{index_name}:")
                for stat_name, value in stats.items():
                    logger.info(f"{stat_name}: {value:.4f}")
        else:
            logger.error("Failed to preview processed tile")
            
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()