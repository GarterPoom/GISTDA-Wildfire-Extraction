import os
import sys
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window
from rasterio.merge import merge
from rasterio.io import MemoryFile
from pathlib import Path
import shutil
import time
import logging

def setup_logging():
    """
    Set up logging to help diagnose issues.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('sentinel_processing.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def resample_image(input_path, output_path, target_resolution=10):
    """
    Resamples a single image to a target resolution using Rasterio and saves it as a compressed GeoTIFF file.
    """
    try:
        print(f"Resampling image: {input_path} to {output_path} at {target_resolution}m resolution.")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with rasterio.open(input_path) as src:
            # Original transform and resolution
            transform = src.transform
            input_res = transform.a

            # Calculate rescaling factor
            scale = input_res / target_resolution

            # New dimensions
            dst_height = int(src.height * scale)
            dst_width = int(src.width * scale)

            # New transform
            new_transform = Affine(transform.a / scale, transform.b, transform.c,
                                   transform.d, transform.e / scale, transform.f)

            # Set up destination metadata
            kwargs = src.meta.copy()
            kwargs.update({
                'height': dst_height, # New height
                'width': dst_width, # New width 
                'transform': new_transform, # New transform
                'compress': 'lzw', # Compression
                'tiled': True, # Tiling
                'blockxsize': 256, # Tile size
                'blockysize': 256, # Tile size
                'bigtiff': 'yes' # BigTIFF
            })

            # Read and resample data
            data = src.read(
                out_shape=(src.count, dst_height, dst_width),
                resampling=Resampling.nearest # Nearest neighbor resampling
            )

            # Write resampled data
            with rasterio.open(output_path, 'w', **kwargs) as dst:
                dst.write(data)

        print(f"Resampling completed with compression: {output_path}")
        return True

    except Exception as e:
        print(f"Error resampling image {input_path}: {str(e)}")
        return False

def safe_remove(file_path, max_attempts=5, delay=1):
    """
    Safely remove a file with multiple attempts and delay between attempts.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return True
    
    for attempt in range(max_attempts):
        try:
            file_path.unlink()
            return True
        except PermissionError:
            if attempt < max_attempts - 1:
                time.sleep(delay)
            continue
        except Exception as e:
            print(f"Error removing file {file_path}: {str(e)}")
            return False
    return False

def build_pyramids_nearest(raster_path, overview_levels=[2, 4, 8, 16, 32]):

    """
    Build raster pyramids (overviews) for Faster Display Raster data with GIS Software 
    using specified resampling algorithm with Rasterio.
    
    Args:
        raster_path (str): Path to the GeoTIFF file
        overview_levels (list): List of overview levels to build
        resample_alg (str): Resampling algorithm ('NEAREST', 'AVERAGE', etc.)
    """

    try:
        print(f"Building pyramids (nearest) for: {raster_path}")

        with rasterio.open(raster_path, 'r+') as dataset:
            dataset.build_overviews(overview_levels, resampling=Resampling.nearest) # Build pyramids using NEAREST resampling
            dataset.update_tags(ns='rio_overview', resampling='NEAREST')

        print(f"Successfully built pyramids: {overview_levels} using NEAREST")
        return True

    except Exception as e:
        print(f"Error building pyramids for {raster_path}: {str(e)}")
        return False

def process_bands(input_folder, output_folder, scl_output_folder=None):
    """
    Processes Sentinel-2 band files in a given input folder, resampling them to a target resolution
    and saving the output as compressed GeoTIFF files. Optionally processes Scene Classification Layer (SCL) files
    if provided.

    Args:
        input_folder (str or Path): Path to the input folder containing .jp2 files.
        output_folder (str or Path): Path to the output folder where processed files will be saved.
        scl_output_folder (str or Path, optional): Path to the folder for saving resampled SCL files, if any.

    Raises:
        ValueError: If no valid band files are processed.
        Exception: If an error occurs during processing, such as file I/O errors.

    Returns:
        None
    """

    try:
        print(f"Processing bands in folder: {input_folder}")

        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        if scl_output_folder:
            scl_output_folder = Path(scl_output_folder)
            scl_output_folder.mkdir(parents=True, exist_ok=True)

        jp2_files = list(input_folder.glob('*.jp2'))
        if not jp2_files:
            print("No JP2 files found in the input folder.")
            return

        temp_folder = output_folder / 'temp'
        temp_folder.mkdir(parents=True, exist_ok=True)

        resampled_files = []
        band_paths = {}
        scl_file = None

        for jp2_file in jp2_files:
            output_path = temp_folder / f"{jp2_file.stem}_resampled.tif"

            if resample_image(str(jp2_file), str(output_path)):
                resampled_files.append(str(output_path))

                band_map = {
                    'B02': 'B02', 'B03': 'B03', 'B04': 'B04', 'B05': 'B05', 'B06': 'B06', 'B07': 'B07', 
                    'B08': 'B08', 'B8A': 'B8A', 'B11': 'B11', 'B12': 'B12'
                }

                if 'SCL' in jp2_file.name:
                    scl_file = str(output_path)
                    continue

                for band_key in band_map:
                    if band_key in jp2_file.name:
                        band_paths[band_key] = str(output_path)
                        break

<<<<<<< HEAD
        ordered_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
        final_resampled_files = [band_paths[band] for band in ordered_bands if band in band_paths]
=======
        # Process regular bands
        ordered_bands = ['B12', 'B8A', 'B04', 'B02', 'B03', 'B05', 'B06', 'B07', 'B08', 'B11']
        final_resampled_files = [band_paths[band] for band in ordered_bands 
                               if band in band_paths]
>>>>>>> 8b41eb8704ca4c663bbc33629f9771f9a6672e85

        if not final_resampled_files:
            raise ValueError("No valid band files were processed")

        sample_filename = jp2_files[0].name
        parts = sample_filename.split('_')
        tile_date_timestamp = f"{parts[0]}_{parts[1]}"
        output_filename = f"{tile_date_timestamp}.tif"
        output_path = output_folder / output_filename

        print(f"Creating compressed output file: {output_path}")

        srcs = [rasterio.open(f) for f in final_resampled_files]
        meta = srcs[0].meta.copy()
        meta.update({
            'count': len(final_resampled_files),
            'compress': 'lzw',
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
            'bigtiff': 'yes'
        })

        with rasterio.open(output_path, 'w', **meta) as dst:
            for idx, src in enumerate(srcs, start=1):
                dst.write(src.read(1), idx)
                dst.set_band_description(idx, ordered_bands[idx - 1])

        for src in srcs:
            src.close()

        build_pyramids_nearest(str(output_path))

        if scl_file and scl_output_folder:
            scl_output_filename = f"{tile_date_timestamp}_SCL.tif"
            scl_output_path = scl_output_folder / scl_output_filename
            shutil.copy(scl_file, scl_output_path)
            print(f"Exported SCL file: {scl_output_path}")

        print("Cleaning up temporary files.")
        for file in resampled_files:
            os.remove(file)
        if temp_folder.exists():
            temp_folder.rmdir()

    except Exception as e:
        print(f"Error processing bands: {str(e)}")
        raise

def find_and_process_folders(root_folder, output_folder):
    """
    Searches for and processes folders containing .jp2 files.
    """
    try:
        root_folder = Path(root_folder)
        output_folder = Path(output_folder)
        
        print(f"Searching for folders in: {root_folder}")
        
        for dirpath in root_folder.rglob('*'):
            if dirpath.is_dir() and any(f.suffix == '.jp2' for f in dirpath.iterdir()):
                relative_path = dirpath.relative_to(root_folder)
                current_output_folder = output_folder / relative_path
                print(f"Found JP2 files in: {dirpath}. Processing...")
                process_bands(dirpath, current_output_folder)
        
        print("All folders processed.")
        
    except Exception as e:
        print(f"Error processing folders: {str(e)}")
        sys.exit(1)

def main():
    # Configure logging
    logger = setup_logging()

    try:
        # Get current working directory
        current_dir = Path.cwd()
        logger.info(f"Current working directory: {current_dir}")

        # Check input folders
        root_folder = current_dir / 'Classified_Image'
        output_folder = current_dir / 'Raster_Classified'
        scl_output_folder = current_dir / 'SCL_Classified'

        # Check if input folder exists
        if not root_folder.exists():
            logger.error(f"Input folder does not exist: {root_folder}")
            logger.info("Please create a 'Classified_Image' folder and place Sentinel-2 JP2 files inside.")
            sys.exit(1)

        # Create output folders if they don't exist
        output_folder.mkdir(parents=True, exist_ok=True)
        scl_output_folder.mkdir(parents=True, exist_ok=True)

        # Find and process JP2 files
        jp2_files = list(root_folder.rglob('*.jp2'))
        
        if not jp2_files:
            logger.error(f"No JP2 files found in {root_folder}")
            logger.info("Ensure Sentinel-2 JP2 files are present in the 'Classified_Image' folder.")
            sys.exit(1)

        logger.info(f"Found {len(jp2_files)} JP2 files to process")

        def find_and_process_folders(root_folder, output_folder, scl_output_folder):
            try:
                root_folder = Path(root_folder)
                output_folder = Path(output_folder)
                scl_output_folder = Path(scl_output_folder)
                
                logger.info(f"Searching for folders in: {root_folder}")
                
                processed_folders = 0
                for dirpath in root_folder.rglob('*'):
                    if dirpath.is_dir() and any(f.suffix == '.jp2' for f in dirpath.iterdir()):
                        relative_path = dirpath.relative_to(root_folder)
                        current_output_folder = output_folder / relative_path
                        current_scl_output_folder = scl_output_folder / relative_path
                        logger.info(f"Found JP2 files in: {dirpath}. Processing...")
                        process_bands(dirpath, current_output_folder, current_scl_output_folder)
                        processed_folders += 1
                
                if processed_folders == 0:
                    logger.warning("No folders with JP2 files were processed.")
                else:
                    logger.info(f"Processed {processed_folders} folders.")
                
            except Exception as e:
                logger.error(f"Error processing folders: {str(e)}")
                sys.exit(1)
        
        # Run processing
        find_and_process_folders(root_folder, output_folder, scl_output_folder)
        
        logger.info("Processing complete.")

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()