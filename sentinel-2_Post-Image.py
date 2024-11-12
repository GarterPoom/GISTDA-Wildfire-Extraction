import os
import rasterio
from rasterio.enums import Resampling
from osgeo import gdal
import sys
from pathlib import Path
import time

def resample_image(input_path, output_path, target_resolution=10):
    """
    Resamples a single image to a target resolution and saves it as an 8-bit GeoTIFF file.
    """
    try:
        print(f"Resampling image: {input_path} to {output_path} at {target_resolution}m resolution.")
        
        with rasterio.open(input_path) as src:
            scale_factor = src.res[0] / target_resolution
            
            data = src.read(
                out_shape=(
                    src.count,
                    int(src.height * scale_factor),
                    int(src.width * scale_factor)
                ),
                resampling=Resampling.nearest
            )
            
            transform = src.transform * src.transform.scale(
                (src.width / data.shape[-1]),
                (src.height / data.shape[-2])
            )
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Use rasterio's context manager for proper file handling
            profile = src.profile.copy()
            profile.update({
                'driver': 'GTiff',
                'height': data.shape[1],
                'width': data.shape[2],
                'count': src.count,
                'dtype': 'uint32',
                'transform': transform
            })

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data)
        
        print(f"Resampling completed: {output_path}")
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

def process_bands(input_folder, output_folder):
    """
    Processes Sentinel-2 band files in a given input folder.
    """
    temp_folder = None
    vrt = None
    try:
        print(f"Processing bands in folder: {input_folder}")
        
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        jp2_files = list(input_folder.glob('*.jp2'))
        if not jp2_files:
            print("No JP2 files found in the input folder.")
            return

        temp_folder = output_folder / 'temp'
        temp_folder.mkdir(parents=True, exist_ok=True)

        resampled_files = []
        band_paths = {}

        for jp2_file in jp2_files:
            output_path = temp_folder / f"{jp2_file.stem}_resampled.tif"
            
            if resample_image(str(jp2_file), str(output_path)):
                resampled_files.append(str(output_path))
                
                # Map band to file path using a simplified approach
                band_map = {
                    'B01': 'B01', 'B02': 'B02', 'B03': 'B03', 'B04': 'B04',
                    'B05': 'B05', 'B06': 'B06', 'B07': 'B07', 'B08': 'B08',
                    'B8A': 'B8A', 'B09': 'B09', 'B11': 'B11', 'B12': 'B12'
                }
                
                for band_key in band_map:
                    if band_key in jp2_file.name:
                        band_paths[band_key] = str(output_path)
                        break

        ordered_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 
                        'B08', 'B8A', 'B09', 'B11', 'B12']
        final_resampled_files = [band_paths[band] for band in ordered_bands 
                               if band in band_paths]

        if not final_resampled_files:
            raise ValueError("No valid band files were processed")

        sample_filename = jp2_files[0].name
        parts = sample_filename.split('_')
        tile_date_timestamp = f"{parts[0]}_{parts[1]}"
        output_filename = f"{tile_date_timestamp}_post.tif"  # Changed to _post.tif
        output_path = output_folder / output_filename

        print(f"Creating output file: {output_path}")

        # Create VRT and ensure it's properly closed
        vrt = gdal.BuildVRT('', final_resampled_files, separate=True)
        gdal.Translate(str(output_path), vrt)
        vrt = None  # Explicitly close VRT
        
        # Verify output file creation
        if not output_path.exists():
            raise ValueError(f"Failed to create output file: {output_path}")
        
        # Clean up with delay
        print("Cleaning up temporary files.")
        for file in resampled_files:
            safe_remove(file)
        
        # Attempt to remove temp folder
        if temp_folder and temp_folder.exists():
            try:
                temp_folder.rmdir()
            except Exception as e:
                print(f"Warning: Could not remove temp folder: {str(e)}")

    except Exception as e:
        print(f"Error processing bands: {str(e)}")
        raise
    finally:
        # Ensure VRT is closed
        if vrt is not None:
            vrt = None
        
        # Final cleanup attempt
        if temp_folder and temp_folder.exists():
            try:
                for file in temp_folder.glob('*'):
                    safe_remove(file)
                temp_folder.rmdir()
            except Exception as e:
                print(f"Warning: Failed final cleanup: {str(e)}")

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

if __name__ == "__main__":
    root_folder = Path('Post-Image')  # Changed to Post-Image
    output_folder = Path('Raster/input')
    output_folder.mkdir(parents=True, exist_ok=True)
    find_and_process_folders(root_folder, output_folder)