import os
from osgeo import gdal
import sys
from pathlib import Path
import time

def resample_image(input_path, output_path, target_resolution=10):
    """
    Resamples a single image to a target resolution using GDAL and saves it as a compressed GeoTIFF file.
    """
    try:
        print(f"Resampling image: {input_path} to {output_path} at {target_resolution}m resolution.")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Open the input dataset
        src_ds = gdal.Open(input_path)
        if not src_ds:
            raise ValueError(f"Could not open {input_path}")
            
        # Get the input resolution
        gt = src_ds.GetGeoTransform()
        input_res = gt[1]  # pixel width
        
        # Calculate new dimensions
        src_xsize = src_ds.RasterXSize
        src_ysize = src_ds.RasterYSize
        dst_xsize = int(src_xsize * (input_res / target_resolution))
        dst_ysize = int(src_ysize * (input_res / target_resolution))

        # Create translation options
        translate_options = gdal.TranslateOptions(
            format='GTiff',
            width=dst_xsize,
            height=dst_ysize,
            resampleAlg=gdal.GRA_NearestNeighbour,
            creationOptions=[
                'COMPRESS=LZW',
                'PREDICTOR=2',
                'TILED=YES',
                'BLOCKXSIZE=256',
                'BLOCKYSIZE=256',
                'BIGTIFF=YES'
            ]
        )
        
        # Perform resampling with compression
        gdal.Translate(
            destName=output_path,
            srcDS=src_ds,
            options=translate_options
        )
        
        # Close the dataset
        src_ds = None
        
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

def process_bands(input_folder, output_folder):
    """
    Processes Sentinel-2 band files in a given input folder with GDAL compression.
    """
    temp_folder = None
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
        output_filename = f"{tile_date_timestamp}_pre.tif"
        output_path = output_folder / output_filename

        print(f"Creating compressed output file: {output_path}")

        # Create VRT with options
        vrt_options = gdal.BuildVRTOptions(separate=True)
        vrt_path = str(temp_folder / 'temp.vrt')
        vrt_ds = gdal.BuildVRT(vrt_path, final_resampled_files, options=vrt_options)
        
        # Create final output with compression
        translate_options = gdal.TranslateOptions(
            format='GTiff',
            creationOptions=[
                'COMPRESS=LZW',
                'PREDICTOR=2',
                'TILED=YES',
                'BLOCKXSIZE=256',
                'BLOCKYSIZE=256',
                'BIGTIFF=YES'
            ]
        )
        
        gdal.Translate(
            destName=str(output_path),
            srcDS=vrt_ds,
            options=translate_options
        )
        
        # Close VRT dataset
        vrt_ds = None
        
        if not output_path.exists():
            raise ValueError(f"Failed to create output file: {output_path}")
        
        # Clean up temporary files
        print("Cleaning up temporary files.")
        for file in resampled_files:
            safe_remove(file)
        safe_remove(vrt_path)
        
        if temp_folder and temp_folder.exists():
            try:
                temp_folder.rmdir()
            except Exception as e:
                print(f"Warning: Could not remove temp folder: {str(e)}")

    except Exception as e:
        print(f"Error processing bands: {str(e)}")
        raise
    finally:
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
    # Enable GDAL exceptions
    gdal.UseExceptions()
    
    root_folder = Path('Pre-Image')
    output_folder = Path('Raster/input')
    output_folder.mkdir(parents=True, exist_ok=True)
    find_and_process_folders(root_folder, output_folder)