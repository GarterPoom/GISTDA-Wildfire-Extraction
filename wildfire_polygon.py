import os
import re
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
import rasterio as rio
import numpy as np
from IPython.display import display
import pandas as pd

def find_all_files_in_root(root_folder, file_extension='.tif'):
    """
    Recursively search for all files with the specified extension in the root folder and its subfolders.

    Parameters:
    - root_folder: str, the root folder path to search
    - file_extension: str, the file extension to search for (default is '.tif')

    Returns:
    - list: a list of full paths to the found files, or an empty list if none found
    """
    file_paths = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(file_extension):
                file_paths.append(os.path.join(dirpath, filename))
    return file_paths

def extract_fire_date_from_filename(filename):
    """
    Extract the fire date from the given TIFF file name and format it as YYYY-MM-DD.

    Parameters:
    - filename: str, the name of the TIFF file

    Returns:
    - str: extracted fire date in YYYY-MM-DD format, or None if no match
    """
    match = re.search(r'_(\d{8})T', filename)
    if match:
        date_str = match.group(1)
        # Convert YYYYMMDD to YYYY-MM-DD
        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        return formatted_date
    return None

def create_polygon_shapefile_from_burnt_areas(tif_file_path, output_folder):
    """
    Create a polygon shapefile from the GeoTIFF where Burn_Classified == 1.

    Parameters:
    - tif_file_path: str, the path to the classified GeoTIFF
    - output_folder: str, the folder where the output shapefile will be saved

    This function generates polygons where the pixel value is 1 (Burn_Classified) in the GeoTIFF.
    """
    # Extract filename without extension for output
    base_name = os.path.splitext(os.path.basename(tif_file_path))[0]

    # Dynamically generate output shapefile name based on TIFF filename
    output_shapefile_name = f"{base_name}_Burn_classified.shp"
    output_shapefile_path = os.path.join(output_folder, output_shapefile_name)

    # Open the classified GeoTIFF and read data
    with rio.open(tif_file_path) as src:
        burn_classified = src.read(1)
        transform = src.transform
        crs = src.crs

    # Generate shapes (polygons) for the burnt areas
    mask = burn_classified == 1  # Burnt areas have a value of 1
    burnt_shapes = shapes(burn_classified, mask=mask, transform=transform)

    # Convert shapes into a GeoDataFrame
    geoms = []
    for geom, value in burnt_shapes:
        if value == 1:
            geoms.append(shape(geom))

    # Create a GeoDataFrame from the geometries
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)

    # Convert CRS to Latitude and Longitude
    gdf = gdf.to_crs(epsg=4326)

    # Extract FIRE_DATE from TIFF file name
    fire_date = extract_fire_date_from_filename(os.path.basename(tif_file_path))
    gdf['FIRE_DATE'] = fire_date

    # Add LATITUDE and LONGITUDE columns as centroids of each polygon
    gdf['LATITUDE'] = gdf.geometry.centroid.y
    gdf['LONGITUDE'] = gdf.geometry.centroid.x

    # Reproject to UTM CRS for area calculation (assumes EPSG:32647, adjust as needed)
    gdf = gdf.to_crs(epsg=32647)

    # Calculate the area of each polygon in square meters and add AREA column
    gdf['AREA'] = gdf.geometry.area

    # Move the geometry column to the last position
    cols = list(gdf.columns)
    cols.append(cols.pop(cols.index('geometry')))
    gdf = gdf[cols]

    # Display the GeoDataFrame
    display(gdf)

    # Save the GeoDataFrame to a shapefile in the specified output folder
    gdf.to_file(output_shapefile_path, driver='ESRI Shapefile')
    print(f"Polygon shapefile '{output_shapefile_path}' has been created.")

# Example usage
if __name__ == "__main__":
    root_folder = r'Classified Output'  # Path to the root folder where the TIFF is located
    output_folder = r'Wildfire Polygon'  # Path to the folder where the shapefile will be saved

    # Find all TIFF files in the root folder and subfolders
    tif_files = find_all_files_in_root(root_folder)

    # Process each found TIFF file
    for tif_file_path in tif_files:
        print(f"\nProcessing file: {tif_file_path}")
        create_polygon_shapefile_from_burnt_areas(tif_file_path, output_folder)