import os
import re
import shutil  # To move files to the new folder
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
import rasterio as rio
import numpy as np
from IPython.display import display
import pandas as pd
import reverse_geocoder as rg  # Reverse geocoding library

def find_all_files_in_root(root_folder, file_extension='.tif'):
    """
    Finds all files in a root folder with a given file extension.

    Parameters
    ----------
    root_folder : str
        The root folder to search in.
    file_extension : str, optional
        The file extension to search for. Defaults to '.tif'.

    Returns
    -------
    list
        A list of file paths (full paths) of all files found with the given extension.
    """
    file_paths = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(file_extension):
                file_paths.append(os.path.join(dirpath, filename))
    return file_paths

def create_output_folder_for_file(input_file_path, output_base_folder):
    """
    Creates an output folder for a given input file path within a specified base folder.

    Parameters
    ----------
    input_file_path : str
        The file path of the input file for which the output folder is to be created.
    output_base_folder : str
        The base folder where the output folder will be created.

    Returns
    -------
    str
        The path to the created output folder.
    """

    base_name = os.path.splitext(os.path.basename(input_file_path))[0]
    output_folder = os.path.join(output_base_folder, base_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return output_folder

def extract_fire_date_from_filename(filename):
    """
    Extracts the fire date from a given filename in the format 'YYYYMMDDT<rest of filename>'
    and returns it as a string in the format 'YYYY-MM-DD'. If no fire date is found, returns None.

    Parameters
    ----------
    filename : str
        The filename to extract the fire date from.

    Returns
    -------
    str
        The fire date as a string in the format 'YYYY-MM-DD', or None if no fire date is found.
    """
    match = re.search(r'_(\d{8})T', filename)
    if match:
        date_str = match.group(1)
        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        return formatted_date
    return None

def create_polygon_shapefile_from_burnt_areas(tif_file_path, output_folder, admin_shp_path):
    """
    Creates a polygon shapefile from a given GeoTIFF file containing classified burnt areas.
    """
    base_name = os.path.splitext(os.path.basename(tif_file_path))[0]
    output_shapefile_name = f"{base_name}_Burn_classified.shp"
    output_shapefile_path = os.path.join(output_folder, output_shapefile_name)

    # Load GeoTIFF file
    with rio.open(tif_file_path) as src:
        burn_classified = src.read(1)
        transform = src.transform
        crs = src.crs
    
    # Create shapes from raster
    mask = burn_classified == 1
    burnt_shapes = shapes(burn_classified, mask=mask, transform=transform)
    
    # Convert to geometries
    geoms = [shape(geom) for geom, value in burnt_shapes if value == 1]
    
    # If no geometries found, create empty GeoDataFrame with correct structure
    if not geoms:
        print(f"Warning: No burnt areas (value=1) found in {tif_file_path}")
        # Create empty GeoDataFrame with correct columns
        gdf = gpd.GeoDataFrame(columns=['FIRE_DATE', 'LATITUDE', 'LONGITUDE', 'AREA', 
                                      'TB_TN', 'TB_EN', 'AP_TN', 'AP_EN', 'PV_TN', 
                                      'PV_EN', 'COUNTRY', 'ISO3'], 
                              geometry=[], crs=crs)
    else:
        # Create GeoDataFrame with the found geometries
        gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
        
        # Calculate area in original projection (UTM)
        gdf['AREA'] = gdf.geometry.area
        
        # Convert to WGS84 for lat/long calculations
        gdf_wgs84 = gdf.to_crs(epsg=4326)
        gdf['LATITUDE'] = gdf_wgs84.geometry.centroid.y
        gdf['LONGITUDE'] = gdf_wgs84.geometry.centroid.x
        
        # Add fire date
        gdf['FIRE_DATE'] = extract_fire_date_from_filename(os.path.basename(tif_file_path))
        
        # Load admin boundaries
        admin_gdf = gpd.read_file(admin_shp_path, encoding='TIS-620')
        
        # Convert admin boundaries to UTM for accurate intersection
        admin_gdf = admin_gdf.to_crs(gdf.crs)
        
        # Perform spatial intersection
        intersected_gdf = gpd.overlay(gdf, admin_gdf, how='intersection')
        
        # If intersection results in empty GeoDataFrame, keep original with empty admin columns
        if intersected_gdf.empty:
            print("Warning: Intersection resulted in empty GeoDataFrame. Check if burnt areas overlap with admin boundaries.")
            for col in admin_gdf.columns:
                if col != 'geometry':
                    gdf[col] = pd.NA
            intersected_gdf = gdf
        
        # Convert final result to WGS84 for storage
        intersected_gdf = intersected_gdf.to_crs(epsg=4326)
    
    # Reorder columns consistently
    final_columns = ['FIRE_DATE', 'LATITUDE', 'LONGITUDE', 'AREA', 
                    'TB_TN', 'TB_EN', 'AP_TN', 'AP_EN', 'PV_TN', 
                    'PV_EN', 'COUNTRY', 'ISO3', 'geometry']
    
    # Ensure all columns exist (fill with NA if missing)
    for col in final_columns:
        if col not in intersected_gdf.columns and col != 'geometry':
            intersected_gdf[col] = pd.NA
    
    # Reorder columns
    intersected_gdf = intersected_gdf[final_columns]
    
    # Save to file
    intersected_gdf.to_file(output_shapefile_path, driver='ESRI Shapefile')
    
    # Display results
    display(intersected_gdf)
    print(f"Polygon shapefile '{output_shapefile_path}' has been created.")
    
    return intersected_gdf

def main():
    """
    Main function to process GeoTIFF files and create polygon shapefiles for classified burnt areas.

    This function searches for all GeoTIFF files in the 'Classified_Output' folder, creates an output
    folder for each file in the 'Wildfire_Polygon' base folder, and generates polygon shapefiles representing
    the classified burnt areas. The shapefiles are created with information about fire date, location,
    area, and reverse-geocoded administrative details.

    The process includes:
    1. Finding all GeoTIFF files in the specified root folder.
    2. Creating an output folder for each GeoTIFF file.
    3. Generating a polygon shapefile from each GeoTIFF file's classified burnt areas.

    The polygon shapefiles are saved in the EPSG:4326 coordinate reference system.
    """

    root_folder = r'Classified_Output'
    output_base_folder = r'Wildfire_Polygon'
    admin_shp_path = r'Thailand_Administrative_Boundary\Thailand_Administrative_Boundary.shp'

    tif_files = find_all_files_in_root(root_folder)
    for tif_file_path in tif_files:
        print(f"\nProcessing file: {tif_file_path}")
        output_folder = create_output_folder_for_file(tif_file_path, output_base_folder)
        create_polygon_shapefile_from_burnt_areas(tif_file_path, output_folder, admin_shp_path)

if __name__ == "__main__":
    main()
