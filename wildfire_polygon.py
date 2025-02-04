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

def reverse_geocode(lat, lon):

    """
    Reverse geocodes a given latitude and longitude to get the administrative region
    (e.g. province/state) and country name. If the country code is found in the
    COUNTRY_CODES mapping, it will be replaced with the corresponding country name.

    Parameters
    ----------
    lat : float
        Latitude
    lon : float
        Longitude

    Returns
    -------
    tuple
        (name, admin1, country_name, country_code)

        name : str
            The name of the nearest administrative region
        admin1 : str
            The administrative level 1 name (e.g. province/state)
        country_name : str
            The country name
        country_code : str
            The country code (e.g. TH, LA, MM, etc.)
    """
    # Country code to country name mapping
    COUNTRY_CODES = {
        'TH': 'Thailand',
        'LA': "Lao People's Democratic Republic",
        'MM': 'Myanmar',
        'KH': 'Cambodia',
        'VN': 'Vietnam',
        # Add more if needed
    }

    results = rg.search((lat, lon), mode=1)
    if results:
        result = results[0]
        country_code = result.get('cc')
        country_name = COUNTRY_CODES.get(country_code, None)  # Map cc to country name
        return result.get('name'), result.get('admin1'), country_name, country_code
    return None, None, None, None

def create_polygon_shapefile_from_burnt_areas(tif_file_path, output_folder):
    """
    Creates a polygon shapefile from a given GeoTIFF file containing classified burnt areas.

    Parameters
    ----------
    tif_file_path : str
        The path to the GeoTIFF file containing classified burnt areas.
    output_folder : str
        The folder path where the output shapefile will be saved.

    Returns
    -------
    str
        The path to the created shapefile.

    Notes
    -----
    The shapefile will contain the following fields:

    - FIRE_DATE: The fire date in the format 'YYYY-MM-DD'.
    - LATITUDE: The latitude of the centroid of each polygon.
    - LONGITUDE: The longitude of the centroid of each polygon.
    - AREA: The area in square meters of each polygon.
    - AP_EN: The administrative level 1 name in English.
    - PV_EN: The province name in English.
    - COUNTRY: The country name in English.
    - ISO3: The ISO 3-letter country code.

    The shapefile will be written in the EPSG:4326 coordinate reference system.
    """
    base_name = os.path.splitext(os.path.basename(tif_file_path))[0]
    output_shapefile_name = f"{base_name}_Burn_classified.shp"
    output_shapefile_path = os.path.join(output_folder, output_shapefile_name)

    with rio.open(tif_file_path) as src:
        burn_classified = src.read(1)
        transform = src.transform
        crs = src.crs

    mask = burn_classified == 1
    burnt_shapes = shapes(burn_classified, mask=mask, transform=transform)

    geoms = [shape(geom) for geom, value in burnt_shapes if value == 1]
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)

    gdf = gdf.to_crs(epsg=4326)  # Convert to EPSG:4326 For get Latitude and Longitude
    gdf['FIRE_DATE'] = extract_fire_date_from_filename(os.path.basename(tif_file_path))
    gdf['LATITUDE'] = gdf.geometry.centroid.y
    gdf['LONGITUDE'] = gdf.geometry.centroid.x

    # Calculate area
    gdf = gdf.to_crs(crs)  # Convert back to original CRS
    gdf['AREA'] = gdf.geometry.area  # Calculate area after projection

    # Reverse Geocoding
    ap_en_list, pv_en_list, country_list, iso3_list = [], [], [], []
    for lat, lon in zip(gdf['LATITUDE'], gdf['LONGITUDE']):
        ap_en, pv_en, country, iso3 = reverse_geocode(lat, lon)
        ap_en_list.append(ap_en)
        pv_en_list.append(pv_en)
        country_list.append(country)
        iso3_list.append(iso3)

    gdf['AP_EN'] = ap_en_list
    gdf['PV_EN'] = pv_en_list
    gdf['COUNTRY'] = country_list
    gdf['ISO3'] = iso3_list

    cols = list(gdf.columns)
    cols.append(cols.pop(cols.index('geometry')))
    gdf = gdf[cols]

    display(gdf)
    gdf.to_file(output_shapefile_path, driver='ESRI Shapefile')
    print(f"Polygon shapefile '{output_shapefile_path}' has been created.")

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

    tif_files = find_all_files_in_root(root_folder)
    for tif_file_path in tif_files:
        print(f"\nProcessing file: {tif_file_path}")
        output_folder = create_output_folder_for_file(tif_file_path, output_base_folder)
        create_polygon_shapefile_from_burnt_areas(tif_file_path, output_folder)

if __name__ == "__main__":
    main()
