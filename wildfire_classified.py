# Library Package
import os
import re
import rasterio as rio
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import shape
from datetime import datetime
import logging
from rasterio.features import shapes
from scipy.ndimage import label
from skimage.transform import resize
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from lightgbm import LGBMClassifier
from IPython.display import display, Markdown
import pickle

pd.set_option("display.max_columns", None) # To show all columns in a pandas DataFrame

# Set up logging configuration for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to recursively search for TIFF files
def find_tif_files(root_folder):
    """
    Recursively search for all files with the extension '.tif' in the root folder and its subfolders.

    Parameters:
    - root_folder: str, the root folder path to search

    Returns:
    - list: a list of full paths to the found TIFF files, or an empty list if none found
    """
    tif_files = []
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".tif"):
                tif_files.append(os.path.join(subdir, file))
    return tif_files

# Function to read TIFF into DataFrame
def tif_to_dataframe(tif_file, chunk_size=10000):
    """
    Reads a TIFF file into a Pandas DataFrame.

    Parameters
    ----------
    tif_file : str
        Full path to the TIFF file to be read.
    chunk_size : int, optional
        Number of rows to read at a time. Default is 10000.

    Returns
    -------
    Pandas DataFrame
        DataFrame with the data from the TIFF file, with columns named 'Band_1', 'Band_2', etc.

    Notes
    -----
    The function prints out the metadata of the TIFF file, the image dimensions and the number of bands.
    It also prints out the progress of the reading process.
    """
    with rio.open(tif_file) as src:
        print(f"\nProcessing file: {tif_file}")
        print("\nMetadata:")
        for key, value in src.meta.items():
            print(f"{key}: {value}")
       
        height, width = src.shape
        n_bands = src.count
       
        print(f"Image dimensions: {width}x{height}")
        print(f"Number of bands: {n_bands}")
       
        df_list = []
        for i in range(0, height, chunk_size):
            data_chunk = src.read(window=rio.windows.Window(0, i, width, min(chunk_size, height - i)))
            reshaped_data_chunk = data_chunk.reshape(data_chunk.shape[0], -1).T
            df_chunk = pd.DataFrame(reshaped_data_chunk, columns=[f'Band_{i+1}' for i in range(data_chunk.shape[0])])
            df_list.append(df_chunk)
        df = pd.concat(df_list, ignore_index=True)
        return df

# Rename bands to the standard Sentinel-2 bands
def rename_bands(df):
    """
    Rename columns of a Pandas DataFrame to the standard Sentinel-2 bands.

    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame containing the data to be renamed.

    Returns
    -------
    Pandas DataFrame
        DataFrame with the columns renamed to the standard Sentinel-2 bands.

    Notes
    -----
    The standard Sentinel-2 bands are: ['Band_1', 'Band_2', 'Band_3', 'Band_4', 'Band_5', 'Band_6', 'Band_7',
    'Band_8', 'Band_8A', 'Band_9', 'Band_11', 'Band_12', 'NBR', 'NDWI', 'EVI']
    """

    new_col_names = ['Band_1', 'Band_2', 'Band_3', 'Band_4', 'Band_5', 'Band_6', 'Band_7',
                     'Band_8', 'Band_8A', 'Band_9', 'Band_11', 'Band_12']
    df.columns = new_col_names

    return df

def clean_data(df):
    """
    Replace infinite values with NaN and drop rows containing NaN.

    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame containing the data to be cleaned.

    Returns
    -------
    Pandas DataFrame
        DataFrame with infinite values replaced with NaN and rows containing NaN dropped.

    Notes
    -----
    The function does not modify the input DataFrame, but returns a new DataFrame with the modifications.
    """
    # Define a tolerance for comparison
    tolerance = 1e-5

    # Remove rows where any column value is close to -0.9999
    df = df[~np.isclose(df, -0.9999, atol=tolerance).any(axis=1)]
    
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()
    print("Shape after cleaning data:", df_clean.shape)
    return df_clean

def fire_index(df_clean):
    """
    Calculate fire indices from a Pandas DataFrame containing Sentinel-2 bands.

    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame containing the Sentinel-2 bands.

    Returns
    -------
    Pandas DataFrame
        DataFrame with the calculated WP and BAIS2 indices.

    Notes
    -----
    The standard Sentinel-2 bands required are: ['Band_1', 'Band_2', 'Band_3', 'Band_4', 'Band_5', 'Band_6', 'Band_7',
    'Band_8', 'Band_8A', 'Band_9', 'Band_11', 'Band_12', 'NDVI', 'NDWI']
    """
    
    # Normalized difference vegetation index Calculation
    ndvi = (df_clean['Band_8'] - df_clean['Band_4']) / (df_clean['Band_8'] + df_clean['Band_4'])
    df_clean['NDVI'] = ndvi
    
    # Normalized difference water index
    ndwi = (df_clean['Band_3'] - df_clean['Band_8']) / (df_clean['Band_3'] + df_clean['Band_8'])
    df_clean["NDWI"] = ndwi
    
    return df_clean

def normalize_data(df, scaler_path):
    """
    Normalize a Pandas DataFrame using a saved MinMaxScaler.

    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame containing the data to be normalized.
    scaler_path : str
        Path to the saved MinMaxScaler.

    Returns
    -------
    Pandas DataFrame
        DataFrame with the normalized data.

    Notes
    -----
    The function loads the saved MinMaxScaler and uses it to transform the input DataFrame.
    """
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    normalized_data = scaler.transform(df)
    df_normalized = pd.DataFrame(normalized_data, columns=df.columns)
    print("Shape after normalization:", df_normalized.shape)
    return df_normalized

def load_model(model_path):
    """
    Load a trained LightGBM model from a pickle file.

    Parameters
    ----------
    model_path : str
        Path to the saved LightGBM model.

    Returns
    -------
    LightGBM model
        The loaded LightGBM model.

    Notes
    -----
    The function assumes that the model was saved using the `pickle` library.
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def make_predictions(model, df):
    
    """
    Make predictions using a trained LightGBM model.

    Parameters
    ----------
    model : LightGBM model
        The trained LightGBM model to use for prediction.
    df : Pandas DataFrame
        DataFrame containing the data to be predicted.

    Returns
    -------
    numpy array
        The predicted values.

    Notes
    -----
    The function assumes that the model was trained using the same features as the input DataFrame.
    """
    
    return model.predict(df)

def resize_predictions(predictions, target_shape):
    """
    Resize a numpy array of predictions to match a target shape.

    Parameters
    ----------
    predictions : numpy array
        Array of predictions to be resized.
    target_shape : tuple
        Shape to resize the predictions to.

    Returns
    -------
    numpy array
        Resized predictions.

    Notes
    -----
    The function uses scipy's `resize` function with order=0 to perform nearest neighbor interpolation.
    The `preserve_range` parameter is set to True to ensure that the output array has the same dtype as the input
    and that the output values are within the same range as the input values.
    """
    resized_predictions = resize(predictions, target_shape, order=0, preserve_range=True)
    return resized_predictions

def create_geotiff_from_predictions(predictions, original_tif_path, output_tif_path):
    """
    Create a new GeoTIFF file from an array of predictions.

    Parameters
    ----------
    predictions : numpy array or Pandas Series
        Array of predictions to be saved as a GeoTIFF.
    original_tif_path : str
        Path to the original GeoTIFF file to copy the metadata from.
    output_tif_path : str
        Path to the new GeoTIFF file to be created.

    Returns
    -------
    None

    Notes
    -----
    The function assumes that the predictions array has the same shape as the original raster.
    The output GeoTIFF file will have the same transform, crs, and nodata value as the original raster.
    The function uses 255 as the no-data value for the output raster.
    """
    if isinstance(predictions, np.ndarray):
        predictions_array = predictions
    else:
        predictions_array = predictions.values

    with rio.open(original_tif_path) as src:
        original_metadata = src.meta.copy()
    
    original_metadata.update({
        'dtype': 'uint8',
        'count': 1,
    })

    full_size_predictions = np.full((original_metadata['height'], original_metadata['width']), 255, dtype=np.uint8)

    with rio.open(original_tif_path) as src:
        valid_data_mask = src.read_masks(1).astype(bool)

    flat_mask = valid_data_mask.flatten()
    full_size_predictions.flat[flat_mask] = predictions_array

    with rio.open(output_tif_path, 'w', **original_metadata) as new_img:
        new_img.write(full_size_predictions, 1)

    print(f"New GeoTIFF file '{output_tif_path}' has been created.")

def process_tif_file(tif_file_path, scaler_path, model_path, output_tif_path):
    """
    Process a TIFF file and generate a new GeoTIFF with predictions.

    Parameters
    ----------
    tif_file_path : str
        Path to the TIFF file to be processed.
    scaler_path : str
        Path to the saved MinMaxScaler.
    model_path : str
        Path to the saved model.
    output_tif_path : str
        Path to the new GeoTIFF file to be created.

    Returns
    -------
    Pandas DataFrame
        DataFrame with the normalized data and predictions.

    Notes
    -----
    The function assumes that the TIFF file has the same shape as the original raster.
    The output GeoTIFF file will have the same transform, crs, and nodata value as the original raster.
    The function uses 255 as the no-data value for the output raster.
    """
    df = tif_to_dataframe(tif_file_path)
    print('Raw DataFrame:')
    display(df) # Raw DataFrame

    original_shape = df.shape[0]
    df = rename_bands(df)
    df_clean = clean_data(df)

    df_clean = fire_index(df_clean)
    print('Clean DataFrame:')
    display(df_clean) # Clean DataFrame

    df_normalized = normalize_data(df_clean, scaler_path)
    print('Normalized DataFrame:')
    display(df_normalized) # Normalized DataFrame

    model = load_model(model_path)
    predictions = make_predictions(model, df_normalized)
    df_predictions = pd.Series(predictions, name='Burn_Classified')
    df_combined = pd.concat([df_normalized, df_predictions], axis=1)
    full_size_predictions = np.full(original_shape, 255, dtype=np.uint8)
    full_size_predictions[df_clean.index] = predictions
    create_geotiff_from_predictions(full_size_predictions, tif_file_path, output_tif_path)
    return df_combined

# Example usage
if __name__ == "__main__":
    root_folder = r'Raster Classified'
    scaler_path = r'Export Model/MinMax_Scaler.pkl'
    model_path = r'Export Model/Model_LGBM.sav' # Choose Model from Export Model Folder
    output_path = r'Classified Output'

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Find all TIFF files in the root folder and subfolders
    tif_files = find_tif_files(root_folder)
    
    # Process each found TIFF file
    for tif_file_path in tif_files:
        # Generate output path based on input file name
        file_name = os.path.basename(tif_file_path)
        output_tif_path = os.path.join(output_path, file_name.replace(".tif", "_Burn_classified.tif"))
        
        print(f"\nProcessing: {tif_file_path}")
        result = process_tif_file(tif_file_path, scaler_path, model_path, output_tif_path)
        display(result)
        
        label_counts = result['Burn_Classified'].value_counts()
        print("\nLabel Counts:")
        print(label_counts)