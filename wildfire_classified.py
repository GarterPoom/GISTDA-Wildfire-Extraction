# Library Package
import os
import rasterio as rio
import numpy as np
import pandas as pd
import logging
from skimage.transform import resize
from IPython.display import display
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
    'Band_8', 'Band_8A', 'Band_9', 'Band_11', 'Band_12']
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

def process_tif_file_in_chunks(tif_file_path, scaler_path, model_path, output_tif_path, chunk_size=50000):
    """
    Process a TIFF file in chunks to reduce memory usage.

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
    chunk_size : int, optional
        Number of rows to process in each chunk. Default is 50000.

    Returns
    -------
    Pandas DataFrame
        DataFrame with the normalized data and predictions.
    """
    # Open the original TIFF file to get metadata
    with rio.open(tif_file_path) as src:
        height, width = src.shape
        n_bands = src.count
        original_metadata = src.meta.copy()

    # Load the model and scaler once
    model = load_model(model_path)
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)

    # Prepare the final predictions array
    final_predictions = np.full((height, width), 255, dtype=np.uint8)

    # Read the mask to identify valid pixels
    with rio.open(tif_file_path) as src:
        valid_data_mask = src.read_masks(1).astype(bool)

    # Process the file in chunks
    for start_row in range(0, height, chunk_size):
        end_row = min(start_row + chunk_size, height)
        chunk_height = end_row - start_row

        # Read the chunk
        with rio.open(tif_file_path) as src:
            chunk_data = src.read(window=rio.windows.Window(0, start_row, width, chunk_height))

        # Reshape the chunk
        reshaped_data_chunk = chunk_data.reshape(chunk_data.shape[0], -1).T
        df_chunk = pd.DataFrame(reshaped_data_chunk, columns=[f'Band_{i+1}' for i in range(chunk_data.shape[0])])
        print("Raw DataFrame: ")
        display(df_chunk.head())
        print() # Add Blank line

        # Rename bands
        df_chunk = rename_bands(df_chunk)
        print("Rename DataFrame: ")
        display(df_chunk.head())
        print() # Add Blank line

        # Clean data
        df_clean = clean_data(df_chunk)
        print("Cleaned DataFrame: ")
        display(df_clean.head())
        print() # Add Blank line

        # Compute fire indices
        df_clean = fire_index(df_clean)
        print("Full DataFrame: ")
        display(df_clean.head())
        print() # Add Blank line

        # Normalize data
        df_normalized = pd.DataFrame(
            scaler.transform(df_clean), 
            columns=df_clean.columns
        )
        print("Normalized DataFrame: ")
        display(df_normalized.head())
        print() # Add Blank line

        # Make predictions
        predictions = make_predictions(model, df_normalized)

        # Update the final predictions array
        chunk_mask = valid_data_mask[start_row:end_row, :].flatten()
        chunk_indices = np.arange(start_row * width, end_row * width)[chunk_mask]
        
        final_predictions.flat[chunk_indices] = predictions

        print(f"Processed chunk {start_row} to {end_row}")

    # Create the output GeoTIFF
    updated_metadata = original_metadata.copy()
    updated_metadata.update({
        'dtype': 'uint8',
        'count': 1,
    })

    with rio.open(output_tif_path, 'w', **updated_metadata) as new_img:
        new_img.write(final_predictions, 1)

    print(f"New GeoTIFF file '{output_tif_path}' has been created.")

    # Return summary of predictions
    unique, counts = np.unique(final_predictions[final_predictions != 255], return_counts=True)
    prediction_summary = dict(zip(unique, counts))
    print("\nPrediction Summary:")
    print(prediction_summary)

    return prediction_summary

def process_all_tif_files(root_folder, scaler_path, model_path, output_path, chunk_size=50000):
    """
    Process all TIFF files in a root folder with chunked processing.

    Parameters
    ----------
    root_folder : str
        Root folder to search for TIFF files.
    scaler_path : str
        Path to the saved MinMaxScaler.
    model_path : str
        Path to the saved model.
    output_path : str
        Path to save the processed TIFF files.
    chunk_size : int, optional
        Number of rows to process in each chunk. Default is 50000.
    """
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
        result = process_tif_file_in_chunks(
            tif_file_path, 
            scaler_path, 
            model_path, 
            output_tif_path, 
            chunk_size=chunk_size
        )

# Example usage
if __name__ == "__main__":
    root_folder = r'Raster Classified'
    scaler_path = r'Export Model/MinMax_Scaler.pkl'
    model_path = r'Export Model/Model_XGB.sav' # Choose model from Export Model
    output_path = r'Classified Output'

    process_all_tif_files(
        root_folder, 
        scaler_path, 
        model_path, 
        output_path, 
        chunk_size=4096  # Adjust chunk size based on your system's memory
    )