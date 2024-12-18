# Library Package
import os
import rasterio as rio
import numpy as np
import pandas as pd
import logging
from skimage.transform import resize
from IPython.display import display
import pickle
import warnings
from tensorflow.keras.models import load_model

pd.set_option("display.max_columns", None)

# Set up logging configuration for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def find_tif_files(root_folder):
    """
    Recursively search for all files with the extension '.tif' in the root folder and its subfolders.
    """
    tif_files = []
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".tif"):
                tif_files.append(os.path.join(subdir, file))
    return tif_files

def read_raster_with_nodata(tif_file, chunk_size=10000):
    """
    Read a TIFF file, handling no data and NaN values more robustly.
    
    Returns
    -------
    tuple
        DataFrame with raster data, valid data mask, and nodata value
    """
    with rio.open(tif_file) as src:
        # Print metadata
        print(f"\nProcessing file: {tif_file}")
        print("\nMetadata:")
        for key, value in src.meta.items():
            print(f"{key}: {value}")
       
        height, width = src.shape
        n_bands = src.count
        nodata_value = src.nodata if src.nodata is not None else np.nan
       
        print(f"Image dimensions: {width}x{height}")
        print(f"Number of bands: {n_bands}")
        print(f"Nodata value: {nodata_value}")
       
        df_list = []
        mask_list = []
        
        for i in range(0, height, chunk_size):
            # Read data chunk
            data_chunk = src.read(window=rio.windows.Window(0, i, width, min(chunk_size, height - i)))
            
            # Create mask for valid pixels
            if nodata_value is not None:
                valid_mask = ~np.isclose(data_chunk, nodata_value, equal_nan=True)
            else:
                valid_mask = ~np.isnan(data_chunk)
            
            # Reshape data
            reshaped_data_chunk = data_chunk.reshape(data_chunk.shape[0], -1).T
            df_chunk = pd.DataFrame(reshaped_data_chunk, columns=[f'Band_{i+1}' for i in range(data_chunk.shape[0])])
            
            # Reshape mask
            reshaped_mask_chunk = valid_mask.reshape(valid_mask.shape[0], -1).T
            mask_chunk = reshaped_mask_chunk.all(axis=1)
            
            df_list.append(df_chunk)
            mask_list.append(mask_chunk)
        
        df = pd.concat(df_list, ignore_index=True)
        valid_mask = np.concatenate(mask_list)
        
        return df, valid_mask, nodata_value

def rename_bands(df):
    """
    Rename columns of a Pandas DataFrame to the standard Sentinel-2 bands.
    """
    new_col_names = ['Band_1', 'Band_2', 'Band_3', 'Band_4', 'Band_5', 'Band_6', 'Band_7',
                     'Band_8', 'Band_8A', 'Band_9', 'Band_11', 'Band_12']
    df.columns = new_col_names
    return df

def clean_data(df, valid_mask):
    """
    Clean data by applying valid pixel mask and handling NaN/infinite values.
    """
    # Apply valid pixel mask
    df_clean = df[valid_mask].copy()
    
    # Define a tolerance for comparison
    tolerance = 1e-5

    # Remove rows with extreme or invalid values
    df_clean = df_clean[~np.isclose(df_clean, -0.9999, atol=tolerance).any(axis=1)]
    
    # Replace infinite values with NaN and drop NaN rows
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
    
    print("Shape after cleaning data:", df_clean.shape)
    
    return df_clean

def fire_index(df_clean):
    """
    Calculate fire indices from a Pandas DataFrame containing Sentinel-2 bands.
    """
    # Normalized difference vegetation index Calculation
    ndvi = (df_clean['Band_8'] - df_clean['Band_4']) / (df_clean['Band_8'] + df_clean['Band_4'])
    df_clean['NDVI'] = ndvi
    
    # Normalized difference water index
    ndwi = (df_clean['Band_3'] - df_clean['Band_8']) / (df_clean['Band_3'] + df_clean['Band_8'])
    df_clean["NDWI"] = ndwi
    
    return df_clean

def process_tif_file_in_chunks(tif_file_path, scaler_path, model_path, output_tif_path, chunk_size=50000):
    """
    Process a TIFF file in chunks using a TensorFlow model with binary output.
    """
    # Read raster with robust no data handling
    df, valid_mask, nodata_value = read_raster_with_nodata(tif_file_path)

    # Open the original TIFF file to get metadata
    with rio.open(tif_file_path) as src:
        height, width = src.shape
        original_metadata = src.meta.copy()

    # Prepare the final predictions array 
    # Use 0 as the default value for invalid/no data pixels
    final_predictions = np.zeros((height * width), dtype=np.uint8)

    # Load the TensorFlow model and scaler
    model = load_model(model_path)  # Load the TensorFlow .h5 model
    
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)

    # Rename bands
    df = rename_bands(df)
    print("Rename DataFrame: ")
    display(df.head())
    print()

    # Clean data
    df_clean = clean_data(df, valid_mask)
    print("Cleaned DataFrame: ")
    display(df_clean.head())
    print()

    # Compute fire indices
    df_clean = fire_index(df_clean)
    print("Full DataFrame: ")
    display(df_clean.head())
    print()

    # Normalize data
    df_normalized = pd.DataFrame(
        scaler.transform(df_clean), 
        columns=df_clean.columns
    )
    print("Normalized DataFrame: ")
    display(df_normalized.head())
    print()

    # Convert normalized DataFrame to NumPy array for TensorFlow model
    normalized_array = df_normalized.to_numpy()

    # Make predictions using the TensorFlow model
    predictions = model.predict(normalized_array)
    
    # Post-process predictions to binary (0 or 1)
    binary_predictions = (predictions > 0.5).astype(np.uint8).flatten()

    # Update final predictions with valid predictions
    final_predictions[valid_mask.flatten()] = binary_predictions

    # Reshape predictions to original raster shape
    final_predictions_2d = final_predictions.reshape(height, width)

    # Create the output GeoTIFF
    updated_metadata = original_metadata.copy()
    updated_metadata.update({
        'dtype': 'uint8',  # Change to uint8 for binary output
        'count': 1,
        'nodata': 0  # Use 0 as no data value
    })

    with rio.open(output_tif_path, 'w', **updated_metadata) as new_img:
        new_img.write(final_predictions_2d, 1)

    print(f"New GeoTIFF file '{output_tif_path}' has been created.")

    # Return summary of predictions
    unique, counts = np.unique(final_predictions_2d, return_counts=True)
    prediction_summary = dict(zip(unique, counts))
    print("\nPrediction Summary:")
    print(prediction_summary)

    return prediction_summary

def process_all_tif_files(root_folder, scaler_path, model_path, output_path, chunk_size=50000):
    """
    Process all TIFF files in a root folder with binary output.
    """
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Find all TIFF files in the root folder and subfolders
    tif_files = find_tif_files(root_folder)
    
    # Process each found TIFF file
    for tif_file_path in tif_files:
        # Generate output path based on input file name
        file_name = os.path.basename(tif_file_path)
        output_tif_path = os.path.join(output_path, file_name.replace(".tif", "DL_Burn_classified.tif"))
        
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
    root_folder = r'Raster Classified Cloud Mask'
    scaler_path = r'Export Model/MinMax_Scaler.pkl'
    model_path = r'Export Model/DL_Model.h5'
    output_path = r'Classified Output'

    process_all_tif_files(
        root_folder, 
        scaler_path, 
        model_path, 
        output_path, 
        chunk_size=512
    )