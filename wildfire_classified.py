import os
import rasterio as rio
import numpy as np
import pandas as pd
import logging
from IPython.display import display
import pickle
import warnings

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
    new_col_names = ['Band_2', 'Band_3', 'Band_4', 'Band_5', 'Band_6', 'Band_7',
                     'Band_8', 'Band_8A', 'Band_11', 'Band_12']
    df.columns = new_col_names
    return df

def clean_data(df, valid_mask, chunk_size=100000):
    """
    Clean data by applying valid pixel mask and handling NaN/infinite values.
    Memory-optimized version that processes data in chunks.
    """
    # Apply valid pixel mask
    df_clean = df[valid_mask].copy().reset_index(drop=True)
    
    # Define a tolerance for comparison
    tolerance = 1e-5
    
    # Process in chunks to reduce memory usage
    total_rows = len(df_clean)
    chunks = []
    
    for start_idx in range(0, total_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_rows)
        chunk = df_clean.iloc[start_idx:end_idx].copy()
        
        # Remove rows with values close to -0.9999
        for col in chunk.columns:
            chunk = chunk[~np.isclose(chunk[col], -0.9999, atol=tolerance)]
        
        # Replace infinite values with NaN and drop NaN rows
        chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()
        
        chunks.append(chunk)
    
    if not chunks:
        return pd.DataFrame(columns=df_clean.columns)
    
    return pd.concat(chunks, ignore_index=True)

def fire_index(df_clean, chunk_size=100000):
    """
    Calculate fire indices from a Pandas DataFrame containing Sentinel-2 bands.
    Memory-optimized version that processes data in chunks.
    """
    total_rows = len(df_clean)
    chunks = []
    
    for start_idx in range(0, total_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_rows)
        chunk = df_clean.iloc[start_idx:end_idx].copy()
        
        # Normalized Burn Ratio
        chunk['NBR'] = (chunk['Band_8A'] - chunk['Band_12']) / (chunk['Band_8A'] + chunk['Band_12'])

        # Normalized difference vegetation index Calculation
        chunk['NDVI'] = (chunk['Band_8'] - chunk['Band_4']) / (chunk['Band_8'] + chunk['Band_4'])
        
        # Normalized difference water index
        chunk['NDWI'] = (chunk['Band_3'] - chunk['Band_8']) / (chunk['Band_3'] + chunk['Band_8'])

        # Differenced Normalized Burn Ratio Short Wave Infrared (NBRSWIR)
        chunk['NBRSWIR'] = (chunk['Band_12'] - chunk['Band_11'] - 0.02) / (chunk['Band_12'] + chunk['Band_11'] + 0.1)
        
        # Define a tolerance for comparison
        tolerance = 1e-5

        # Remove rows with extreme or invalid values - process column by column to save memory
        for col in chunk.columns:
            chunk = chunk[~np.isclose(chunk[col], -0.9999, atol=tolerance)]
        
        # Replace infinite values with NaN and drop NaN rows
        chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()
        
        chunks.append(chunk)
    
    if not chunks:
        return pd.DataFrame(columns=list(df_clean.columns) + ['NBR', 'NDVI', 'NDWI', 'NBRSWIR'])
    
    result = pd.concat(chunks, ignore_index=True)
    print("Shape after calculating fire indices and cleaning data:", result.shape)
    
    return result

def process_tif_file_in_chunks(tif_file_path, scaler_path, model_path, output_tif_path, chunk_size=50000):
    """
    Process a TIFF file in chunks with binary output.
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

    # Load the model and scaler
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)

    # Rename bands
    df = rename_bands(df)
    print("Rename DataFrame: ")
    display(df.head())
    print()

    # Clean data with chunk processing
    df_clean = clean_data(df, valid_mask, chunk_size=chunk_size)
    print("Cleaned DataFrame: ")
    display(df_clean.head())
    print()

    # Calculate indices with chunk processing
    df_clean = fire_index(df_clean, chunk_size=chunk_size)
    print("Full DataFrame: ")
    display(df_clean.head())
    print()

    # Filter the DataFrame to only include rows where NDWI is less than 0.5 (not water)
    # We can do this in chunks too if the DataFrame is very large
    total_rows = len(df_clean)
    filtered_chunks = []
    
    for start_idx in range(0, total_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_rows)
        chunk = df_clean.iloc[start_idx:end_idx]
        filtered_chunk = chunk[chunk['NDWI'] < 0.5]
        filtered_chunks.append(filtered_chunk)
    
    if not filtered_chunks:
        print("No valid data left after filtering. Returning empty predictions.")
        return {"0": width * height}
    
    df_clean = pd.concat(filtered_chunks, ignore_index=True)

    # Get the indices of the remaining valid pixels after filtering
    valid_indices = df_clean.index.to_numpy()

    # Process predictions in chunks to reduce memory usage
    total_rows = len(df_clean)
    
    for start_idx in range(0, total_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_rows)
        chunk = df_clean.iloc[start_idx:end_idx]
        
        # Normalize data
        chunk_normalized = pd.DataFrame(
            scaler.transform(chunk), 
            columns=chunk.columns
        )
        
        # Make predictions
        chunk_predictions = model.predict(chunk_normalized)
        
        # Update final predictions for this chunk
        chunk_indices = valid_indices[start_idx:end_idx]
        final_predictions[chunk_indices] = chunk_predictions.astype(np.uint8)
    
    # Display sample of normalized data
    if len(df_clean) > 0:
        sample_normalized = pd.DataFrame(
            scaler.transform(df_clean.head(5)), 
            columns=df_clean.columns
        )
        print("Normalized DataFrame (Sample): ")
        display(sample_normalized)
        print()

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
        output_tif_path = os.path.join(output_path, file_name.replace(".tif", "_Burn_classified.tif"))
        
        print(f"\nProcessing: {tif_file_path}")
        result = process_tif_file_in_chunks(
            tif_file_path, 
            scaler_path, 
            model_path, 
            output_tif_path, 
            chunk_size=chunk_size
        )

def main():
    """
    Example usage of the script.

    Process all classified TIFF files in 'Raster_Classified_Cloud_Mask' folder,
    using the scaler and model from 'Export_Model' folder, and save the results
    in 'Classified_Output' folder.
    """
    root_folder = r'Raster_Classified_Cloud_Mask'
    scaler_path = r'Export_Model/MinMax_Scaler.pkl'
    model_path = r'Export_Model/Model_XGB.sav'
    output_path = r'Classified_Output'

    # Using smaller chunk sizes to reduce memory usage
    process_all_tif_files(
        root_folder, 
        scaler_path, 
        model_path, 
        output_path, 
        chunk_size=10000  # Reduced chunk size
    )

if __name__ == "__main__":
    main()