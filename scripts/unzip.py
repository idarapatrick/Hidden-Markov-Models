import os
import zipfile
import pandas as pd

def extract_merge_and_label():
    """
    Extracts Accelerometer and Gyroscope data from zip files, merges them, 
    and automatically renames the output file based on the activity.
    """
    # Get the absolute path of the directory this script is in
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go one level up to get the main project folder
    project_root = os.path.dirname(script_dir)
    
    # Build absolute paths to the folders safely
    zip_folder = os.path.join(project_root, 'raw_zips_inbox')
    output_folder = os.path.join(project_root, 'data', 'raw')
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all zip files in your inbox folder
    zip_files = [f for f in os.listdir(zip_folder) if f.endswith('.zip')]
    
    for index, filename in enumerate(zip_files):
        # Extract the activity name from the start of the zip file
        activity_name = filename.split('_')[0].lower() 
        zip_path = os.path.join(zip_folder, filename)
        
        with zipfile.ZipFile(zip_path, 'r') as z:
            with z.open('Accelerometer.csv') as acc_file:
                acc_df = pd.read_csv(acc_file)
                
            with z.open('Gyroscope.csv') as gyro_file:
                gyro_df = pd.read_csv(gyro_file)
        
        # Sort both dataframes by time to ensure accurate merging
        acc_df = acc_df.sort_values('time')
        gyro_df = gyro_df.sort_values('time')
        
        # Merge by finding the nearest gyroscope timestamp for every accelerometer timestamp
        merged_df = pd.merge_asof(
            acc_df, 
            gyro_df, 
            on='time', 
            direction='nearest',
            suffixes=('_acc', '_gyro')
        )
        
        # Generates a clean name like: "standing_01.csv"
        new_filename = f"{activity_name}_{index+1:02d}.csv"
        output_filepath = os.path.join(output_folder, new_filename)
        
        merged_df.to_csv(output_filepath, index=False)
        print(f"Success: {filename} has been saved as {new_filename}")

# Run the extraction for all files in the inbox
extract_merge_and_label()