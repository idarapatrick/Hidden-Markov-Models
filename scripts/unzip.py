import os
import zipfile
import pandas as pd

def extract_merge_and_label():
    """
    Dynamically searches zip files for sensor data, ignoring folder structures 
    and case sensitivity, then merges and standardizes them.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    zip_folder = os.path.join(project_root, 'raw_zips_inbox')
    
    # Target the unseen test data folder
    output_folder = os.path.join(project_root, 'data', 'unseen_test_data')
    os.makedirs(output_folder, exist_ok=True)
    
    zip_files = [f for f in os.listdir(zip_folder) if f.endswith('.zip')]
    
    for index, filename in enumerate(zip_files):
        activity_name = filename.split('_')[0].lower() 
        zip_path = os.path.join(zip_folder, filename)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                # Generate a list of everything inside the zip file
                file_list = z.namelist()
                
                # Dynamically search for the exact paths of our required files
                acc_path = next((f for f in file_list if 'accelerometer.csv' in f.lower()), None)
                gyro_path = next((f for f in file_list if 'gyroscope.csv' in f.lower()), None)
                
                # If either file is completely missing, skip gracefully
                if not acc_path or not gyro_path:
                    print(f"Skipping {filename}: Required CSVs not found inside.")
                    continue
                    
                # Open using the dynamically found paths
                with z.open(acc_path) as acc_file:
                    acc_df = pd.read_csv(acc_file)
                    
                with z.open(gyro_path) as gyro_file:
                    gyro_df = pd.read_csv(gyro_file)
                    
        except zipfile.BadZipFile:
            print(f"Skipping {filename}: Zip archive is corrupted.")
            continue
        except Exception as e:
            print(f"Skipping {filename} due to unexpected error: {e}")
            continue
            
        acc_df = acc_df.sort_values('time')
        gyro_df = gyro_df.sort_values('time')
        
        merged_df = pd.merge_asof(
            acc_df, 
            gyro_df, 
            on='time', 
            direction='nearest',
            suffixes=('_acc', '_gyro')
        )
        
        new_filename = f"{activity_name}_{index+1:02d}.csv"
        output_filepath = os.path.join(output_folder, new_filename)
        
        merged_df.to_csv(output_filepath, index=False)
        print(f"Success: {filename} has been dynamically located and saved as {new_filename}")

extract_merge_and_label()