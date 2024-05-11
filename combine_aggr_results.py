import pandas as pd
import os
import argparse




def combine_csv(root_dir):
    combined_data = pd.DataFrame()
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith('.csv'):
                # Construct the full file path
                file_path = os.path.join(dirpath, file)
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Extract features from the path and filename
                path_parts = dirpath.split(os.sep)
                
                model_used = path_parts[1]  
                num_epochs = path_parts[2].split()[0]  
                data_set = path_parts[-2]
                method = path_parts[-1] 
                
    
                df['Num_Epochs'] = num_epochs
                df['Model_Used'] = model_used
                df["data_set"] = data_set
                # Append this dataframe to the combined data
                combined_data = pd.concat([combined_data, df], ignore_index=True)

    return combined_data

# Specify the root directory containing the CSV files
root_directory = 'aggr_results'
# Call the function with the directory path
result_df = combine_csv(root_directory)
# Saving the combined results into csv
result_df.to_csv('./combined_results.csv', index=False)