import pandas as pd
import os
import argparse


def combine_csv(root_dir):
   # shape = 0
    overall_combined=pd.DataFrame()
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if dirpath.split("/")[-1] in ['idk', 'KL', 'dpo', 'grad_ascent', 'grad_diff']:
         #   print(filenames)
            combined_data = pd.DataFrame()
            for file in filenames:
                if file.endswith('.csv') and file.split("-")[0]=="checkpoint":
                    # Construct the full file path
                    file_path = os.path.join(dirpath, file)
                    # Read the CSV file
                    df = pd.read_csv(file_path)
                #         break
                #     # Extract features from the path and filename
                    path_parts = dirpath.split(os.sep)
                    
                    model_used = path_parts[1]  
                    num_epochs = path_parts[2].split()[0]  
                    data_set = path_parts[-2]
                    checkpoint_number = file.split('-')[1].split(".csv")[0]
                    df["Checkpoint"] = int(checkpoint_number)
                    df["data_set"] = data_set
                    df['Num_Epochs'] = num_epochs
                    df['Model_Used'] = model_used
                    reorder_cols = list(df.columns[-6:])+ list(df.columns[:-6])
                    df = df[reorder_cols]
                #    df = df.drop(columns=["Submitted By"])
                        # Append this dataframe to the combined data
                    combined_data = pd.concat([combined_data, df], ignore_index=True)
            overall_combined = pd.concat([overall_combined,combined_data])
            combined_data.to_csv(dirpath + "/combined.csv")

    return overall_combined

# Specify the root directory containing the CSV files
root_directory = 'aggr_results'
# Call the function with the directory path
result_df = combine_csv(root_directory)
#print(result_df.shape)
# Saving the combined results into csv
result_df.to_csv('./combined_results.csv', index=False)