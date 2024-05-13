import pandas as pd
import numpy as np
import os
import argparse


def combine_csv(root_dir):
   # shape = 0
    overall_combined=pd.DataFrame()
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if dirpath.split("/")[-1] in ['idk', 'KL', 'dpo', 'grad_ascent', 'grad_diff','dpo_perturbed']:
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
                    total_epochs = path_parts[2].split()[0]  
                    data_set = path_parts[-2]
                    checkpoint_number = file.split('-')[1].split(".csv")[0]
                    df["Checkpoint"] = int(checkpoint_number)
                    df["data_set"] = data_set
                    df['Total_Epochs'] = int(total_epochs)
                    df['Model_Used'] = model_used
                    df = df.drop(columns=["Submitted By"])
            
                    combined_data = pd.concat([combined_data, df], ignore_index=True)
             # rearranging checkpoint column with ascending order which reflects epochs in correct order (highest checkpoint - last epoch)
            combined_data = combined_data.sort_values(by="Checkpoint")
            #Calculating the number of epoch
            combined_data["num_epoch"] = combined_data["Total_Epochs"]*(np.array([i for i in range(1,len(combined_data)+1)]))
            combined_data["num_epoch"]/=len(combined_data)
            reorder_cols = list(combined_data.columns[-6:])+ list(combined_data.columns[:-6])
            combined_data = combined_data[reorder_cols]
            columns_ordered = ["num_epoch","Checkpoint"] + [i for i in combined_data.columns if i!="Checkpoint" and i!="num_epoch"]
            combined_data = combined_data[columns_ordered]
            overall_combined = pd.concat([overall_combined,combined_data])
            combined_data.to_csv(dirpath + "/combined.csv",index=False)

    return overall_combined

# Specify the root directory containing the CSV files
root_directory = 'aggr_results'
# Call the function with the directory path
result_df = combine_csv(root_directory)
#print(result_df.head(),result_df.shape)
# Saving the combined results into csv
result_df.to_csv('./combined_results.csv', index=False)