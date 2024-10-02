import os
import pandas as pd

# Specify the input folder containing the CSV files
input_folder = 'D:/Pycharm_result/Emotion/ADFES_result/sheet_new'

# Specify the output folder for the updated CSV files
output_folder = 'D:/Pycharm_result/Emotion/ADFES_result/delta_sheet'

# Ensure the output folder exists, or create it if it doesn't
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all CSV files in the input folder
csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

# Loop through each CSV file in the input folder
for csv_file in csv_files:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(os.path.join(input_folder, csv_file))

    # Fill NaN values in the original columns with a specific value (0 in this case)
    df = df.fillna(0)

    # Loop through each column (excluding the first column, assumed to be 'Frame no')
    for column in df.columns[1:]:
        # Apply the single delta technique and create a new column
        new_column_name = f"{column}_delta"
        df[new_column_name] = df[column].diff()

    # Reorder the columns to have original columns first, followed by the delta columns
    original_columns = df.columns[:len(df.columns)//2]
    delta_columns = df.columns[len(df.columns)//2:]
    reordered_columns = [col for pair in zip(original_columns, delta_columns) for col in pair]

    # Save the updated DataFrame to a new CSV file in the output folder
    output_file_path = os.path.join(output_folder, csv_file)
    df = df[reordered_columns]
    df.to_csv(output_file_path, index=False)

print("Transformation completed for all files.")
