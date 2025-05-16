import pandas as pd
import os

def generate_summary_csv(output_dir, csv_name, df, col_pairs=None, exclude_columns=None):
    """
    Generate summary statistics for specified columns in a DataFrame and save the results to CSV.
    
    If col_pairs is None, summary statistics will be computed for all columns in the DataFrame, 
    except those specified in exclude_columns.
    
    Parameters:
        output_dir (str): Directory path where the summary CSV will be saved.
        csv_name (str): The name of the CSV file (e.g., 'summary.csv').
        df (pd.DataFrame): The input DataFrame with a MultiIndex for its columns.
        col_pairs (list of tuple, optional): List of tuples representing the columns to summarize,
                                               e.g., [('Dose (Gy)', 'mean'), ('Dose (Gy)', 'argmax_density'),
                                                      ('Dose grad (Gy/mm)', 'argmax_density')].
                                               If None, all columns in the DataFrame (not excluded) will be summarized.
        exclude_columns (list of tuple, optional): List of column pairings to exclude from the summary. 
                                                     These should be in the same format as col_pairs.
    
    The function computes the descriptive statistics (e.g., count, mean, std, min, quartiles, max) 
    for each column (aggregating all rows) and saves the final summary table to a CSV file.
    """
    # Ensure the output directory exists; create it if it doesn't
    os.makedirs(output_dir, exist_ok=True)

    # If col_pairs is None, use all columns in the DataFrame.
    if col_pairs is None:
        col_pairs = list(df.columns)
    
    # If exclude_columns is provided, remove them from col_pairs.
    if exclude_columns:
        col_pairs = [pair for pair in col_pairs if pair not in exclude_columns]

    summary_list = []
    pairing_labels = []

    # Iterate over each requested column pair
    for pair in col_pairs:
        if pair in df.columns:
            # Compute descriptive statistics for the column (as a Series)
            stats = df[pair].describe()
            summary_list.append(stats)
            # Create a label for this pairing; if the second level label is empty, just use the first level label
            label = f"{pair[0]} | {pair[1]}" if pair[1] else pair[0]
            pairing_labels.append(label)
        else:
            print(f"Warning: Column {pair} not found in the DataFrame.")
    
    # Ensure that at least one valid column was summarized
    if summary_list:
        # Combine the Series into a DataFrame, with each row representing one column pairing.
        summary_df = pd.DataFrame(summary_list, index=pairing_labels)
        
        # Construct the full output file path
        output_path = os.path.join(output_dir, csv_name)
        
        # Save the summary DataFrame to CSV
        summary_df.to_csv(output_path)
        print(f"Summary CSV successfully saved to: {output_path}")
    else:
        print("No valid columns were provided. No CSV was generated.")


