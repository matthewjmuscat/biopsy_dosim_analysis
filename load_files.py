import os
from pathlib import Path
import pandas as pd
import pickle


def find_csv_files(directory, suffixes):
    """
    Recursively searches through a directory for CSV files with specific suffixes.

    Parameters:
        directory (str or Path): The directory to search.
        suffixes (list of str): List of suffixes to match (e.g., ['_data.csv', '_results.csv']).

    Returns:
        list of Path: List of paths to matching CSV files.
    """
    directory = Path(directory)  # Ensure the directory is a Path object
    if not directory.is_dir():
        raise ValueError(f"The provided directory '{directory}' is not a valid directory.")
    
    matching_files = []
    
    # Traverse the directory recursively
    for root, _, files in os.walk(directory):
        for file in files:
            # Check if the file ends with any of the specified suffixes
            if any(file.endswith(suffix) for suffix in suffixes):
                matching_files.append(Path(root) / file)
    
    return matching_files


def load_csv_as_dataframe(file_path):
    """
    Loads a CSV file into a pandas DataFrame.

    Parameters:
        file_path (str or Path): Path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    file_path = Path(file_path)  # Ensure the file path is a Path object
    if not file_path.is_file():
        raise ValueError(f"The file '{file_path}' does not exist.")
    
    # Load the CSV as a DataFrame
    try:
        df = pd.read_csv(file_path, usecols=lambda column: column not in ["Unnamed: 0"])
        print(f"Loaded file: {file_path}")
        return df
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None
    

def load_multiindex_csv(file_path, index_col=None, header_rows=None):
    """
    Loads a CSV file that may contain MultiIndex headers and/or indexes.
    
    Parameters:
        file_path (str or Path): Path to the CSV file.
        index_col (int, list of int, None): Column(s) to set as index. Use None if no index.
        header_rows (list of int, int, None): Row(s) to use as header. Use a list for MultiIndex headers or None for no header.
    
    Returns:
        pd.DataFrame: The loaded DataFrame with the correct MultiIndex structure.
    """
    file_path = Path(file_path)  # Ensures the file path is a Path object
    if not file_path.is_file():
        raise ValueError(f"The file '{file_path}' does not exist.")

    # Attempt to load the DataFrame with specified index and header rows
    try:
        df = pd.read_csv(file_path, header=header_rows, index_col=index_col)
        
        # Check for 'Unnamed' columns in the DataFrame after loading
        if any("Unnamed" in str(col) for col in df.columns.get_level_values(0)):  # Check in the first level
            # Iterate through all levels of the MultiIndex columns
            df.columns = pd.MultiIndex.from_tuples([
                tuple("" if "Unnamed" in part else part for part in col)
                for col in df.columns
            ])
            # Filter out completely empty columns (where all parts are '')
            df = df.loc[:, ~df.columns.to_series().apply(lambda x: all(part == "" for part in x))]
        
        return df
    except Exception as e:
        raise Exception(f"Error loading file {file_path}: {e}")
    






def load_parquet_as_dataframe(file_path):
    """
    Loads a Parquet file into a pandas DataFrame.

    Parameters:
        file_path (str or Path): Path to the Parquet file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    file_path = Path(file_path)  # Ensure the file path is a Path object
    if not file_path.is_file():
        raise ValueError(f"The file '{file_path}' does not exist.")
    
    try:
        df = pd.read_parquet(file_path)
        print(f"Loaded file: {file_path}")
        return df
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None





def load_master_dict(main_output_path,
                      master_structure_dict):
    
    # Use rglob to traverse directories recursively and find folders ending with "pickled data"
    found_folders = [folder for folder in main_output_path.rglob("*") 
                    if folder.is_dir() and folder.name.endswith("pickled data")]

    # Print out the found folders
    if found_folders:
        print("Found the following folder(s) ending with 'pickled data':")
        for folder in found_folders:
            print(folder)
    else:
        print("No folder ending with 'pickled data' was found under the specified directory.")

    # Assuming you want to use the first found folder
    pickled_data_folder = found_folders[0]

    # master_structure_info_dict
    results_master_structure_dict_path = pickled_data_folder.joinpath(master_structure_dict)
    print(f'Loading {master_structure_dict} from: {results_master_structure_dict_path}')
    with open(results_master_structure_dict_path, "rb") as preprocessed_master_structure_dict_file:
        master_structure_dict = pickle.load(preprocessed_master_structure_dict_file) 
    print(f"Done!")

    return master_structure_dict