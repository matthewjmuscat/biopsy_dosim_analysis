import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def get_filtered_statistics(df, columns=None, simulated_type=None, patient_id=None):
    """
    Calculate statistics for specific columns in the DataFrame, filtered by Simulated type and Patient ID(s),
    and include additional statistics like skewness and kurtosis.

    Args:
        df (pd.DataFrame): The input DataFrame containing biopsy spatial features.
        columns (list, optional): List of column names for which to calculate statistics. Defaults to None (all numeric columns).
        simulated_type (str, optional): Filter the DataFrame by this Simulated type value. Defaults to None (no filtering).
        patient_id (str or list, optional): Filter the DataFrame by Patient ID(s). Defaults to None (assumes all patients).

    Returns:
        pd.DataFrame: A DataFrame containing statistics for the specified columns.
    """
    # Filter the DataFrame by Patient ID if provided
    if patient_id is not None:
        if isinstance(patient_id, list):
            df = df[df['Patient ID'].isin(patient_id)]
        else:
            df = df[df['Patient ID'] == patient_id]

    # Filter the DataFrame by Simulated type if provided
    if simulated_type is not None:
        df = df[df['Simulated type'] == simulated_type]

    # If no columns are specified, use all numeric columns
    if columns is None:
        columns = df.select_dtypes(include='number').columns.tolist()

    # Ensure the specified columns exist in the DataFrame
    columns = [col for col in columns if col in df.columns]

    # Calculate basic statistics for the specified columns
    stats_df = df[columns].describe(percentiles=[0.05, 0.95]).transpose()

    # Calculate additional statistics: skewness and kurtosis
    #stats_df['skewness'] = df[columns].apply(skew, nan_policy='omit')
    #stats_df['kurtosis'] = df[columns].apply(kurtosis, nan_policy='omit')

    return stats_df


# Write a function that finds the percentage of biopsies in each double setrant considering the columns of the dataframe 'Bx position in prostate LR', 'Bx position in prostate AP', 'Bx position in prostate SI'
def find_biopsy_double_sextant_percentages(df, patient_id=None, simulated_type=None):
    """
    Calculate the percentage and count of biopsies in each double sextant based on the biopsy positions in the
    prostate Left/Right, Anterior/Posterior, Base (Superior)/Mid/Apex (Inferior), with optional filtering by
    Patient ID and Simulated type.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing biopsy spatial features.
        patient_id (str or list, optional): Filter the DataFrame by Patient ID(s). Defaults to None (assumes all patients).
        simulated_type (str, optional): Filter the DataFrame by Simulated type. Defaults to None (assumes all types).
    
    Returns:
        pd.DataFrame: A DataFrame containing the count, percentage, and total biopsies information.
    """
    # Correct mapping for sextants based on actual values in the DataFrame
    sextant_mapping = {
        'Left': 'L',
        'Right': 'R',
        'Anterior': 'A',
        'Posterior': 'P',
        'Base (Superior)': 'S',
        'Mid': 'M',
        'Apex (Inferior)': 'I'
    }
    
    # Filter the DataFrame by Patient ID if provided
    if patient_id is not None:
        if isinstance(patient_id, list):
            df = df[df['Patient ID'].isin(patient_id)]
        else:
            df = df[df['Patient ID'] == patient_id]
    
    # Filter the DataFrame by Simulated type if provided
    if simulated_type is not None:
        df = df[df['Simulated type'] == simulated_type]
    
    # Ensure required columns are present
    required_columns = ['Bx position in prostate LR', 'Bx position in prostate AP', 'Bx position in prostate SI']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"The DataFrame must contain the following columns: {required_columns}")
    
    # Map sextant values and create a new column for the double sextant
    df['Double Sextant'] = (
        df['Bx position in prostate LR'].map(sextant_mapping) +
        df['Bx position in prostate AP'].map(sextant_mapping) +
        df['Bx position in prostate SI'].map(sextant_mapping)
    )
    
    # Count occurrences of each double sextant
    sextant_counts = df['Double Sextant'].value_counts().reset_index()
    sextant_counts.columns = ['Double Sextant', 'Count']
    
    # Ensure all possible double sextants are included
    all_double_sextants = [
        f"{lr}{ap}{si}" for lr in ['L', 'R']
        for ap in ['A', 'P']
        for si in ['S', 'M', 'I']
    ]
    sextant_counts = sextant_counts.set_index('Double Sextant').reindex(all_double_sextants, fill_value=0).reset_index()
    
    # Calculate percentages
    total_biopsies = sextant_counts['Count'].sum()
    if total_biopsies > 0:
        sextant_counts['Percentage'] = (sextant_counts['Count'] / total_biopsies) * 100
    else:
        sextant_counts['Percentage'] = 0
    
    # Add a "Total Biopsies" row at the top
    total_row = pd.DataFrame({
        'Double Sextant': ['Total Biopsies'],
        'Count': [total_biopsies],
        'Percentage': [100.0]
    })
    sextant_counts = pd.concat([total_row, sextant_counts], ignore_index=True)
    
    return sextant_counts[['Double Sextant', 'Count', 'Percentage']]