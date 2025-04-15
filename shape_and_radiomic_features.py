import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def get_radiomic_statistics(df, patient_id=None, structure_types=None, exclude_columns=None):
    """
    Generate a multi-index DataFrame with statistical information for radiomic feature columns,
    grouped by structure type across all patients in the given list.

    Args:
        df (pd.DataFrame): The input DataFrame containing radiomic features.
        patient_id (str or list, optional): Filter by Patient ID(s). Defaults to None.
        structure_types (list, optional): List of structure types to include. Defaults to None.
        exclude_columns (list, optional): List of columns to exclude from the statistics. Defaults to None.

    Returns:
        pd.DataFrame: A multi-index DataFrame containing statistical information for each structure type.
    """
    # Filter the DataFrame based on the provided arguments
    filtered_df = df.copy()
    if patient_id:
        if isinstance(patient_id, list):
            filtered_df = filtered_df[filtered_df['Patient ID'].isin(patient_id)]
        else:
            filtered_df = filtered_df[filtered_df['Patient ID'] == patient_id]
    if structure_types:
        if isinstance(structure_types, list):
            filtered_df = filtered_df[filtered_df['Structure type'].isin(structure_types)]
        else:
            filtered_df = filtered_df[filtered_df['Structure type'] == structure_types]

    # Determine radiomic feature columns
    exclude_columns = exclude_columns or []
    non_radiomic_columns = ['Patient ID', 'Structure ID', 'Structure type', 'Structure refnum']
    radiomic_columns = [col for col in filtered_df.columns if col not in non_radiomic_columns + exclude_columns]

    # Group by structure type and calculate statistics for radiomic feature columns
    stats = []
    for structure_type, group in filtered_df.groupby('Structure type'):
        # Basic statistics
        stats_df = group[radiomic_columns].describe(percentiles=[0.05, 0.95]).transpose()
        
        # Additional statistics: skewness and kurtosis
        stats_df['skewness'] = group[radiomic_columns].apply(skew, nan_policy='omit')
        stats_df['kurtosis'] = group[radiomic_columns].apply(kurtosis, nan_policy='omit')
        
        # Add structure type to the DataFrame
        stats_df['Structure type'] = structure_type
        stats.append(stats_df)

    # Combine statistics into a single DataFrame with a multi-index
    result_df = pd.concat(stats).set_index('Structure type', append=True).reorder_levels(['Structure type', None])

    return result_df




def find_dil_double_sextant_percentages(df, patient_id=None):
    """
    Calculate the percentage and count of DILs in each double sextant based on the DIL positions in the
    prostate Left/Right, Anterior/Posterior, Base (Superior)/Mid/Apex (Inferior).
    
    Args:
        df (pd.DataFrame): The input DataFrame containing DIL spatial features.
        patient_id (str or list, optional): Filter the DataFrame by Patient ID(s). Defaults to None (assumes all patients).
    
    Returns:
        pd.DataFrame: A DataFrame containing the count, percentage, and total DILs information.
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
    
    # Ensure required columns are present
    required_columns = ['DIL prostate sextant (LR)', 'DIL prostate sextant (AP)', 'DIL prostate sextant (SI)']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"The DataFrame must contain the following columns: {required_columns}")
    
    # Map sextant values and create a new column for the double sextant
    df['Double Sextant'] = (
        df['DIL prostate sextant (LR)'].map(sextant_mapping) +
        df['DIL prostate sextant (AP)'].map(sextant_mapping) +
        df['DIL prostate sextant (SI)'].map(sextant_mapping)
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
    total_dils = sextant_counts['Count'].sum()
    if total_dils > 0:
        sextant_counts['Percentage'] = (sextant_counts['Count'] / total_dils) * 100
    else:
        sextant_counts['Percentage'] = 0
    
    # Add a "Total DILs" row at the top
    total_row = pd.DataFrame({
        'Double Sextant': ['Total DILs'],
        'Count': [total_dils],
        'Percentage': [100.0]
    })
    sextant_counts = pd.concat([total_row, sextant_counts], ignore_index=True)
    
    return sextant_counts[['Double Sextant', 'Count', 'Percentage']]




def calculate_structure_counts_and_stats(df, patient_id=None, structure_types=None):
    """
    Calculate the number of structures for each patient grouped by structure type and produce descriptive statistics.

    Args:
        df (pd.DataFrame): The input DataFrame containing shape and radiomic features.
        patient_id (str or list, optional): Filter the DataFrame by Patient ID(s). Defaults to None (assumes all patients).
        structure_types (list, optional): List of structure types to include. Defaults to None (assumes all structure types).

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: A multi-index DataFrame with the count of structures for each patient grouped by structure type.
            - pd.DataFrame: A DataFrame containing the descriptive statistics of the structure counts for each structure type.
    """
    # Ensure the required columns are present
    if 'Patient ID' not in df.columns or 'Structure type' not in df.columns:
        raise ValueError("The DataFrame must contain 'Patient ID' and 'Structure type' columns.")

    # Filter the DataFrame by Patient ID if provided
    if patient_id is not None:
        if isinstance(patient_id, list):
            df = df[df['Patient ID'].isin(patient_id)]
        else:
            df = df[df['Patient ID'] == patient_id]

    # Filter the DataFrame by structure types if provided
    if structure_types is not None:
        if isinstance(structure_types, list):
            df = df[df['Structure type'].isin(structure_types)]
        else:
            df = df[df['Structure type'] == structure_types]

    # Group by structure type and calculate counts for each patient
    structure_counts = (
        df.groupby(['Structure type', 'Patient ID'])
        .size()
        .reset_index(name='Count')
    )

    # Pivot the counts to create a multi-index DataFrame
    structure_counts_pivot = structure_counts.pivot_table(
        index='Patient ID',
        columns='Structure type',
        values='Count',
        fill_value=0
    )

    # Calculate descriptive statistics for each structure type
    structure_statistics = structure_counts.groupby('Structure type')['Count'].describe(percentiles=[0.05, 0.95])

    return structure_counts_pivot, structure_statistics


def cumulative_dil_volume_stats(patient_ids, df):
    """
    Calculate the mean and standard deviation of the cumulative DIL ref volumes on a per patient basis.
    
    For each patient in the provided list, the function sums the 'Volume' values for rows
    where 'Structure type' is 'DIL ref'. It then computes the mean and standard deviation
    of these cumulative volumes across all patients.

    Parameters:
        patient_ids (list): List of patient identifiers to filter the data.
        df (pd.DataFrame): DataFrame containing the radiomic features with columns such as:
                           'Patient ID', 'Structure type', and 'Volume'.
    
    Returns:
        tuple: A tuple (mean_volume, std_volume) where:
            - mean_volume (float): Mean cumulative volume of DIL ref across the patients.
            - std_volume (float): Standard deviation of the cumulative volumes.
    
    Example:
        If patient 1 has three DIL entries with volumes 100, 200, 300,
        the cumulative volume for patient 1 is 600.
    """
    # Filter DataFrame for the specified patient IDs
    df_patients = df[df['Patient ID'].isin(patient_ids)]
    
    # Filter for rows that represent DIL ref entries
    df_dil = df_patients[df_patients['Structure type'] == 'DIL ref']
    
    # Group by 'Patient ID' and sum the 'Volume' column to obtain cumulative volume per patient
    cumulative_volume = df_dil.groupby('Patient ID')['Volume'].sum()
    
    # Calculate the mean and standard deviation across patients
    mean_volume = cumulative_volume.mean()
    std_volume = cumulative_volume.std()
    
    return mean_volume, std_volume