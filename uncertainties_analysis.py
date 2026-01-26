import pandas as pd

def compute_statistics_by_structure_type(df, columns, patient_uids=None):
    """
    Computes statistics for each unique value of 'Structure type' for the specified columns.

    Parameters:
        df (pd.DataFrame): Input DataFrame with the required columns.
        columns (list): List of column names for which to compute statistics.
        patient_uids (list or None): List of Patient UIDs to filter by. If None, all patients are included.

    Returns:
        pd.DataFrame: Multi-index DataFrame with statistics for each 'Structure type'.
    """
    # Validate input columns
    required_columns = [
        "Patient UID", "Structure ID", "Structure type",
        "Structure index", "Frame of reference", "mu (X)", "mu (Y)", "mu (Z)",
        "sigma (X)", "sigma (Y)", "sigma (Z)", "Dilations mu (XY)", "Dilations mu (Z)",
        "Dilations sigma (XY)", "Dilations sigma (Z)", "Rotations mu (X)", "Rotations mu (Y)",
        "Rotations mu (Z)", "Rotations sigma (X)", "Rotations sigma (Y)", "Rotations sigma (Z)"
    ]
    if not set(required_columns).issubset(df.columns):
        raise ValueError("Input DataFrame does not contain the required columns.")
    if not set(columns).issubset(df.columns):
        raise ValueError("Specified columns are not present in the DataFrame.")

    # Filter by patient UIDs if provided
    if patient_uids is not None:
        df = df[df["Patient UID"].isin(patient_uids)]

    # Group by 'Structure type'
    grouped = df.groupby("Structure type")

    # Compute statistics for the specified columns
    stats = []
    for structure_type, group in grouped:
        for col in columns:
            col_stats = group[col].describe(percentiles=[0.05, 0.95]).to_dict()
            col_stats["kurtosis"] = group[col].kurt()
            col_stats["skewness"] = group[col].skew()
            stats.append({
                "Structure type": structure_type,
                "Column": col,
                **col_stats
            })

    # Convert the statistics list to a DataFrame
    stats_df = pd.DataFrame(stats)
    stats_df.set_index(["Structure type", "Column"], inplace=True)
    return stats_df