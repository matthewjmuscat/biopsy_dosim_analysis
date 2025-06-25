from pingouin import compute_effsize
import pandas as pd
import numpy as np

def create_eff_size_dataframe(result_df, patient_id_col, bx_index_col, bx_id_col, voxel_index_col, dose_col, eff_size="cohen", paired_bool=False):
    """
    Create a single DataFrame containing effect size values (e.g., Cohen's d) for voxel pairs
    across all unique combinations of Patient ID, Bx index, and Bx ID.
    
    Args:
        result_df (pd.DataFrame): Input DataFrame with voxel data.
        patient_id_col (str): Column name for Patient ID.
        bx_index_col (str): Column name for Bx index.
        bx_id_col (str): Column name for Bx ID.
        voxel_index_col (str): Column name for voxel index.
        dose_col (str): Column name for dose values.
        eff_size (str): Effect size type, e.g., "cohen", "hedges", "glass". Defaults to "cohen".
        valid_efftypes = ['cohen', 'hedges', 'AUC', 'CLES', 'mean_diff'], not technically complete list but woens that are appropriate for dosimetry

    
    Returns:
        pd.DataFrame: A single DataFrame containing effect size values with columns for:
                      Patient ID, Bx index, Bx ID, Voxel 1, Voxel 2, and the effect size value.
    """
    results = []
    
    # Group by Patient ID, Bx index, and Bx ID
    grouped = result_df.groupby([patient_id_col, bx_index_col, bx_id_col])
    
    for (patient_id, bx_index, bx_id), group in grouped:
        # Get unique voxel indices
        voxels = sorted(group[voxel_index_col].unique())
        
        # Loop through voxel pairs
        for i, voxel1 in enumerate(voxels):
            for j, voxel2 in enumerate(voxels):
                if i >= j:  # Avoid redundant calculations
                    continue
                
                # Extract dose values for both voxels
                group1 = group[group[voxel_index_col] == voxel1][[dose_col, 'MC trial num']]
                group2 = group[group[voxel_index_col] == voxel2][[dose_col, 'MC trial num']]

                # Sort by 'MC trial num' to ensure alignment
                group1_sorted = group1.sort_values('MC trial num')
                group2_sorted = group2.sort_values('MC trial num')

                # extract the dose values
                dose1 = group1_sorted[dose_col].values
                dose2 = group2_sorted[dose_col].values

                
                # Calculate effect size using pingouin
                if len(dose1) > 1 and len(dose2) > 1:  # Ensure sufficient data
                    if eff_size == "mean_diff":
                        if paired_bool:
                            if len(dose1) != len(dose2):
                                raise ValueError("For paired mean difference, both groups must have the same number of observations.")
                            eff_value = np.mean(dose1 - dose2)  # paired

                        else:
                            eff_value = np.mean(dose1) - np.mean(dose2)  # unpaired
                    else:
                        eff_value = compute_effsize(dose1, dose2, eftype=eff_size, paired = paired_bool)

                    #eff_value = compute_effsize(group1, group2, eftype=eff_size)
                    results.append({
                        patient_id_col: patient_id,
                        bx_index_col: bx_index,
                        bx_id_col: bx_id,
                        "Voxel 1": voxel1,
                        "Voxel 2": voxel2,
                        "Effect Size": eff_value,
                        "Num Observations Voxel 1": len(dose1),
                        "Num Observations Voxel 2": len(dose2),
                        "Effect Size Type": eff_size
                    })
    
    # Convert results to a DataFrame
    eff_size_df = pd.DataFrame(results)
    return eff_size_df



# quite slow for very large datasets, but it is a brute force method to compute dose differences between voxel pairs
def compute_dose_differences(df):
    results = []

    # Group by each biopsy (Patient ID + Bx index)
    biopsy_groups = df.groupby(['Patient ID', 'Bx index'])

    for (patient_id, bx_index), biopsy_df in biopsy_groups:
        # Find voxel index range for this biopsy
        min_voxel = biopsy_df['Voxel index'].min()
        max_voxel = biopsy_df['Voxel index'].max()
        max_distance = max_voxel - min_voxel

        # Group further by MC trial
        for mc_trial, trial_df in biopsy_df.groupby('MC trial num'):
            # Set index for fast lookup by Voxel index
            trial_df = trial_df.set_index('Voxel index').sort_index()

            available_voxels = trial_df.index.values

            for length_scale in range(1, max_distance + 1):
                for voxel in available_voxels:
                    paired_voxel = voxel + length_scale

                    # Ensure valid and symmetric-safe comparison
                    if paired_voxel in trial_df.index:
                        dose1 = trial_df.loc[voxel, 'Dose (Gy)']
                        dose2 = trial_df.loc[paired_voxel, 'Dose (Gy)']
                        dose_diff = dose2 - dose1

                        results.append({
                            'Patient ID': patient_id,
                            'Bx index': bx_index,
                            'MC trial num': mc_trial,
                            'Voxel 1': voxel,
                            'Voxel 2': paired_voxel,
                            'Length scale': length_scale,
                            'Dose diff (Gy)': dose_diff
                        })

    return pd.DataFrame(results)




def compute_dose_differences_vectorized(df):
    df = df.copy()

    # Rename original for clarity
    df.rename(columns={'Voxel index': 'voxel_idx', 'Dose (Gy)': 'dose'}, inplace=True)

    # Determine max length scale per biopsy
    biopsy_range = (
        df.groupby(['Patient ID', 'Bx index'])['voxel_idx']
        .agg(['min', 'max'])
        .assign(max_diff=lambda x: x['max'] - x['min'])
    )

    max_global_distance = biopsy_range['max_diff'].max()

    all_results = []

    for length_scale in range(1, max_global_distance + 1):
        # Left side: original voxels
        left = df[['Patient ID', 'Bx index', 'MC trial num', 'voxel_idx', 'dose']].copy()
        left.rename(columns={
            'voxel_idx': 'voxel_1',
            'dose': 'dose_1'
        }, inplace=True)

        # Right side: shifted voxels
        right = df[['Patient ID', 'Bx index', 'MC trial num', 'voxel_idx', 'dose']].copy()
        right['voxel_idx'] -= length_scale  # look ahead by length scale
        right.rename(columns={
            'voxel_idx': 'voxel_1',  # join key
            'dose': 'dose_2',
        }, inplace=True)
        right = right[['Patient ID', 'Bx index', 'MC trial num', 'voxel_1', 'dose_2']]

        # Perform the join
        merged = pd.merge(
            left, right,
            on=['Patient ID', 'Bx index', 'MC trial num', 'voxel_1'],
            how='inner'
        )

        merged['voxel_2'] = merged['voxel_1'] + length_scale
        merged['length_scale'] = length_scale
        merged['dose_diff'] = merged['dose_2'] - merged['dose_1']

        all_results.append(merged)

    # Combine all results
    result_df = pd.concat(all_results, ignore_index=True)

    # generate an absolute value dose difference column
    result_df['dose_diff_abs'] = result_df['dose_diff'].abs()

    return result_df[['Patient ID', 'Bx index', 'MC trial num',
                      'voxel_1', 'voxel_2', 'length_scale', 'dose_diff', 'dose_diff_abs']]