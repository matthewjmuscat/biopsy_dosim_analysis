from pingouin import compute_effsize
import pandas as pd
import numpy as np
from typing import Optional, List
import os


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




def compute_dose_differences_vectorized(df, column_name: str = 'Dose (Gy)') -> pd.DataFrame:
    df = df.copy()

    # Rename original for clarity
    df.rename(columns={'Voxel index': 'voxel_idx', column_name: 'dose'}, inplace=True)

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





def _concat_lists(series):
    return [x for sublist in series for x in sublist]

def create_diff_stats_dataframe(
    result_df: pd.DataFrame,
    patient_id_col: str,
    bx_index_col: str,
    bx_id_col: str,
    voxel_index_col: str,
    dose_col: str,
    trial_num_col: str = 'MC trial num',
    output_dir: Optional[str] = None,
    csv_name_stats_out: Optional[str] = None,
    csv_name_diffs_out: Optional[str] = None,
    csv_name_patient_pooled_out: Optional[str] = None,
    csv_name_cohort_pooled_out: Optional[str] = None
) -> pd.DataFrame:
    """
    For each biopsy (patient_id_col, bx_index_col, bx_id_col) and each unique pair of voxels
    in that biopsy, compute the paired differences (dose1 - dose2) across trials and then
    return a DataFrame of summary statistics on those differences:
      - count, mean, std, min, 5th, 25th, 50th, 75th, 95th, max

    Parameters
    ----------
    result_df : pd.DataFrame
        Input table with one row per (voxel, trial).
    patient_id_col : str
        Column name for patient ID.
    bx_index_col : str
        Column name for biopsy index.
    bx_id_col : str
        Column name for biopsy ID.
    voxel_index_col : str
        Column name for voxel index.
    dose_col : str
        Column name for the dose values.
    trial_num_col : str, default 'MC trial num'
        Column name for the trial‐alignment key.
    output_dir : str, optional
        Directory to save CSV (if csv_name also provided).
    csv_name : str, optional
        Filename for output CSV.

    Returns
    -------
    pd.DataFrame
        One row per (patient_id, bx_index, bx_id, voxel1, voxel2) with columns:
        ['count', 'mean', 'std', 'min', '5th', '25th', '50th', '75th', '95th', 'max'].
    """
    rows_statistics = []
    rows_diffs_and_abs_diffs = []
    # group per biopsy
    grp = result_df.groupby([patient_id_col, bx_index_col, bx_id_col])
    for (pid, bxi, bxid), sub in grp:
        voxels = sorted(sub[voxel_index_col].unique())
        # pre‐slice subframes & sort once
        sub_sorted = sub.sort_values(trial_num_col)
        for i, v1 in enumerate(voxels):
            for v2 in voxels[i+1:]:
                d1 = sub_sorted.loc[
                    sub_sorted[voxel_index_col] == v1, dose_col
                ].values
                d2 = sub_sorted.loc[
                    sub_sorted[voxel_index_col] == v2, dose_col
                ].values

                # pair up to the shorter length
                n = min(len(d1), len(d2))
                if n < 1:
                    continue

                diffs = d1[:n] - d2[:n]
                abs_diffs = np.abs(diffs)

                rows_diffs_and_abs_diffs.append({
                    patient_id_col: pid,
                    bx_index_col: bxi,
                    bx_id_col: bxid,
                    'voxel1': v1,
                    'voxel2': v2,
                    'dose_diffs': diffs,
                    'abs_dose_diffs': abs_diffs
                })



                pct = np.percentile(diffs, [5,25,50,75,95])

                rows_statistics.append({
                    patient_id_col: pid,
                    bx_index_col: bxi,
                    bx_id_col: bxid,
                    'voxel1': v1,
                    'voxel2': v2,
                    'count':       n,
                    'mean_diff':   diffs.mean(),
                    'std_diff':    diffs.std(ddof=1),
                    'min_diff':    diffs.min(),
                    '5th_diff':         pct[0],
                    '25th_diff':        pct[1],
                    '50th_diff':        pct[2],
                    '75th_diff':        pct[3],
                    '95th_diff':        pct[4],
                    'max_diff':    diffs.max(),
                    'mean_abs_diff':   abs_diffs.mean(),
                    'std_abs_diff':    abs_diffs.std(ddof=1),
                    'min_abs_diff':    abs_diffs.min(),
                    '5th_abs_diff':         np.percentile(abs_diffs, 5),
                    '25th_abs_diff':        np.percentile(abs_diffs, 25),
                    '50th_abs_diff':        np.percentile(abs_diffs, 50),
                    '75th_abs_diff':        np.percentile(abs_diffs, 75),
                    '95th_abs_diff':        np.percentile(abs_diffs, 95),
                    'max_abs_diff':    abs_diffs.max()
                })

    

    statistics_out_df = pd.DataFrame(rows_statistics)

    diffs_df_out = pd.DataFrame(rows_diffs_and_abs_diffs)


    # --- build counts that depend on voxel pairings ---
    _pair = ['voxel1', 'voxel2']
    _biopsy_keys = [patient_id_col, bx_index_col, bx_id_col]

    # Count unique biopsies/patients contributing to each voxel pair (cohort-wide)
    _unique_biopsies_per_pair = (
        diffs_df_out[_biopsy_keys + _pair]
        .drop_duplicates()
    )
    n_biopsies_per_pair = _unique_biopsies_per_pair.groupby(_pair).size()  # MultiIndex Series
    n_patients_per_pair = _unique_biopsies_per_pair.groupby(_pair)[patient_id_col].nunique()

    # Count unique biopsies contributing to each (patient, voxel1, voxel2)
    n_biopsies_per_patient_pair = (
        _unique_biopsies_per_pair
        .groupby([patient_id_col] + _pair)
        .size()
    )


    # group by patient and find summary statistics of mean diff and absolute mean diff between voxel pairs
    # then save to a csv file if output_dir and csv_name_stats_out are provided

    patient_pooled = diffs_df_out.groupby(['Patient ID', 'voxel1', 'voxel2']).agg({
            'dose_diffs': _concat_lists,
            'abs_dose_diffs': _concat_lists}).reset_index()
    
    cohort_pooled = diffs_df_out.groupby(['voxel1', 'voxel2']).agg({
            'dose_diffs': _concat_lists,
            'abs_dose_diffs': _concat_lists}).reset_index()
    


    def _stats_from_lists(diffs_list, abs_list):
        diffs = np.asarray(diffs_list, dtype=float)
        abs_diffs = np.asarray(abs_list, dtype=float)
        n = diffs.size

        if n:
            pct = np.percentile(diffs, [5, 25, 50, 75, 95])
            pct_abs = np.percentile(abs_diffs, [5, 25, 50, 75, 95])
        else:
            pct = pct_abs = [np.nan]*5

        return {
            'count':            int(n),
            'mean_diff':        diffs.mean() if n else np.nan,
            'std_diff':         diffs.std(ddof=1) if n > 1 else np.nan,
            'min_diff':         diffs.min() if n else np.nan,
            '5th_diff':         pct[0],
            '25th_diff':        pct[1],
            '50th_diff':        pct[2],
            '75th_diff':        pct[3],
            '95th_diff':        pct[4],
            'max_diff':         diffs.max() if n else np.nan,
            'mean_abs_diff':    abs_diffs.mean() if n else np.nan,
            'std_abs_diff':     abs_diffs.std(ddof=1) if n > 1 else np.nan,
            'min_abs_diff':     abs_diffs.min() if n else np.nan,
            '5th_abs_diff':     pct_abs[0],
            '25th_abs_diff':    pct_abs[1],
            '50th_abs_diff':    pct_abs[2],
            '75th_abs_diff':    pct_abs[3],
            '95th_abs_diff':    pct_abs[4],
            'max_abs_diff':     abs_diffs.max() if n else np.nan,
        }

    # ----- per-patient pooled (across all biopsies within a patient) -----
    patient_stats = patient_pooled.apply(
        lambda r: pd.Series({
            'pid':        r['Patient ID'],
            # biopsies this patient contributed for THIS voxel pair
            'n_biopsies': int(n_biopsies_per_patient_pair.get((r['Patient ID'], r['voxel1'], r['voxel2']), 0)),
            'voxel1':     r['voxel1'],
            'voxel2':     r['voxel2'],
            **_stats_from_lists(r['dose_diffs'], r['abs_dose_diffs'])
        }),
        axis=1
    )

    # ----- cohort pooled (across all patients & biopsies) -----
    cohort_stats = cohort_pooled.apply(
        lambda r: pd.Series({
            # patients/biopsies contributing to THIS voxel pair (cohort-wide)
            'n_patients': int(n_patients_per_pair.get((r['voxel1'], r['voxel2']), 0)),
            'n_biopsies': int(n_biopsies_per_pair.get((r['voxel1'], r['voxel2']), 0)),
            'voxel1':     r['voxel1'],
            'voxel2':     r['voxel2'],
            **_stats_from_lists(r['dose_diffs'], r['abs_dose_diffs'])
        }),
        axis=1
    )


    # Optional: set column order explicitly
    cols_patient = [
        'pid','n_biopsies','voxel1','voxel2','count',
        'mean_diff','std_diff','min_diff','5th_diff','25th_diff','50th_diff','75th_diff','95th_diff','max_diff',
        'mean_abs_diff','std_abs_diff','min_abs_diff','5th_abs_diff','25th_abs_diff','50th_abs_diff','75th_abs_diff','95th_abs_diff','max_abs_diff'
    ]
    cols_cohort = [
        'n_patients','n_biopsies','voxel1','voxel2','count',
        'mean_diff','std_diff','min_diff','5th_diff','25th_diff','50th_diff','75th_diff','95th_diff','max_diff',
        'mean_abs_diff','std_abs_diff','min_abs_diff','5th_abs_diff','25th_abs_diff','50th_abs_diff','75th_abs_diff','95th_abs_diff','max_abs_diff'
    ]
    patient_stats = patient_stats[cols_patient]
    cohort_stats = cohort_stats[cols_cohort]

    if output_dir and csv_name_patient_pooled_out:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, csv_name_patient_pooled_out)
        patient_stats.to_csv(path, index=False)
        print(f"Saved patient-pooled diff‐summary CSV to {path}")

    if output_dir and csv_name_cohort_pooled_out:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, csv_name_cohort_pooled_out)
        cohort_stats.to_csv(path, index=False)
        print(f"Saved cohort-pooled diff‐summary CSV to {path}")

    if output_dir and csv_name_stats_out:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, csv_name_stats_out)
        statistics_out_df.to_csv(path, index=False)
        print(f"Saved diff‐summary CSV to {path}")

    if output_dir and csv_name_diffs_out:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, csv_name_diffs_out)
        diffs_df_out.to_csv(path, index=False)
        print(f"Saved diffs CSV to {path}")

    return statistics_out_df, diffs_df_out, patient_stats, cohort_stats