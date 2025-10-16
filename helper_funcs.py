from pingouin import compute_effsize
import pandas as pd
import numpy as np
from typing import Optional, List, Union, Dict, Iterable, Tuple
import os
import re
from scipy.stats import skew, kurtosis


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






### compute DVH metrics per trial 
def compute_dvh_metrics_per_trial(
    df: pd.DataFrame,
    d_perc_list: List[Union[int, float]],
    v_perc_list: List[Union[int, float]],
    *,
    # How to define the reference dose for V_Y% thresholds:
    # - EITHER pass a single float (same ref for all groups),
    # - OR pass the name of a column in df with the per-voxel ref dose (it must be constant within each group),
    # - OR pass a dict {(patient_id, bx_index): ref_dose_gy}.
    ref_dose_gy: Optional[float] = None,
    ref_dose_col: Optional[str] = None,
    ref_dose_map: Optional[Dict[tuple, float]] = None,
    # I/O
    output_dir: Optional[str] = None,
    csv_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute DVH metrics (D_X% and V_Y%) per MC trial for each (Patient ID, Bx index) pair.
    Keeps the metadata columns: 'Simulated bool', 'Simulated type', 'Bx refnum', 'Bx ID'.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least:
        ['Voxel index','MC trial num','Dose (Gy)','Patient ID','Bx index','Bx ID',
         'Simulated bool','Simulated type','Bx refnum']
    d_perc_list : list[int|float]
        X values for D_X% (e.g., [2, 50, 98]). D_X% is computed as the dose at
        quantile q = 1 - X/100 (i.e., near-maximum for small X).
    v_perc_list : list[int|float]
        Y values for V_Y% (e.g., [100, 125, 150, 200, 300]).
        V_Y% is percent of voxels with dose >= (Y/100)*reference_dose.
    ref_dose_gy : float, optional
        A cohort-wide reference dose (Gy) used for all groups when computing V_Y%.
    ref_dose_col : str, optional
        Name of a column in df giving a per-voxel reference dose; must be constant
        within each (Patient ID, Bx index, MC trial) group (we’ll validate).
    ref_dose_map : dict[(patient_id, bx_index) -> float], optional
        Mapping that provides reference dose per biopsy (used if provided).
    output_dir : str, optional
        If provided with csv_name, results are saved as CSV.
    csv_name : str, optional
        Filename for the CSV.

    Returns
    -------
    pd.DataFrame
        One row per (Patient ID, Bx index, MC trial num) with metadata columns and
        DVH columns:
          - D_{X%} (Gy) for each X in d_perc_list
          - V_{Y%} (%) for each Y in v_perc_list
    """
    required_cols = [
        'Voxel index', 'MC trial num', 'Dose (Gy)', 'Patient ID', 'Bx index',
        'Bx ID', 'Simulated bool', 'Simulated type', 'Bx refnum'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Make sure dose is numeric and drop NaNs on dose only (other metadata may be object)
    work = df.copy()
    work['Dose (Gy)'] = pd.to_numeric(work['Dose (Gy)'], errors='coerce')
    work = work.dropna(subset=['Dose (Gy)'])

    # Grouping keys (trial-level within biopsy)
    group_keys = ['Patient ID', 'Bx index', 'MC trial num']

    # We will carry these metadata columns through by taking a unique value per group
    meta_cols = ['Simulated bool', 'Simulated type', 'Bx refnum', 'Bx ID']

    # Helper to resolve the reference dose (Gy) for V_Y% in a given group
    def resolve_ref_dose(group_df: pd.DataFrame) -> float:
        # Priority: ref_dose_map > ref_dose_col > ref_dose_gy
        if ref_dose_map is not None:
            key = (group_df['Patient ID'].iloc[0], group_df['Bx index'].iloc[0])
            if key not in ref_dose_map:
                raise ValueError(f"ref_dose_map missing key {key}")
            return float(ref_dose_map[key])

        if ref_dose_col is not None:
            if ref_dose_col not in group_df.columns:
                raise ValueError(f"ref_dose_col '{ref_dose_col}' not found in dataframe.")
            vals = pd.to_numeric(group_df[ref_dose_col], errors='coerce').dropna().unique()
            if len(vals) == 0:
                raise ValueError(f"ref_dose_col '{ref_dose_col}' has no numeric values in group.")
            if len(vals) > 1:
                raise ValueError(
                    f"ref_dose_col '{ref_dose_col}' is not constant within a group: found {vals} "
                    f"for Patient ID={group_df['Patient ID'].iloc[0]}, Bx index={group_df['Bx index'].iloc[0]}, "
                    f"MC trial num={group_df['MC trial num'].iloc[0]}"
                )
            return float(vals[0])

        if ref_dose_gy is not None:
            return float(ref_dose_gy)

        # If nothing provided, this is ambiguous—explicit error is safer than guessing.
        raise ValueError(
            "No reference dose provided for V_Y% computation. "
            "Pass ref_dose_gy (float) or ref_dose_col (str) or ref_dose_map (dict)."
        )

    # Per-group computation
    records = []
    for (pid, bx_idx, trial), g in work.groupby(group_keys, sort=False):
        doses = g['Dose (Gy)'].to_numpy()
        if doses.size == 0:
            continue

        # D_X% metrics (dose at quantile q = 1 - X/100, ascending)
        d_metrics = {}
        for X in d_perc_list:
            q = 1.0 - float(X) / 100.0
            q = min(max(q, 0.0), 1.0)  # clamp
            d_val = float(np.quantile(doses, q, method="linear"))
            d_metrics[f"D_{int(X)}% (Gy)"] = d_val

        # V_Y% metrics (percent of voxels >= Y% of reference dose)
        ref_dose = resolve_ref_dose(g)
        v_metrics = {}
        for Y in v_perc_list:
            thr = (float(Y) / 100.0) * ref_dose
            pct = 100.0 * (doses >= thr).mean()
            v_metrics[f"V_{int(Y)}% (%)"] = float(pct)

        # Metadata: ensure single values per group
        meta = {c: g[c].iloc[0] for c in meta_cols}

        rec = {
            'Patient ID': pid,
            'Bx index': bx_idx,
            'MC trial num': trial,
            **meta,
            **d_metrics,
            **v_metrics,
        }
        records.append(rec)

    out = pd.DataFrame.from_records(records)

    # Optional save
    if output_dir and csv_name:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, csv_name)
        out.to_csv(path, index=False)
        print(f"DVH per-trial metrics saved to: {path}")

    return out




# IMPORTANT: This function is only valid if the fed dataframe only has 1 sample point per voxel index per trial, otherwise overweighting can occur.
def compute_dvh_metrics_per_trial_vectorized(
        df: pd.DataFrame,
        d_perc_list: List[Union[int, float]],
        v_perc_list: List[Union[int, float]],
        *,
        # Reference dose for V_Y%:
        #  - ref_dose_gy: one scalar for all groups, OR
        #  - ref_dose_col: column in df, constant per group, OR
        #  - ref_dose_map: {(Patient ID, Bx index): ref_gy}
        ref_dose_gy: Optional[float] = None,
        ref_dose_col: Optional[str] = None,
        ref_dose_map: Optional[Dict[tuple, float]] = None,
        # I/O
        output_dir: Optional[str] = None,
        csv_name: Optional[str] = None,
    ) -> pd.DataFrame:
    """
    Vectorized DVH metrics per MC trial for each (Patient ID, Bx index) group.
    Produces D_X% (Gy) and V_Y% (%) columns, one row per (Patient ID, Bx index, MC trial num),
    carrying metadata: 'Simulated bool', 'Simulated type', 'Bx refnum', 'Bx ID'.

    Assumptions:
      - D_X% = dose at quantile q = 1 - X/100 (e.g., D2% ~ 98th percentile).
      - V_Y% = 100 * mean( dose >= (Y/100) * reference_dose ).
        Reference can be scalar, column (constant within group), or map by (Patient ID, Bx index).

    Returns:
      DataFrame with one row per (Patient ID, Bx index, MC trial num) including DVH metrics + metadata.
    """
    required = [
        'Voxel index', 'MC trial num', 'Dose (Gy)', 'Patient ID', 'Bx index',
        'Bx ID', 'Simulated bool', 'Simulated type', 'Bx refnum'
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Work copy; ensure numeric dose
    work = df.copy()
    work['Dose (Gy)'] = pd.to_numeric(work['Dose (Gy)'], errors='coerce')
    work = work.dropna(subset=['Dose (Gy)'])

    keys = ['Patient ID', 'Bx index', 'MC trial num']
    meta_cols = ['Simulated bool', 'Simulated type', 'Bx refnum', 'Bx ID']

    # -------------------------
    # 1) Vectorized D_X% using groupby.quantile at once
    # -------------------------
    # q = 1 - X/100
    qs = sorted(set(max(0.0, min(1.0, 1.0 - float(x) / 100.0)) for x in d_perc_list))
    # Compute all quantiles in one call
    qdf = (
        work.groupby(keys)['Dose (Gy)']
            .quantile(q=qs, interpolation='higher')  # pandas >=1.5: 'method' renamed to 'interpolation'
            .unstack(level=-1)                       # columns = quantiles
    )
    # Rename quantile columns to D_{X%} (Gy)
    q_map = {1.0 - q: int(round((1.0 - q) * 100)) for q in qdf.columns}  # map quantile -> X
    # But columns currently are the quantiles q themselves (floats). Build name map properly:
    col_name_map = {}
    for q in qdf.columns:
        X = int(round((1.0 - float(q)) * 100))
        col_name_map[q] = f"D_{X}% (Gy)"
    d_part = qdf.rename(columns=col_name_map)

    # Only keep requested X's (in case rounding produced extras)
    wanted_d_cols = [f"D_{int(x)}% (Gy)" for x in d_perc_list]
    d_part = d_part.reindex(columns=wanted_d_cols)

    # -------------------------
    # 2) Vectorized V_Y% using dose/reference ratios
    # -------------------------
    # Resolve per-row reference dose
    if ref_dose_map is not None:
        # Map by (Patient ID, Bx index)
        ref_series = work.set_index(['Patient ID', 'Bx index']).index.map(
            lambda t: ref_dose_map.get((t[0], t[1]), np.nan)
        )
        work['_ref_dose'] = ref_series.values
    elif ref_dose_col is not None:
        if ref_dose_col not in work.columns:
            raise ValueError(f"ref_dose_col '{ref_dose_col}' not in df.")
        work['_ref_dose'] = pd.to_numeric(work[ref_dose_col], errors='coerce')
    elif ref_dose_gy is not None:
        work['_ref_dose'] = float(ref_dose_gy)
    else:
        raise ValueError("Provide one of ref_dose_gy, ref_dose_col, or ref_dose_map for V_Y%.")

    if work['_ref_dose'].isna().any():
        bad = work.loc[work['_ref_dose'].isna(), ['Patient ID', 'Bx index']].drop_duplicates()
        raise ValueError(f"Missing reference dose for some rows; first few:\n{bad.head()}")

    # Ratio per row; safe for zero ref? Guard:
    if (work['_ref_dose'] == 0).any():
        raise ValueError("Reference dose is zero for some rows; cannot form V_Y% thresholds.")
    work['_dose_ratio'] = work['Dose (Gy)'] / work['_ref_dose']

    # For each Y, compute % of voxels with ratio >= Y/100, vectorized:
    v_frames = []
    for Y in v_perc_list:
        thr = float(Y) / 100.0
        meets = (work['_dose_ratio'] >= thr)
        # mean() per group gives fraction meeting threshold
        v_col = f"V_{int(Y)}% (%)"
        v_series = 100.0 * meets.groupby(work[keys].apply(tuple, axis=1)).mean()
        v_series.index = pd.MultiIndex.from_tuples(v_series.index, names=keys)
        v_frames.append(v_series.rename(v_col))
    if v_frames:
        v_part = pd.concat(v_frames, axis=1)
    else:
        v_part = pd.DataFrame(index=d_part.index)  # empty, keep alignment

    # -------------------------
    # 3) Metadata per group (first value)
    # -------------------------
    meta_part = (
        work.groupby(keys)[meta_cols].first()
    )

    # -------------------------
    # 4) Assemble output
    # -------------------------
    out = pd.concat([meta_part, d_part, v_part], axis=1).reset_index()

    # Optional save
    if output_dir and csv_name:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, csv_name)
        out.to_csv(path, index=False)
        print(f"DVH per-trial metrics (vectorized) saved to: {path}")

    return out



# Shorten metric names for legacy output
def _short_metric_name(colname: str) -> str:
    # 'D_2% (Gy)' → 'D_2', 'V_125% (%)' → 'V_125' (keeps decimals)
    m = re.match(r'^(D|V)_(\d+(?:\.\d+)?)%.*$', colname)
    if not m:
        return colname
    prefix, num = m.groups()
    f = float(num)
    return f"{prefix}_{int(f) if f.is_integer() else f:g}"
# Summarize per-trial DVH metrics into legacy schema (+ 'Nominal') with ONE row per biopsy
def build_dvh_summary_one_row_per_biopsy(
    per_trial_df: pd.DataFrame,
    *,
    nominal_trial: int = 0,
) -> pd.DataFrame:
    """
    Summarize per-trial DVH metrics into legacy schema (+ 'Nominal') with
    ONE row per (Patient ID, Bx ID, Struct type, Simulated bool, Simulated type) per metric.
    """
    df = per_trial_df.copy()

    # Normalize metadata to legacy schema
    if "Dicom ref num" not in df.columns and "Bx refnum" in df.columns:
        df = df.rename(columns={"Bx refnum": "Dicom ref num"})
    if "Struct type" not in df.columns:
        df["Struct type"] = "Bx ref"
    if "Struct index" not in df.columns:
        df["Struct index"] = df.get("Bx index", pd.Series(index=df.index, dtype="float64"))

    # Identify metric columns and coerce to numeric
    metric_cols = [c for c in df.columns if c.startswith("D_") or c.startswith("V_")]
    if not metric_cols:
        raise ValueError("No DVH metric columns found (expected columns starting with 'D_' or 'V_').")
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Grouping keys to ensure exactly one row per biopsy
    group_keys = ["Patient ID","Bx ID","Struct type","Simulated bool","Simulated type"]

    rows = []
    # Iterate per metric to keep memory predictable
    for mcol in metric_cols:
        mname = _short_metric_name(mcol)

        for keys, sub in df.groupby(group_keys, observed=False):
            # Pull a stable representative for display-only columns
            first = sub.iloc[0]
            dicom_ref = first.get("Dicom ref num", np.nan)
            struct_index = first.get("Struct index", first.get("Bx index", np.nan))

            # Values across trials for this biopsy & metric
            arr = sub[mcol].to_numpy(dtype=float)
            arr = arr[~np.isnan(arr)]
            n = arr.size

            if n == 0:
                mean = std = sem = vmin = vmax = q05 = q25 = q50 = q75 = q95 = iqr = ipr90 = np.nan
                sk = ku = np.nan
            else:
                mean = float(np.nanmean(arr))
                std  = float(np.nanstd(arr, ddof=0))          # population std → aligns with your SEM
                sem  = float(std/np.sqrt(n))
                vmin = float(np.nanmin(arr))
                vmax = float(np.nanmax(arr))
                q05, q25, q50, q75, q95 = np.nanpercentile(arr, [5, 25, 50, 75, 95])
                iqr   = float(q75 - q25)
                ipr90 = float(q95 - q05)
                var = float(np.nanvar(arr))
                if np.isfinite(var) and var > 1e-12:
                    sk = float(skew(arr, bias=True, nan_policy="omit"))
                    ku = float(kurtosis(arr, bias=True, fisher=True, nan_policy="omit"))
                else:
                    sk = ku = np.nan

            # Nominal from MC trial == nominal_trial
            nom_s = pd.to_numeric(
                sub.loc[sub["MC trial num"] == nominal_trial, mcol],
                errors="coerce"
            ).dropna()
            nominal = float(nom_s.iloc[0]) if not nom_s.empty else np.nan

            rows.append({
                "Patient ID": keys[0],
                "Metric": mname,
                "Bx ID": keys[1],
                "Struct type": keys[2],
                "Dicom ref num": dicom_ref,
                "Simulated bool": keys[3],
                "Simulated type": keys[4],
                "Struct index": struct_index,
                "Mean": mean,
                "STD": std,
                "SEM": sem,
                "Max": vmax,
                "Min": vmin,
                "Skewness": sk,
                "Kurtosis": ku,
                "Q05": q05,
                "Q25": q25,
                "Q50": q50,
                "Q75": q75,
                "Q95": q95,
                "IQR": iqr,
                "IPR90": ipr90,
                "Nominal": nominal,
            })

    out = pd.DataFrame.from_records(rows)

    # Exact legacy order (+ Nominal)
    desired_cols = [
        "Patient ID","Metric","Bx ID","Struct type","Dicom ref num",
        "Simulated bool","Simulated type","Struct index",
        "Mean","STD","SEM","Max","Min","Skewness","Kurtosis",
        "Q05","Q25","Q50","Q75","Q95","IQR","IPR90","Nominal",
    ]
    out = out.reindex(columns=desired_cols)

    return out








def build_cumulative_dvh_by_mc_trial_number_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-trial cumulative DVH curves.
    - One row per unique dose level (within group), reporting % volume receiving >= dose.
    - Equal-voxel weighting (each row in df is one voxel sample).

    Input df must contain (at least):
      ['Patient ID','Bx ID','Bx index','Simulated bool','Simulated type',
       'MC trial num','Dose (Gy)']

    Returns columns (exact order):
      ['Patient ID','Bx ID','Bx index','Simulated bool','Simulated type',
       'Percent volume','Dose (Gy)','MC trial']
    """
    required = [
        'Patient ID','Bx ID','Bx index','Simulated bool','Simulated type',
        'MC trial num','Dose (Gy)'
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df.copy()
    work['Dose (Gy)'] = pd.to_numeric(work['Dose (Gy)'], errors='coerce')
    work = work.dropna(subset=['Dose (Gy)'])

    keys = ['Patient ID','Bx index','MC trial num']
    meta_cols = ['Bx ID','Simulated bool','Simulated type']

    out_frames = []
    # Group per (patient, bx, trial)
    for (pid, bxi, trial), g in work.groupby(keys, sort=False):
        doses = np.sort(g['Dose (Gy)'].to_numpy(dtype=float))  # ascending
        if doses.size == 0:
            continue

        # Unique dose levels with counts, then form tail counts (>= dose)
        vals, cnts = np.unique(doses, return_counts=True)
        tail_cnts = cnts[::-1].cumsum()[::-1]  # cumulative from the right
        percent = (tail_cnts / doses.size) * 100.0

        first = g.iloc[0]
        tmp = pd.DataFrame({
            'Patient ID': pid,
            'Bx ID': first['Bx ID'],
            'Bx index': bxi,
            'Simulated bool': first['Simulated bool'],
            'Simulated type': first['Simulated type'],
            'Percent volume': percent.astype(float),
            'Dose (Gy)': vals.astype(float),
            'MC trial': trial,
        })
        out_frames.append(tmp)

    if not out_frames:
        return pd.DataFrame(columns=[
            'Patient ID','Bx ID','Bx index','Simulated bool','Simulated type',
            'Percent volume','Dose (Gy)','MC trial'
        ])

    out = pd.concat(out_frames, ignore_index=True)
    # Column order as requested
    out = out[['Patient ID','Bx ID','Bx index','Simulated bool','Simulated type',
               'Percent volume','Dose (Gy)','MC trial']]
    return out






def build_deltas_vs_gradient_df(
    nominal_deltas_df: pd.DataFrame,
    cohort_by_voxel_df: pd.DataFrame,
    *,
    gradient_top: str = 'Dose grad (Gy/mm)',   # top-level column name for gradient in cohort_by_voxel_df
    gradient_stat: str = 'nominal',            # sub-level under gradient_top: e.g. 'nominal', 'mean', ...
    meta_keep: Optional[Iterable[str]] = (
        'Voxel begin (Z)', 'Voxel end (Z)', 'Voxel index',
        'Patient ID', 'Bx ID', 'Bx index', 'Bx refnum',
        'Simulated bool', 'Simulated type'
    ),
    add_abs: bool = True,                      # add |Δ| columns
    add_log1p: bool = True,                    # add log1p(|Δ|) columns
    return_long: bool = False                  # also return tidy/long version
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Merge voxel-wise ΔD metrics with the per-voxel gradient for the same biopsy/voxel,
    preserving biopsy/patient metadata. Keys used for the join: (Patient ID, Bx index, Voxel index).

    Inputs
    ------
    nominal_deltas_df : output of compute_biopsy_nominal_deltas(...)
        Expected columns (MultiIndex):
            - meta_keep (with second-level '')
            - ('Dose (Gy) deltas', 'nominal_minus_mode')
            - ('Dose (Gy) deltas', 'nominal_minus_q50')
            - ('Dose (Gy) deltas', 'nominal_minus_mean')
    cohort_by_voxel_df : your multi-index per-voxel dataframe containing gradient stats
        Must contain (gradient_top, gradient_stat), e.g. ('Dose grad (Gy/mm)', 'nominal').

    Returns
    -------
    wide : pd.DataFrame
        One row per voxel with metadata, Δ columns, gradient column,
        and optional |Δ| / log1p(|Δ|) columns.
    long : Optional[pd.DataFrame]
        Tidy version (if return_long=True) with columns:
        ['Patient ID','Bx index','Bx ID','Voxel index','Grad (Gy/mm)',
         'Delta kind','Delta (Gy)','|Delta| (Gy)','log1p|Delta|']
    """
    # --- helpers to slice MultiIndex columns robustly ---
    def _need(df, pair):
        if pair not in df.columns:
            raise KeyError(f"Missing required column {pair} in dataframe.")
        return df.loc[:, pair]

    def _flatten_select(df, top_names):
        # keep (top, '') columns and flatten to single level names == top
        cols = [(t, '') for t in top_names if (t, '') in df.columns]
        out = df.loc[:, cols].copy()
        out.columns = [t for (t, _) in cols]
        return out

    # --- 1) Extract metadata from nominal_deltas_df and flatten ---
    meta_keep = tuple(meta_keep) if meta_keep is not None else ()
    meta_flat = _flatten_select(nominal_deltas_df, meta_keep)

    # Δ columns (ensure presence & flatten with nice names)
    delta_pairs = [
        ('Dose (Gy) deltas', 'nominal_minus_mode'),
        ('Dose (Gy) deltas', 'nominal_minus_q50'),
        ('Dose (Gy) deltas', 'nominal_minus_mean'),
    ]
    for p in delta_pairs:
        if p not in nominal_deltas_df.columns:
            raise KeyError(f"Missing delta column {p} in nominal_deltas_df.")
    deltas_flat = nominal_deltas_df.loc[:, delta_pairs].copy()
    deltas_flat.columns = ['Δ_mode (Gy)', 'Δ_median (Gy)', 'Δ_mean (Gy)']

    deltas_block = pd.concat([meta_flat, deltas_flat], axis=1)

    # --- 2) Pull gradient from cohort_by_voxel_df and flatten ---
    # keep same metadata keys from the source (to be safe if types differ)
    source_meta_flat = _flatten_select(cohort_by_voxel_df, meta_keep)
    grad_series = _need(cohort_by_voxel_df, (gradient_top, gradient_stat)).rename('Grad (Gy/mm)')

    grad_block = pd.concat([source_meta_flat, grad_series], axis=1)

    # --- 3) Standardize key dtypes before merge ---
    # merge keys
    keys = ['Patient ID', 'Bx index', 'Voxel index']
    for k in keys:
        if k in deltas_block.columns:
            deltas_block[k] = deltas_block[k].astype(str if k == 'Patient ID' else 'int64', errors='ignore')
        if k in grad_block.columns:
            grad_block[k] = grad_block[k].astype(str if k == 'Patient ID' else 'int64', errors='ignore')

    # --- 4) Merge on biopsy+voxel identity (inner keeps strict matches) ---
    wide = deltas_block.merge(grad_block, on=keys, how='inner')

    # --- 5) Optional derived columns ---
    if add_abs or add_log1p:
        for col in ['Δ_mode (Gy)', 'Δ_median (Gy)', 'Δ_mean (Gy)']:
            if add_abs:
                wide[f'|{col[1:]}' ] = wide[col].abs()     # e.g., '|_mode (Gy)' -> cleaner name below
                wide.rename(columns={f'|{col[1:]}': f'|{col[:-4]}| (Gy)'}, inplace=True)
            if add_log1p:
                abs_name = f'|{col[:-4]}| (Gy)'
                if abs_name not in wide.columns:
                    wide[abs_name] = wide[col].abs()
                wide[f'log1p|{col[:-4]}|'] = np.log1p(wide[abs_name])

    # --- 6) Long/tidy option for plotting/stats ---
    long = None
    if return_long:
        keep_id = ['Patient ID', 'Bx index', 'Bx ID', 'Voxel index', 'Grad (Gy/mm)']
        keep_id = [c for c in keep_id if c in wide.columns]
        long = wide.melt(
            id_vars=keep_id,
            value_vars=[c for c in ['Δ_mode (Gy)', 'Δ_median (Gy)', 'Δ_mean (Gy)'] if c in wide.columns],
            var_name='Delta kind',
            value_name='Delta (Gy)'
        )
        if add_abs:
            long['|Delta| (Gy)'] = long['Delta (Gy)'].abs()
        if add_log1p:
            if '|Delta| (Gy)' not in long.columns:
                long['|Delta| (Gy)'] = long['Delta (Gy)'].abs()
            long['log1p|Delta|'] = np.log1p(long['|Delta| (Gy)'])

    return (wide, long)





def build_deltas_vs_gradient_df_with_abs(
    nominal_deltas_df: pd.DataFrame,
    cohort_by_voxel_df: pd.DataFrame,
    *,
    zero_level_index_str: str = 'Dose (Gy)',           # which delta block to use
    gradient_top: str = 'Dose grad (Gy/mm)',           # top-level gradient column
    gradient_stats: Union[str, Iterable[str]] = 'nominal',  # one or many: 'nominal','mean','median/Q50','mode'
    gradient_stat: Optional[str] = None,               # alias (if provided, overrides gradient_stats when str)
    meta_keep: Optional[Iterable[str]] = (
        'Voxel begin (Z)', 'Voxel end (Z)', 'Voxel index',
        'Patient ID', 'Bx ID', 'Bx index', 'Bx refnum',
        'Simulated bool', 'Simulated type'
    ),
    add_abs: bool = True,                              # include precomputed |Δ| columns
    add_log1p: bool = True,                            # add log1p(|Δ|) derived from the abs columns
    return_long: bool = False,                         # also return tidy/long version
    require_precomputed_abs: bool = True,              # expect abs block present
    fallback_recompute_abs: bool = False               # only if you *really* want to compute |Δ| here
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Join voxel-wise Δ metrics (signed + optional absolute) with one or more per-voxel gradients,
    preserving metadata. Keys: (Patient ID, Bx index, Voxel index).

    Reads signed deltas from: (f"{zero_level_index_str} deltas", 'nominal_minus_[mode|q50|mean]')
    Reads absolute deltas from: (f"{zero_level_index_str} abs deltas", 'abs_nominal_minus_[mode|q50|mean]')
    Reads gradients from: (gradient_top, <stat>) for each requested stat.
    """

    # ---- param normalization (support gradient_stat alias) ----
    if gradient_stat is not None:
        gradient_stats = gradient_stat
    if isinstance(gradient_stats, str):
        requested_stats = [gradient_stats]
    else:
        requested_stats = list(gradient_stats)

    # ---- helpers ----
    def _need(df, pair):
        if pair not in df.columns:
            raise KeyError(f"Missing required column {pair} in dataframe.")
        return df.loc[:, pair]

    def _flatten_select(df, top_names):
        cols = [(t, '') for t in top_names if (t, '') in df.columns]
        out = df.loc[:, cols].copy()
        out.columns = [t for (t, _) in cols]
        return out

    def _unit_from_top(top_name: str) -> str:
        if '(' in top_name and ')' in top_name:
            return top_name[top_name.find('('): top_name.rfind(')')+1]
        return ''

    # friendly → dataframe key,pretty label
    stat_map = {
        'nominal':        ('nominal', 'nominal'),
        'mean':           ('mean', 'mean'),
        'median':         ('quantile_50', 'median'),
        'q50':            ('quantile_50', 'median'),
        'Q50':            ('quantile_50', 'median'),
        'mode':           ('argmax_density', 'mode'),
        'argmax_density': ('argmax_density', 'mode'),
    }
    norm_stats = []
    seen = set()
    for s in requested_stats:
        key = str(s).strip()
        canon = stat_map.get(key, (key, key))  # allow raw names if present
        if canon not in seen:
            norm_stats.append(canon)
            seen.add(canon)

    dose_unit = _unit_from_top(zero_level_index_str)
    grad_unit = _unit_from_top(gradient_top)

    # ---- metadata from deltas df ----
    meta_keep = tuple(meta_keep) if meta_keep is not None else ()
    meta_flat = _flatten_select(nominal_deltas_df, meta_keep)

    # signed deltas
    signed_top = f"{zero_level_index_str} deltas"
    signed_pairs = [
        (signed_top, 'nominal_minus_mode'),
        (signed_top, 'nominal_minus_q50'),
        (signed_top, 'nominal_minus_mean'),
    ]
    missing_signed = [p for p in signed_pairs if p not in nominal_deltas_df.columns]
    if missing_signed:
        raise KeyError(f"Missing signed delta columns for '{zero_level_index_str}': {missing_signed}")
    deltas_signed = nominal_deltas_df.loc[:, signed_pairs].copy()
    deltas_signed.columns = [f'Δ_mode {dose_unit}', f'Δ_median {dose_unit}', f'Δ_mean {dose_unit}']

    # abs deltas (precomputed)
    abs_top = f"{zero_level_index_str} abs deltas"
    abs_pairs = [
        (abs_top, 'abs_nominal_minus_mode'),
        (abs_top, 'abs_nominal_minus_q50'),
        (abs_top, 'abs_nominal_minus_mean'),
    ]
    has_abs = all(p in nominal_deltas_df.columns for p in abs_pairs)

    if add_abs:
        if not has_abs and require_precomputed_abs and not fallback_recompute_abs:
            raise KeyError(
                f"Absolute delta columns not found for '{zero_level_index_str}' and recompute is disabled. "
                f"Expected: {abs_pairs}. Provide *_with_abs DataFrame or set fallback_recompute_abs=True."
            )
        if has_abs:
            deltas_abs = nominal_deltas_df.loc[:, abs_pairs].copy()
            deltas_abs.columns = [f'|Δ_mode| {dose_unit}', f'|Δ_median| {dose_unit}', f'|Δ_mean| {dose_unit}']
        elif fallback_recompute_abs:
            deltas_abs = deltas_signed.abs().copy()
            deltas_abs.columns = [f'|Δ_mode| {dose_unit}', f'|Δ_median| {dose_unit}', f'|Δ_mean| {dose_unit}']
        else:
            deltas_abs = pd.DataFrame(index=deltas_signed.index)

    pieces = [meta_flat, deltas_signed]
    if add_abs:
        pieces.append(deltas_abs)
    deltas_block = pd.concat(pieces, axis=1)

    # ---- gradient columns (one per requested stat) ----
    source_meta_flat = _flatten_select(cohort_by_voxel_df, meta_keep)
    grads = []
    grad_colnames = []
    for stat_key, label in norm_stats:
        pair = (gradient_top, stat_key)
        if pair not in cohort_by_voxel_df.columns:
            raise KeyError(f"Gradient column {pair} not found in cohort_by_voxel_df.")
        col_name = f'Grad[{label}] {grad_unit}'
        grads.append(_need(cohort_by_voxel_df, pair).rename(col_name))
        grad_colnames.append(col_name)
    grad_block = pd.concat([source_meta_flat] + grads, axis=1)

    # ---- coerce keys & merge ----
    keys = ['Patient ID', 'Bx index', 'Voxel index']
    for k in keys:
        if k in deltas_block.columns:
            deltas_block[k] = deltas_block[k].astype(str if k == 'Patient ID' else 'int64', errors='ignore')
        if k in grad_block.columns:
            grad_block[k] = grad_block[k].astype(str if k == 'Patient ID' else 'int64', errors='ignore')

    wide = deltas_block.merge(grad_block, on=keys, how='inner')

    # ---- log1p(|Δ|) from abs only ----
    if add_log1p:
        abs_cols = [c for c in [f'|Δ_mode| {dose_unit}', f'|Δ_median| {dose_unit}', f'|Δ_mean| {dose_unit}'] if c in wide.columns]
        if add_abs and not abs_cols and require_precomputed_abs and not fallback_recompute_abs:
            raise KeyError("Requested log1p(|Δ|) but no absolute delta columns are present.")
        for c in abs_cols:
            wide[f'log1p{c}'] = np.log1p(wide[c])

    # ---- long version (keeps gradient columns wide) ----
    long = None
    if return_long:
        keep_id = ['Patient ID', 'Bx index', 'Bx ID', 'Voxel index'] + [gc for gc in grad_colnames if gc in wide.columns]
        keep_id = [c for c in keep_id if c in wide.columns]

        signed_vars = [c for c in [f'Δ_mode {dose_unit}', f'Δ_median {dose_unit}', f'Δ_mean {dose_unit}'] if c in wide.columns]
        long = wide.melt(
            id_vars=keep_id,
            value_vars=signed_vars,
            var_name='Delta kind',
            value_name='Delta (signed)'
        )

        if add_abs:
            abs_vars = [c for c in [f'|Δ_mode| {dose_unit}', f'|Δ_median| {dose_unit}', f'|Δ_mean| {dose_unit}'] if c in wide.columns]
            if abs_vars:
                abs_long = wide.melt(
                    id_vars=keep_id,
                    value_vars=abs_vars,
                    var_name='Abs kind',
                    value_name='|Delta|'
                )
                # normalize abs labels to match 'Delta kind' (strip surrounding |...|)
                if dose_unit:
                    # keep the unit; strip only the bars
                    abs_long['Delta kind'] = abs_long['Abs kind'].str.replace('|Δ', 'Δ', regex=False).str.replace('| ', ' ', regex=False)
                    abs_long['Delta kind'] = abs_long['Delta kind'].str.replace('|', '', regex=False)
                else:
                    abs_long['Delta kind'] = abs_long['Abs kind'].str.replace('|', '', regex=False)

                abs_long = abs_long.drop(columns=['Abs kind'])
                long = long.merge(abs_long, on=keep_id + ['Delta kind'], how='left')
            elif require_precomputed_abs and not fallback_recompute_abs:
                raise KeyError("Requested |Δ| in long output but absolute delta columns are not present.")

        if add_log1p and ('|Delta|' in long.columns):
            long['log1p|Delta|'] = np.log1p(long['|Delta|'])


    return wide, long