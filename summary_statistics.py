import pandas as pd
import os
from scipy.stats import gaussian_kde
import numpy as np
from numpy.linalg import LinAlgError
from typing import List, Optional
from pathlib import Path
import pingouin as pg


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

    # Define the quantiles to compute
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]

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
            stats = df[pair].describe(percentiles=quantiles)
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




def generate_summary_csv_with_argmax(output_dir, csv_name, df, col_pairs=None, exclude_columns=None):
    """
    Generate summary statistics (including 5% & 95% quantiles and KDE-mode)
    for specified columns in a DataFrame and save the results to CSV.
    
    If col_pairs is None, summary stats will be computed for all columns in the DataFrame,
    except those specified in exclude_columns.
    
    Parameters:
        output_dir (str): Directory path where the summary CSV will be saved.
        csv_name (str): The name of the CSV file (e.g., 'summary.csv').
        df (pd.DataFrame): The input DataFrame with a MultiIndex for its columns.
        col_pairs (list of tuple, optional): List of tuples representing the columns to summarize,
            e.g. [( 'Dose (Gy)', '' ), ( 'Dose (Gy)', 'grad' )].
            If None, all columns in the DataFrame (not excluded) will be summarized.
        exclude_columns (list of tuple, optional): List of column pairings to exclude.
    """
    os.makedirs(output_dir, exist_ok=True)

    if col_pairs is None:
        col_pairs = list(df.columns)
    if exclude_columns:
        col_pairs = [pair for pair in col_pairs if pair not in exclude_columns]

    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    summary_rows = []
    labels = []

    for pair in col_pairs:
        if pair not in df.columns:
            print(f"Warning: Column {pair} not found, skipping.")
            continue

        series = df[pair].dropna()
        if series.empty:
            print(f"Warning: Column {pair} is all-NaN, skipping.")
            continue

        # descriptive stats with extra quantiles
        stats = series.describe(percentiles=quantiles)

        # compute KDE-mode safely
        data = series.values
        if np.unique(data).size < 2:
            mode = data[0]
        else:
            try:
                kde = gaussian_kde(data)
                xs = np.linspace(data.min(), data.max(), 1_000)
                pdf = kde(xs)
                mode = xs[np.argmax(pdf)]
            except LinAlgError:
                mode = np.median(data)

        stats['kde_mode'] = mode

        summary_rows.append(stats)
        labels.append(f"{pair[0]} | {pair[1]}" if pair[1] else pair[0])

    if not summary_rows:
        print("No valid columns to summarize; no CSV generated.")
        return

    summary_df = pd.DataFrame(summary_rows, index=labels)
    output_path = os.path.join(output_dir, csv_name)
    summary_df.to_csv(output_path)
    print(f"Summary CSV successfully saved to: {output_path}")



def compute_summary(df: pd.DataFrame,
                    group_vars: list,
                    value_vars: list,
                    output_dir = None,
                    flatten = False, 
                    csv_name = None) -> pd.DataFrame:
    """
    Group df by group_vars and compute summary stats on value_vars.
    Returns a tidy DataFrame where each row is one group and the columns
    are count, mean, std, min, median, max for each of the two metrics.
    """

    # 1) compute the summary statistics
    stats = (
        df
        .groupby(group_vars)[value_vars]
        .describe(percentiles=[.05, .25, .5, .75, .95])
        .reset_index()
    )
    
    # 2) flatten the MultiIndex columns  
    if flatten:
        # Flatten the MultiIndex columns
        stats.columns = [
            f"{metric}_{stat}"
            for metric, stat in stats.columns
        ]
    else:
        # Keep the MultiIndex structure
        pass
        
    
    # 3) if output_dir is specified, save the summary to a CSV file
    if output_dir is not None and csv_name is not None:
        output_path = os.path.join(output_dir, csv_name)
        # Save the summary DataFrame to CSV
        stats.to_csv(output_path, index=False)
        print(f"Summary CSV successfully saved to: {output_path}")

    return stats





def compute_summary_with_argmax(df: pd.DataFrame,
                    group_vars: list,
                    value_vars: list,
                    output_dir=None,
                    flatten=False, 
                    csv_name=None) -> pd.DataFrame:
    """
    Group df by group_vars and compute summary stats on value_vars,
    plus the argmax density (KDE mode) for each metric.
    Returns a DataFrame where each row is one group and the columns
    are count, mean, std, min, 5th, 25th, 50th, 75th, 95th, max, 
    and kde_mode for each of the metrics.
    """

    # 1) compute the summary statistics
    stats = (
        df
        .groupby(group_vars)[value_vars]
        .describe(percentiles=[.05, .25, .5, .75, .95])
        .reset_index()
    )

    # 1b) compute KDE-modes for each group/variable
    kde_rows = []
    for name, group in df.groupby(group_vars):
        # build the key for merging back
        if isinstance(name, tuple):
            row = dict(zip(group_vars, name))
        else:
            row = {group_vars[0]: name}
        # for each value_var compute kde-mode
        for var in value_vars:
            data = group[var].dropna().values
            if len(data) >= 2:
                kde = gaussian_kde(data)
                xs = np.linspace(data.min(), data.max(), 1000)
                pdf = kde(xs)
                mode = xs[np.argmax(pdf)]
            elif len(data) == 1:
                mode = data[0]
            else:
                mode = np.nan
            row[(var, 'kde_mode')] = mode
        kde_rows.append(row)

    kde_df = pd.DataFrame(kde_rows)

    # 2) merge the kde_mode back into stats
    #    stats has a MultiIndex on columns: (variable, statname)
    stats = stats.merge(kde_df, on=group_vars)

    # 3) flatten the MultiIndex columns if requested
    if flatten:
        stats.columns = [
            f"{metric}_{stat}"
            for metric, stat in stats.columns
        ]
    # else: leave the MultiIndex in place

    # 4) optionally save to CSV
    if output_dir is not None and csv_name is not None:
        output_path = os.path.join(output_dir, csv_name)
        stats.to_csv(output_path, index=False)
        print(f"Summary CSV successfully saved to: {output_path}")

    return stats


def compute_summary_non_multiindex(
    df: pd.DataFrame,
    value_vars: List[str],
    output_dir: Optional[str] = None,
    csv_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute summary stats on the given value_vars across the entire DataFrame,
    plus the KDE-based mode for each variable.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe.
    value_vars : List[str]
        Columns to summarize.
    output_dir : str, optional
        If provided along with csv_name, save the result as CSV to this directory.
    csv_name : str, optional
        Filename for the CSV (must be provided if output_dir is).

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by variable name with columns:
        ['count', 'mean', 'std', 'min', '5th', '25th', '50th', '75th', '95th', 'max', 'kde_mode'].
    """
    # 1) descriptive stats
    desc = (
        df[value_vars]
        .describe(percentiles=[.05, .25, .5, .75, .95])
        .T
        .rename(columns={
            '5%': '5th', '25%': '25th',
            '50%': '50th', '75%': '75th',
            '95%': '95th'
        })
    )


    # 2) optionally save to CSV
    if output_dir is not None and csv_name is not None:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, csv_name)
        desc.to_csv(path)
        print(f"Summary CSV successfully saved to: {path}")

    return desc

def compute_summary_with_argmax_non_multiindex(
    df: pd.DataFrame,
    value_vars: List[str],
    output_dir: Optional[str] = None,
    csv_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute summary stats on the given value_vars across the entire DataFrame,
    plus the KDE-based mode for each variable.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe.
    value_vars : List[str]
        Columns to summarize.
    output_dir : str, optional
        If provided along with csv_name, save the result as CSV to this directory.
    csv_name : str, optional
        Filename for the CSV (must be provided if output_dir is).

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by variable name with columns:
        ['count', 'mean', 'std', 'min', '5th', '25th', '50th', '75th', '95th', 'max', 'kde_mode'].
    """
    # 1) descriptive stats
    desc = (
        df[value_vars]
        .describe(percentiles=[.05, .25, .5, .75, .95])
        .T
        .rename(columns={
            '5%': '5th', '25%': '25th',
            '50%': '50th', '75%': '75th',
            '95%': '95th'
        })
    )

    # 2) compute KDE-mode for each variable
    modes = {}
    for var in value_vars:
        data = df[var].dropna().values
        if len(data) >= 2:
            kde = gaussian_kde(data)
            xs = np.linspace(data.min(), data.max(), 1000)
            pdf = kde(xs)
            mode = xs[np.argmax(pdf)]
        elif len(data) == 1:
            mode = data[0]
        else:
            mode = np.nan
        modes[var] = mode

    desc['kde_mode'] = pd.Series(modes)

    # 3) optionally save to CSV
    if output_dir is not None and csv_name is not None:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, csv_name)
        desc.to_csv(path)
        print(f"Summary CSV successfully saved to: {path}")

    return desc











def compute_biopsy_nominal_deltas(cohort_global_dosimetry_by_voxel_df: pd.DataFrame,
                                  zero_level_index_str: str = 'Dose (Gy)') -> pd.DataFrame:
    """
    From the cohort multiindex dataframe, compute per-voxel deltas:
        - nominal - mean
        - nominal - mode  (mode ≡ argmax_density)
        - nominal - Q50   (median ≡ quantile_50)
    for zero_level_index_str [such as Dose (Gy)], while preserving the requested metadata columns.

    Returns a new dataframe with metadata + a 'zero_level_index_str deltas' column block.
    """

    df = cohort_global_dosimetry_by_voxel_df

    # --- columns we must preserve (metadata) ---
    meta_cols = [
        ('Voxel begin (Z)', ''), ('Voxel end (Z)', ''), ('Voxel index', ''),
        ('Patient ID', ''), ('Bx ID', ''), ('Bx index', ''), ('Bx refnum', ''),
        ('Simulated bool', ''), ('Simulated type', '')
    ]

    # Sanity-check presence (raise a clear error if missing)
    missing_meta = [c for c in meta_cols if c not in df.columns]
    if missing_meta:
        raise KeyError(f"Missing expected metadata columns: {missing_meta}")

    # --- pull the dose components we need ---
    need = [
        (zero_level_index_str, 'nominal'),
        (zero_level_index_str, 'mean'),
        (zero_level_index_str, 'argmax_density'),
        (zero_level_index_str, 'quantile_50'),
    ]
    missing_need = [c for c in need if c not in df.columns]
    if missing_need:
        raise KeyError(f"Missing expected dose statistic columns: {missing_need}")

    nominal = df[(zero_level_index_str, 'nominal')]
    mean    = df[(zero_level_index_str, 'mean')]
    mode    = df[(zero_level_index_str, 'argmax_density')]
    q50     = df[(zero_level_index_str, 'quantile_50')]

    # --- build the deltas block with a tidy MultiIndex for columns ---
    delta_cols = pd.MultiIndex.from_tuples([
        (zero_level_index_str+' deltas', 'nominal_minus_mean'),
        (zero_level_index_str+' deltas', 'nominal_minus_mode'),
        (zero_level_index_str+' deltas', 'nominal_minus_q50'),
    ])

    deltas = pd.concat(
        [
            nominal - mean,
            nominal - mode,
            nominal - q50,
        ],
        axis=1
    )
    deltas.columns = delta_cols

    # --- assemble output: metadata + deltas ---
    out = pd.concat([df.loc[:, meta_cols].copy(), deltas], axis=1)

    # (optional) stable column ordering by first and second level
    out = out.reindex(columns=pd.MultiIndex.from_tuples(meta_cols + list(delta_cols)))

    # (optional) assert uniqueness of voxels within biopsies if you want a guardrail
    # unique_key = out[('Patient ID','')].astype(str) + '|' + out[('Bx index','')].astype(str) + '|' + out[('Voxel index','')].astype(str)
    # if unique_key.duplicated().any():
    #     raise ValueError("Duplicate (Patient ID, Bx index, Voxel index) rows detected.")

    return out







def save_delta_boxplot_summary_csv(
    deltas_df: pd.DataFrame,
    output_dir,
    csv_name: str,
    *,
    zero_level_index_str: str = 'Dose (Gy)',
    include_patient_ids: list | None = None,
    decimals: int = 3
) -> Path:
    """
    Summarize the three delta distributions produced by `compute_biopsy_nominal_deltas`
    and save as CSV.

    Stats per delta:
      n_voxels, n_biopsies, mean, std, sem, min, q05, q25, q50, q75, q95, max, iqr

    Parameters
    ----------
    deltas_df : output of compute_biopsy_nominal_deltas(...)
    output_dir : directory to write CSV
    csv_name : file name ('.csv' added if missing)
    zero_level_index_str : e.g., 'Dose (Gy)' or 'Dose grad (Gy/mm)'
    include_patient_ids : optional list of patient IDs to filter
    decimals : rounding precision in the CSV

    Returns
    -------
    Path to the written CSV.
    """
    data = deltas_df
    if include_patient_ids is not None:
        data = data[data[('Patient ID','')].isin(include_patient_ids)]
        if data.empty:
            raise ValueError("Patient filter returned no rows.")

    # Identify delta columns
    block = f"{zero_level_index_str} deltas"
    cols = [
        (block, 'nominal_minus_mean'),
        (block, 'nominal_minus_mode'),
        (block, 'nominal_minus_q50'),
    ]
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise KeyError(
            f"Missing delta columns. Did you compute deltas with zero_level_index_str='{zero_level_index_str}'? "
            f"Missing: {missing}"
        )

    # Count unique biopsies among the included rows
    n_biopsies = (
        data.loc[:, [('Patient ID',''), ('Bx index','')]]
        .drop_duplicates()
        .shape[0]
    )

    # Build tidy series for easy grouping
    tidy = data.loc[:, cols].copy()
    tidy.columns = ['Nominal - Mean', 'Nominal - Mode', 'Nominal - Median (Q50)']
    tidy = tidy.melt(var_name='Delta', value_name='Value').dropna(subset=['Value'])

    # Helper to compute stats on a Series
    def _summarize(s: pd.Series) -> pd.Series:
        q = s.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
        out = pd.Series({
            'n_voxels': int(s.count()),
            'n_biopsies': int(n_biopsies),
            'mean': s.mean(),
            'std': s.std(ddof=1),
            'sem': s.sem(ddof=1),
            'min': s.min(),
            'q05': q.loc[0.05],
            'q25': q.loc[0.25],
            'q50': q.loc[0.50],
            'q75': q.loc[0.75],
            'q95': q.loc[0.95],
            'max': s.max(),
            'iqr': q.loc[0.75] - q.loc[0.25],
        })
        return out

    summary = tidy.groupby('Delta', as_index=True)['Value'].apply(_summarize)
    summary = summary.reset_index().rename(columns={'Value': 'stat'})  # cleanup from groupby-apply
    # After reset_index, 'stat' is not used; groupby-apply already expanded. Drop redundant column if present.
    if 'stat' in summary.columns and summary['stat'].isna().all():
        summary = summary.drop(columns=['stat'])

    # Round
    numeric_cols = [c for c in summary.columns if c != 'Delta']
    summary[numeric_cols] = summary[numeric_cols].round(decimals)

    # Write CSV
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not csv_name.lower().endswith('.csv'):
        csv_name += '.csv'
    out_path = output_dir / csv_name
    summary.to_csv(out_path, index=False)
    return out_path





# ---------------- 1) Compute MC-trial deltas vs nominal ---------------------

def compute_mc_trial_deltas(
    all_voxel_wise_dose_df: pd.DataFrame,
    value_cols: tuple[str, ...] = ('Dose (Gy)', 'Dose grad (Gy/mm)'),
    *,
    trial_col: str = 'MC trial num',
    nominal_trial_value: int | float = 0,
    include_nominal: bool = False,
    group_cols: tuple[str, ...] = ('Patient ID', 'Bx index', 'Voxel index'),
    keep_extra_cols: tuple[str, ...] = (
        'Bx refnum', 'Bx ID', 'Voxel begin (Z)', 'Voxel end (Z)',
        'Simulated bool', 'Simulated type', 'X (Bx frame)', 'Y (Bx frame)',
        'Z (Bx frame)', 'R (Bx frame)'
    ),
    drop_groups_without_nominal: bool = True
) -> pd.DataFrame:
    """
    For each (Patient ID, Bx index, Voxel index) and each MC trial, compute:
        delta = nominal_value (trial == nominal_trial_value) - value_at_trial
    for each column in value_cols.

    Returns a DataFrame with:
        [group_cols] + list(keep_extra_cols present) + [trial_col] + MultiIndex delta columns:
            (f"{v} deltas", "nominal_minus_trial") for v in value_cols
    """
    df = all_voxel_wise_dose_df.copy()

    # ensure required columns exist
    req = list(group_cols) + [trial_col] + list(value_cols)
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # restrict to columns we actually use/keep
    present_extras = [c for c in keep_extra_cols if c in df.columns]
    df = df[list(group_cols) + [trial_col] + list(value_cols) + present_extras].copy()

    # nominal per group
    nom = (
        df[df[trial_col] == nominal_trial_value]
        .loc[:, list(group_cols) + list(value_cols)]
        .rename(columns={v: f"{v} [nominal]" for v in value_cols})
    )

    # join nominal onto all rows
    merged = df.merge(nom, on=list(group_cols), how='left', validate='many_to_one')

    # optionally drop rows where nominal missing
    if drop_groups_without_nominal:
        before = len(merged)
        mask_has_nom = merged[f"{value_cols[0]} [nominal]"].notna()
        merged = merged.loc[mask_has_nom].copy()
        # if nothing left, fail loudly
        if merged.empty and before > 0:
            raise ValueError("All rows dropped due to missing nominal values for the specified groups.")

    # compute deltas
    delta_cols = []
    delta_frames = []
    for v in value_cols:
        nom_series = pd.to_numeric(merged[f"{v} [nominal]"], errors='coerce')
        cur_series = pd.to_numeric(merged[v], errors='coerce')
        d = nom_series - cur_series
        col = (f"{v} deltas", "nominal_minus_trial")
        delta_cols.append(col)
        delta_frames.append(d)

    deltas = pd.concat(delta_frames, axis=1)
    deltas.columns = pd.MultiIndex.from_tuples(delta_cols)

    # build output
    out_cols = list(group_cols) + present_extras + [trial_col]
    out = pd.concat([merged[out_cols].copy(), deltas], axis=1)

    # include/exclude nominal rows
    if not include_nominal:
        out = out[out[trial_col] != nominal_trial_value].copy()

    # sort (optional)
    out = out.sort_values(list(group_cols) + [trial_col]).reset_index(drop=True)
    return out


# ---------------- 2) Summarize and save to CSV ------------------------------

def save_mc_delta_summary_csv(
    mc_delta_df: pd.DataFrame,
    output_dir,
    csv_name: str,
    *,
    value_cols: tuple[str, ...] = ('Dose (Gy)', 'Dose grad (Gy/mm)'),
    group_cols: tuple[str, ...] = ('Patient ID', 'Bx index', 'Voxel index'),
    include_patient_ids: list | None = None,
    decimals: int = 3
) -> Path:
    """
    Summarize nominal-minus-trial deltas across all trials (and groups) for each metric in value_cols.
    Writes a tidy CSV with one row per metric (e.g., Dose (Gy), Dose grad (Gy/mm)).

    Reported columns:
        metric, n_rows, n_trials_unique, n_biopsies, n_group_voxels,
        mean, std, sem, min, q05, q25, q50, q75, q95, max, iqr
    """
    df = mc_delta_df.copy()

    # optional filter by patient ids
    if include_patient_ids is not None:
        if ('Patient ID' not in df.columns) and ('Patient ID', '') not in df.columns:
            raise KeyError("Patient ID column not found for filtering.")
        # handle both flat and MultiIndex columns for safety
        pid_col = ('Patient ID', '') if ('Patient ID', '') in df.columns else 'Patient ID'
        df = df[df[pid_col].isin(include_patient_ids)].copy()
        if df.empty:
            raise ValueError("Patient filter returned no rows.")

    # collect delta columns (MultiIndex) for each metric
    delta_cols = []
    for v in value_cols:
        col = (f"{v} deltas", "nominal_minus_trial")
        if col not in df.columns:
            raise KeyError(f"Missing delta column {col}. Did you run compute_mc_trial_deltas with value_cols including '{v}'?")
        delta_cols.append((v, col))  # (metric_name, column_key)

    # counts
    # unique biopsies = (Patient ID, Bx index)
    if all(c in df.columns for c in ['Patient ID', 'Bx index']):
        n_biopsies = df[['Patient ID', 'Bx index']].drop_duplicates().shape[0]
        n_group_voxels = df[list(group_cols)].drop_duplicates().shape[0] if all(c in df.columns for c in group_cols) else None
    else:
        # if MultiIndex metadata was used upstream
        pid_col = ('Patient ID',''); bxi_col = ('Bx index','')
        vox_col = ('Voxel index','')
        n_biopsies = df[[pid_col, bxi_col]].drop_duplicates().shape[0] if (pid_col in df.columns and bxi_col in df.columns) else None
        n_group_voxels = (
            df[[pid_col, bxi_col, vox_col]].drop_duplicates().shape[0]
            if all(c in df.columns for c in [pid_col, bxi_col, vox_col]) else None
        )

    # build tidy and summarize
    rows = []
    for metric_name, colkey in delta_cols:
        s = pd.to_numeric(df[colkey], errors='coerce').dropna()
        if s.empty:
            continue
        q = s.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
        rows.append({
            'metric': metric_name,
            'n_rows': int(s.shape[0]),
            'n_trials_unique': int(df['MC trial num'].nunique()) if 'MC trial num' in df.columns else None,
            'n_biopsies': int(n_biopsies) if n_biopsies is not None else None,
            'n_group_voxels': int(n_group_voxels) if n_group_voxels is not None else None,
            'mean': s.mean(),
            'std': s.std(ddof=1),
            'sem': s.sem(ddof=1),
            'min': s.min(),
            'q05': q.loc[0.05],
            'q25': q.loc[0.25],
            'q50': q.loc[0.50],
            'q75': q.loc[0.75],
            'q95': q.loc[0.95],
            'max': s.max(),
            'iqr': q.loc[0.75] - q.loc[0.25],
        })

    summary = pd.DataFrame(rows)
    if not summary.empty:
        num_cols = [c for c in summary.columns if c not in ('metric',)]
        summary[num_cols] = summary[num_cols].round(decimals)

    # write CSV
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not csv_name.lower().endswith('.csv'):
        csv_name += '.csv'
    out_path = output_dir / csv_name
    summary.to_csv(out_path, index=False)
    return out_path



def save_paired_effect_sizes_by_trial_csv_fast(
    mc_deltas: pd.DataFrame,
    output_dir,
    csv_name: str,
    *,
    value_cols: tuple[str, ...] = ('Dose (Gy)', 'Dose grad (Gy/mm)'),
    decimals: int = 3,
) -> Path:
    """
    Fast, vectorized paired effect sizes per (Patient ID, Bx index, MC trial num, metric).
    Assumes mc_deltas already contains voxel-wise paired differences:
        (f"{metric} deltas", "nominal_minus_trial").
    Computes: n_voxels, mean_diff, sd_diff, median_diff, q25, q75, iqr,
              CLES_ties, Cliffs_delta, Cohen_d_z, Hedges_g_z.
    """

    # Build a single long frame for all requested metrics
    parts = []
    for m in value_cols:
        col = (f"{m} deltas", "nominal_minus_trial")
        if col not in mc_deltas.columns:
            raise KeyError(f"Missing column {col}. Run compute_mc_trial_deltas for '{m}'.")
        d = mc_deltas[col].astype('float32')  # smaller, faster
        tmp = mc_deltas[['Patient ID', 'Bx index', 'MC trial num']].copy()
        tmp['metric'] = m
        tmp['diff'] = d
        parts.append(tmp)

    long = pd.concat(parts, ignore_index=True)

    # Speed tricks: categorical keys, avoid sorting work
    for k in ('Patient ID', 'Bx index', 'MC trial num', 'metric'):
        long[k] = long[k].astype('category')

    # Precompute booleans once (pandas can take mean of bools -> proportions)
    long['pos']  = long['diff'] > 0
    long['zero'] = long['diff'] == 0

    gb = long.groupby(['Patient ID', 'Bx index', 'MC trial num', 'metric'], observed=True, sort=False)

    # Core aggregates in one pass
    agg_basic = gb['diff'].agg(n_voxels='count', mean_diff='mean', sd_diff='std', median_diff='median')

    # Quartiles (done once; still vectorized). Two calls are cheap since groups are small.
    q25 = gb['diff'].quantile(0.25)
    q75 = gb['diff'].quantile(0.75)

    # Proportions for CLES / Cliff’s δ (no Python loops)
    prop = gb[['pos', 'zero']].mean()  # mean of booleans = proportion
    prop.rename(columns={'pos': 'prop_pos', 'zero': 'prop_zero'}, inplace=True)

    # Join everything
    out = agg_basic.join(q25.rename('q25')).join(q75.rename('q75')).join(prop)

    # Derived stats
    out['iqr'] = out['q75'] - out['q25']
    out['prop_neg'] = 1.0 - out['prop_pos'] - out['prop_zero']
    out['CLES_ties'] = out['prop_pos'] + 0.5 * out['prop_zero']
    out['Cliffs_delta'] = out['prop_pos'] - out['prop_neg']

    # Cohen's d_z (paired; SD of differences) and Hedges' g_z
    # guard against sd == 0
    out['Cohen_d_z'] = out['mean_diff'] / out['sd_diff']
    out.loc[~np.isfinite(out['Cohen_d_z']), 'Cohen_d_z'] = np.nan

    # Hedges correction J(df) with df = n-1
    df = out['n_voxels'] - 1
    J = 1 - 3 / (4*df - 1)
    out['Hedges_g_z'] = out['Cohen_d_z'] * J

    # Clean up & save
    out = out.reset_index()
    num_cols = [c for c in out.columns if c not in ('Patient ID','Bx index','MC trial num','metric')]
    out[num_cols] = out[num_cols].round(decimals)

    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    if not csv_name.lower().endswith('.csv'):
        csv_name += '.csv'
    out_path = output_dir / csv_name
    out.to_csv(out_path, index=False)
    return out_path




def save_nominal_vs_trial_proportions_csv(
    mc_deltas: pd.DataFrame,
    output_dir,
    base_name: str,
    *,
    value_cols: tuple[str, ...] = ('Dose (Gy)', 'Dose grad (Gy/mm)'),
    exclude_nominal_trial: bool = True,   # drop trial 0 rows if present
    decimals: int = 3,
):
    """
    For each voxel, across all MC trials, compute how often nominal > trial (pos),
    nominal < trial (neg), and equal (zero) — for each metric in value_cols.

    Writes:
      - {base_name}_per_voxel.csv : one row per (Patient ID, Bx index, Voxel index, metric)
      - {base_name}_per_biopsy.csv: per-biopsy medians (across voxels) of those proportions

    Returns:
      (Path_to_voxel_csv, Path_to_biopsy_csv)
    """
    # Build long table: keys + metric + diff
    parts = []
    for m in value_cols:
        col = (f"{m} deltas", "nominal_minus_trial")
        if col not in mc_deltas.columns:
            raise KeyError(f"Missing column {col}. Run compute_mc_trial_deltas for '{m}'.")
        tmp = mc_deltas[['Patient ID','Bx index','Voxel index','MC trial num']].copy()
        tmp['metric'] = m
        tmp['diff'] = pd.to_numeric(mc_deltas[col], errors='coerce').astype('float32')
        parts.append(tmp)

    long = pd.concat(parts, ignore_index=True)

    if exclude_nominal_trial and 'MC trial num' in long.columns:
        long = long[long['MC trial num'] != 0]

    # proportions per voxel (across trials)
    long['gt'] = long['diff'] > 0
    long['eq'] = long['diff'] == 0

    gb_vox = long.groupby(['Patient ID','Bx index','Voxel index','metric'], observed=True, sort=False)
    voxel_stats = gb_vox.agg(
        n_trials=('diff','count'),
        prop_gt=('gt','mean'),          # P(nominal > trial)
        prop_eq=('eq','mean')           # P(nominal = trial)
    ).reset_index()
    voxel_stats['prop_lt'] = 1.0 - voxel_stats['prop_gt'] - voxel_stats['prop_eq']
    voxel_stats['CLES_ties'] = voxel_stats['prop_gt'] + 0.5*voxel_stats['prop_eq']

    # per-biopsy medians across voxels (robust)
    gb_bio = voxel_stats.groupby(['Patient ID','Bx index','metric'], observed=True, sort=False)
    biopsy_stats = gb_bio.agg(
        n_voxels=('n_trials','size'),
        n_trials_median=('n_trials','median'),
        prop_gt_median=('prop_gt','median'),
        prop_eq_median=('prop_eq','median'),
        prop_lt_median=('prop_lt','median'),
        CLES_ties_median=('CLES_ties','median'),
        CLES_ties_IQR=('CLES_ties', lambda s: (s.quantile(0.75) - s.quantile(0.25)))
    ).reset_index()

    # round & save
    for df in (voxel_stats, biopsy_stats):
        num_cols = [c for c in df.columns if c not in ('Patient ID','Bx index','Voxel index','metric')]
        df[num_cols] = df[num_cols].round(decimals)

    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    vox_path = output_dir / f"{base_name}_per_voxel.csv"
    bio_path = output_dir / f"{base_name}_per_biopsy.csv"
    voxel_stats.to_csv(vox_path, index=False)
    biopsy_stats.to_csv(bio_path, index=False)
    return vox_path, voxel_stats, bio_path, biopsy_stats



def save_cohort_cles_summary_csv(
    voxel_stats,           # DataFrame OR path to "*_per_voxel.csv"
    biopsy_stats,          # DataFrame OR path to "*_per_biopsy.csv"
    output_dir,
    csv_name: str = "cohort_cles_summary.csv",
    *,
    metrics_col: str = "metric",
    decimals: int = 3,
):
    """
    Build cohort-wide CLES summaries from outputs of save_nominal_vs_trial_proportions_csv.

    voxel_stats expects at least:
        ['Patient ID','Bx index','Voxel index', metrics_col, 'n_trials', 'prop_gt', 'prop_eq']
        (if 'prop_lt' or 'CLES_ties' missing, they are derived)
    biopsy_stats expects at least:
        ['Patient ID','Bx index', metrics_col, 'CLES_ties_median'] (optional 'CLES_ties_IQR')

    Output CSV: one row per metric with
        n_voxels, n_biopsies, n_trials_total,
        pooled_prop_gt/eq/lt, pooled_CLES_strict, pooled_CLES_ties,
        biopsy_level_CLES_median, biopsy_level_CLES_q25/q75, biopsy_level_CLES_IQR
    """
    # Load if file paths were given
    if isinstance(voxel_stats, (str, Path)):
        voxel_stats = pd.read_csv(voxel_stats)
    if isinstance(biopsy_stats, (str, Path)):
        biopsy_stats = pd.read_csv(biopsy_stats)

    # Be robust to missing derived cols
    if 'prop_lt' not in voxel_stats.columns:
        voxel_stats['prop_lt'] = 1.0 - voxel_stats['prop_gt'] - voxel_stats['prop_eq']
    if 'CLES_ties' not in voxel_stats.columns:
        voxel_stats['CLES_ties'] = voxel_stats['prop_gt'] + 0.5 * voxel_stats['prop_eq']

    rows = []
    for m in sorted(voxel_stats[metrics_col].unique()):
        vs = voxel_stats[voxel_stats[metrics_col] == m].copy()
        bs = biopsy_stats[biopsy_stats[metrics_col] == m].copy()

        # Cohort-pooled (weights = number of trials contributing for each voxel)
        w = vs['n_trials'].astype(float)
        W = w.sum()
        pooled_prop_gt = (vs['prop_gt'] * w).sum() / W
        pooled_prop_eq = (vs['prop_eq'] * w).sum() / W
        pooled_prop_lt = 1.0 - pooled_prop_gt - pooled_prop_eq

        # Biopsy-level distribution (each biopsy counts once via its per-biopsy median CLES)
        cles_b = bs['CLES_ties_median'].astype(float)
        cohort_median = cles_b.median()
        cohort_q25    = cles_b.quantile(0.25)
        cohort_q75    = cles_b.quantile(0.75)

        rows.append({
            'metric': m,
            'n_voxels': int(vs[['Patient ID','Bx index','Voxel index']].drop_duplicates().shape[0]),
            'n_biopsies': int(bs[['Patient ID','Bx index']].drop_duplicates().shape[0]),
            'n_trials_total': int(vs['n_trials'].sum()),
            'pooled_prop_gt': pooled_prop_gt,
            'pooled_prop_eq': pooled_prop_eq,
            'pooled_prop_lt': pooled_prop_lt,
            'pooled_CLES_strict': pooled_prop_gt,
            'pooled_CLES_ties': pooled_prop_gt + 0.5 * pooled_prop_eq,
            'biopsy_level_CLES_median': cohort_median,
            'biopsy_level_CLES_q25': cohort_q25,
            'biopsy_level_CLES_q75': cohort_q75,
            'biopsy_level_CLES_IQR': cohort_q75 - cohort_q25,
        })

    out = pd.DataFrame(rows)
    num_cols = [c for c in out.columns if c not in ('metric',)]
    out[num_cols] = out[num_cols].round(decimals)

    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    if not str(csv_name).lower().endswith('.csv'):
        csv_name += '.csv'
    out_path = output_dir / csv_name
    out.to_csv(out_path, index=False)
    return out_path, out




def save_cohort_pooled_cles_from_mc_deltas(
    mc_deltas: pd.DataFrame,
    output_dir,
    csv_name: str = "cohort_pooled_cles.csv",
    *,
    value_cols: tuple[str, ...] = ('Dose (Gy)', 'Dose grad (Gy/mm)'),
    exclude_nominal_trial: bool = True,  # drop trial 0 rows
    decimals: int = 3,
):
    """
    Compute cohort-pooled CLES from mc_deltas (i.e., across ALL patients/biopsies/voxels/trials).

    diff = (nominal - trial) for the SAME voxel.
    CLES_strict = P(diff > 0)
    CLES_ties   = P(diff > 0) + 0.5*P(diff == 0)

    Returns: (path, DataFrame) with columns:
      metric, n_pairs, prop_gt, prop_eq, prop_lt, CLES_strict, CLES_ties
    """
    rows = []
    df = mc_deltas
    if exclude_nominal_trial and 'MC trial num' in df.columns:
        df = df[df['MC trial num'] != 0]

    for m in value_cols:
        col = (f"{m} deltas", "nominal_minus_trial")
        if col not in df.columns:
            raise KeyError(f"Missing column {col}. Run compute_mc_trial_deltas for '{m}' first.")
        s = pd.to_numeric(df[col], errors='coerce').dropna()
        n = int(s.size)
        prop_gt = (s > 0).mean()
        prop_eq = (s == 0).mean()
        prop_lt = 1.0 - prop_gt - prop_eq
        rows.append({
            'metric': m,
            'n_pairs': n,
            'prop_gt': prop_gt,
            'prop_eq': prop_eq,
            'prop_lt': prop_lt,
            'CLES_strict': prop_gt,
            'CLES_ties': prop_gt + 0.5 * prop_eq,
        })

    out = pd.DataFrame(rows)
    num_cols = [c for c in out.columns if c not in ('metric',)]
    out[num_cols] = out[num_cols].round(decimals)

    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    if not csv_name.lower().endswith('.csv'):
        csv_name += '.csv'
    out_path = output_dir / csv_name
    out.to_csv(out_path, index=False)
    return out_path, out