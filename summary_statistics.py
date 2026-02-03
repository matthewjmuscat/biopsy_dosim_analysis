import pandas as pd
import os
from scipy.stats import gaussian_kde
import numpy as np
from numpy.linalg import LinAlgError
from typing import List, Optional, Iterable, Tuple
from pathlib import Path
import pingouin as pg

# Global KDE evaluation grid size (shared with plotting)
KDE_GRID_SIZE = 10000

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

    return summary_df




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

    return summary_df





def compute_summary(
    df: pd.DataFrame,
    group_vars: list,
    value_vars: list,
    output_dir=None,
    flatten: bool = False,
    csv_name: str | None = None
) -> pd.DataFrame:
    """
    Group df by group_vars and compute summary stats on value_vars.

    Includes:
      count, mean, std, min, 5%, 25%, 50% (median), 75%, 95%, max
      plus:
        - IQR   = 75% - 25%
        - IPR90 = 95% - 5%
    """

    # 1) compute the summary statistics (MultiIndex columns: (metric, stat))
    stats = df.groupby(group_vars)[value_vars].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    )

    # 2) add IQR and IPR90 for each metric (still in MultiIndex form)
    for metric in value_vars:
        stats[(metric, "IQR")] = stats[(metric, "75%")] - stats[(metric, "25%")]
        stats[(metric, "IPR90")] = stats[(metric, "95%")] - stats[(metric, "5%")]

    stats = stats.reset_index()

    # 3) optionally flatten the MultiIndex columns
    if flatten and isinstance(stats.columns, pd.MultiIndex):
        stats.columns = [
            col[0] if (col[1] == "" or col[1] is None) else f"{col[0]}_{col[1]}"
            for col in stats.columns
        ]

    # 4) save if requested
    if output_dir is not None and csv_name is not None:
        output_path = os.path.join(output_dir, csv_name)
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
                xs = np.linspace(data.min(), data.max(), KDE_GRID_SIZE)
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
    csv_name: Optional[str] = None,
    kde_grid_size: int = KDE_GRID_SIZE
) -> pd.DataFrame:
    """
    Compute summary stats on the given value_vars across the entire DataFrame,
    plus a KDE-based mode for each variable.

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
    kde_grid_size : int
        Number of points for evaluating the KDE on [min, max]. Default KDE_GRID_SIZE.

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by variable name with columns:
        ['count', 'mean', 'std', 'min', '5th', '25th', '50th',
         '75th', '95th', 'max', 'kde_mode'].
    """
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    rows = []
    labels = []

    for col in value_vars:
        if col not in df.columns:
            print(f"Warning: column '{col}' not found, skipping.")
            continue

        # Coerce to numeric and drop NaNs
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            print(f"Warning: column '{col}' is all-NaN after coercion, skipping.")
            continue

        # Descriptive stats with extra quantiles
        desc = series.describe(percentiles=quantiles)
        # Normalize names to requested output schema
        # pandas outputs: ['count','mean','std','min','5%','25%','50%','75%','95%','max']
        desc = desc.rename(index={
            '5%': '5th', '25%': '25th', '50%': '50th',
            '75%': '75th', '95%': '95th'
        })

        # Compute KDE mode safely
        data = series.values
        if np.unique(data).size < 2:
            mode = float(data[0])
        else:
            try:
                kde = gaussian_kde(data)
                xs = np.linspace(data.min(), data.max(), kde_grid_size)
                pdf = kde(xs)
                mode = float(xs[np.argmax(pdf)])
            except LinAlgError:
                # Fallback: robust central tendency
                mode = float(np.median(data))

        # Assemble row
        row = {
            'count': float(desc['count']),
            'mean': float(desc['mean']),
            'std': float(desc['std']),
            'min': float(desc['min']),
            '5th': float(desc['5th']),
            '25th': float(desc['25th']),
            '50th': float(desc['50th']),
            '75th': float(desc['75th']),
            '95th': float(desc['95th']),
            'max': float(desc['max']),
            'kde_mode': mode
        }
        rows.append(row)
        labels.append(col)

    if not rows:
        raise ValueError("No valid numeric columns to summarize.")

    out = pd.DataFrame(rows, index=labels)

    # Optionally save
    if output_dir is not None and csv_name is not None:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, csv_name)
        out.to_csv(path)
        print(f"Summary CSV successfully saved to: {path}")

    return out

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
        ('Patient ID', ''), ('Bx ID', ''), ('Bx index', ''), 
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





def compute_biopsy_nominal_deltas_with_abs(
    cohort_global_dosimetry_by_voxel_df: pd.DataFrame,
    zero_level_index_str: str = 'Dose (Gy)'
) -> pd.DataFrame:
    """
    Same as `compute_biopsy_nominal_deltas`, but also includes absolute values
    of the three differences. Returns metadata + two column blocks:

      - f"{zero_level_index_str} deltas":               nominal_minus_[mean|mode|q50]
      - f"{zero_level_index_str} abs deltas":           |nominal_minus_[mean|mode|q50]|

    Columns are a tidy MultiIndex (level 0 = block name, level 1 = metric).
    """

    df = cohort_global_dosimetry_by_voxel_df

    # --- columns we must preserve (metadata) ---
    meta_cols = [
        ('Voxel begin (Z)', ''), ('Voxel end (Z)', ''), ('Voxel index', ''),
        ('Patient ID', ''), ('Bx ID', ''), ('Bx index', ''), 
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

    # --- signed deltas ---
    delta_block_name = f'{zero_level_index_str} deltas'
    delta_cols = pd.MultiIndex.from_tuples([
        (delta_block_name, 'nominal_minus_mean'),
        (delta_block_name, 'nominal_minus_mode'),
        (delta_block_name, 'nominal_minus_q50'),
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

    # --- absolute deltas ---
    abs_block_name = f'{zero_level_index_str} abs deltas'
    abs_delta_cols = pd.MultiIndex.from_tuples([
        (abs_block_name, 'abs_nominal_minus_mean'),
        (abs_block_name, 'abs_nominal_minus_mode'),
        (abs_block_name, 'abs_nominal_minus_q50'),
    ])
    abs_deltas = deltas.abs().copy()
    abs_deltas.columns = abs_delta_cols

    # --- assemble output: metadata + signed deltas + abs deltas ---
    out = pd.concat([df.loc[:, meta_cols].copy(), deltas, abs_deltas], axis=1)

    # stable column ordering by first and second level
    desired_cols = pd.MultiIndex.from_tuples(meta_cols + list(delta_cols) + list(abs_delta_cols))
    out = out.reindex(columns=desired_cols)

    # (optional) guardrail for uniqueness within biopsies
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




def save_delta_boxplot_summary_csv_with_absolute(
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

    For EACH delta we write TWO rows:
      1) Signed   :  Δ = nominal - {mean, mode, Q50}
      2) Absolute : |Δ| = abs(nominal - {mean, mode, Q50})

    Stats per row:
      n_voxels, n_biopsies, mean, std, sem, min, q05, q25, q50, q75, q95, max, iqr
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

    # Build tidy series for easy grouping (SIGNED)
    tidy = data.loc[:, cols].copy()
    tidy.columns = ['Nominal - Mean', 'Nominal - Mode', 'Nominal - Median (Q50)']
    tidy = tidy.melt(var_name='Delta', value_name='Value').dropna(subset=['Value'])

    # ABSOLUTE version with clear labels
    tidy_abs = tidy.copy()
    tidy_abs['Value'] = tidy_abs['Value'].abs()
    tidy_abs['Delta'] = '|' + tidy_abs['Delta'] + '|'

    # Combine signed + absolute
    tidy_all = pd.concat([tidy, tidy_abs], ignore_index=True)

    # Helper to compute stats on a Series
    def _summarize(s: pd.Series) -> pd.Series:
        q = s.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
        return pd.Series({
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

    summary = (
        tidy_all
        .groupby('Delta', as_index=True)['Value']
        .apply(_summarize)
        .reset_index()
        .rename(columns={'Value': 'stat'})  # harmless; will be dropped next
    )
    # Drop placeholder column if present
    if 'stat' in summary.columns and summary['stat'].isna().all():
        summary = summary.drop(columns=['stat'])

    # Round numeric columns
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





def save_delta_boxplot_summary_csv_with_absolute_no_recalc(
    deltas_df: pd.DataFrame,
    output_dir,
    csv_name: str,
    *,
    zero_level_index_str: str = 'Dose (Gy)',
    include_patient_ids: list | None = None,
    decimals: int = 3,
    require_precomputed_abs: bool = True,   # <-- new: don't recompute |Δ|; expect abs block present
    fallback_recompute_abs: bool = False    # <-- set True only if you *want* on-the-fly abs as a fallback
) -> Path:
    """
    Summarize the three delta distributions produced by `compute_biopsy_nominal_deltas_with_abs`
    and save as CSV.

    For EACH delta we write TWO rows, both read from the DataFrame:
      1) Signed   : (f"{zero_level_index_str} deltas",     "nominal_minus_[mean|mode|q50]")
      2) Absolute : (f"{zero_level_index_str} abs deltas", "abs_nominal_minus_[mean|mode|q50]")

    No recalculation of |Δ| unless `fallback_recompute_abs=True`.

    Stats per row:
      n_voxels, n_biopsies, mean, std, sem, min, q05, q25, q50, q75, q95, max, iqr
    """
    data = deltas_df

    # Optional patient filter (expects MultiIndex metadata cols as emitted by your compute_* functions)
    if include_patient_ids is not None:
        pid_key = ('Patient ID','')
        if pid_key not in data.columns:
            raise KeyError("Patient filter requested but ('Patient ID','') column not found.")
        data = data[data[pid_key].isin(include_patient_ids)]
        if data.empty:
            raise ValueError("Patient filter returned no rows.")

    # Identify signed and abs delta columns (precomputed)
    block_signed = f"{zero_level_index_str} deltas"
    block_abs    = f"{zero_level_index_str} abs deltas"

    signed_cols = [
        (block_signed, 'nominal_minus_mean'),
        (block_signed, 'nominal_minus_mode'),
        (block_signed, 'nominal_minus_q50'),
    ]
    abs_cols = [
        (block_abs, 'abs_nominal_minus_mean'),
        (block_abs, 'abs_nominal_minus_mode'),
        (block_abs, 'abs_nominal_minus_q50'),
    ]

    # Validate presence
    missing_signed = [c for c in signed_cols if c not in data.columns]
    if missing_signed:
        raise KeyError(
            f"Missing signed delta columns for zero_level_index_str='{zero_level_index_str}'. "
            f"Missing: {missing_signed}"
        )

    missing_abs = [c for c in abs_cols if c not in data.columns]
    if missing_abs:
        if require_precomputed_abs and not fallback_recompute_abs:
            raise KeyError(
                "Absolute delta columns are missing and recomputation is disabled. "
                f"Expected columns: {missing_abs}. "
                "Use DataFrames from compute_biopsy_nominal_deltas_with_abs(), "
                "or set fallback_recompute_abs=True to compute |Δ| on the fly."
            )

    # Count unique biopsies among included rows
    n_biopsies = (
        data.loc[:, [('Patient ID',''), ('Bx index','')]]
        .drop_duplicates()
        .shape[0]
    )

    # --- Build tidy (SIGNED) ---
    tidy_signed = data.loc[:, signed_cols].copy()
    tidy_signed.columns = ['Nominal - Mean', 'Nominal - Mode', 'Nominal - Median (Q50)']
    tidy_signed = tidy_signed.melt(var_name='Delta', value_name='Value').dropna(subset=['Value'])

    # --- Build tidy (ABS) ---
    if not missing_abs:
        tidy_abs = data.loc[:, abs_cols].copy()
        tidy_abs.columns = ['|Nominal - Mean|', '|Nominal - Mode|', '|Nominal - Median (Q50)|']
        tidy_abs = tidy_abs.melt(var_name='Delta', value_name='Value').dropna(subset=['Value'])
    elif fallback_recompute_abs:
        # Explicit fallback path (kept separate for clarity)
        tidy_abs = tidy_signed.copy()
        tidy_abs['Value'] = tidy_abs['Value'].abs()
        tidy_abs['Delta'] = '|' + tidy_abs['Delta'] + '|'
    else:
        # Should never reach here because of the earlier guard
        tidy_abs = pd.DataFrame(columns=['Delta','Value'])

    # Combine
    tidy_all = pd.concat([tidy_signed, tidy_abs], ignore_index=True)

    # Helper to compute stats on a Series
    def _summarize(s: pd.Series) -> pd.Series:
        q = s.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
        return pd.Series({
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

    summary = (
        tidy_all
        .groupby('Delta', as_index=True)['Value']
        .apply(_summarize)
        .reset_index()
        .rename(columns={'Value': 'stat'})  # harmless placeholder to align structure
    )

    if 'stat' in summary.columns and summary['stat'].isna().all():
        summary = summary.drop(columns=['stat'])

    # Round only numeric columns
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
        'Bx ID', 'Voxel begin (Z)', 'Voxel end (Z)',
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



def compute_mc_trial_deltas_with_abs(
    all_voxel_wise_dose_df: pd.DataFrame,
    value_cols: tuple[str, ...] = ('Dose (Gy)', 'Dose grad (Gy/mm)'),
    *,
    trial_col: str = 'MC trial num',
    nominal_trial_value: int | float = 0,
    include_nominal: bool = False,
    group_cols: tuple[str, ...] = ('Patient ID', 'Bx index', 'Voxel index'),
    keep_extra_cols: tuple[str, ...] = (
        'Bx ID', 'Voxel begin (Z)', 'Voxel end (Z)',
        'Simulated bool', 'Simulated type', 'X (Bx frame)', 'Y (Bx frame)',
        'Z (Bx frame)', 'R (Bx frame)'
    ),
    drop_groups_without_nominal: bool = True
) -> pd.DataFrame:
    """
    For each (Patient ID, Bx index, Voxel index) and each MC trial, compute for each v in value_cols:
        signed delta:   Δ_v = v_nominal (trial == nominal_trial_value) - v_trial
        absolute delta: |Δ_v|

    Returns a DataFrame with:
        [group_cols] + present(keep_extra_cols) + [trial_col] + MultiIndex columns:
          (f"{v} deltas",      "nominal_minus_trial")
          (f"{v} abs deltas",  "abs_nominal_minus_trial")
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
        if merged.empty and before > 0:
            raise ValueError("All rows dropped due to missing nominal values for the specified groups.")

    # compute signed deltas per value column
    delta_cols = []
    delta_frames = []
    for v in value_cols:
        nom_series = pd.to_numeric(merged[f"{v} [nominal]"], errors='coerce')
        cur_series = pd.to_numeric(merged[v], errors='coerce')
        d = nom_series - cur_series
        delta_key = (f"{v} deltas", "nominal_minus_trial")
        delta_cols.append(delta_key)
        delta_frames.append(d)

    deltas = pd.concat(delta_frames, axis=1)
    deltas.columns = pd.MultiIndex.from_tuples(delta_cols)

    # compute absolute deltas with parallel MultiIndex names
    abs_cols_map = {
        (f"{v} deltas", "nominal_minus_trial"):
        (f"{v} abs deltas", "abs_nominal_minus_trial")
        for v in value_cols
    }
    abs_deltas = deltas.abs().copy()
    abs_deltas.columns = pd.MultiIndex.from_tuples([abs_cols_map[c] for c in deltas.columns])

    # build output
    out_cols = list(group_cols) + present_extras + [trial_col]
    out = pd.concat([merged[out_cols].copy(), deltas, abs_deltas], axis=1)

    # include/exclude nominal rows
    if not include_nominal:
        out = out[out[trial_col] != nominal_trial_value].copy()

    # order columns so each abs column follows its signed counterpart
    ordered_pairs = []
    for v in value_cols:
        ordered_pairs.append((f"{v} deltas", "nominal_minus_trial"))
        ordered_pairs.append((f"{v} abs deltas", "abs_nominal_minus_trial"))

    desired_order = out_cols + ordered_pairs
    out = out.reindex(columns=desired_order)

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

    Now writes TWO rows per metric:
      - "<metric>"          : signed deltas ( (f"{metric} deltas", "nominal_minus_trial") )
      - "<metric> (abs)"    : absolute deltas ( (f"{metric} abs deltas", "abs_nominal_minus_trial") )

    Reported columns:
        metric, n_rows, n_trials_unique, n_biopsies, n_group_voxels,
        mean, std, sem, min, q05, q25, q50, q75, q95, max, iqr, ipr90
    """
    df = mc_delta_df.copy()

    # Optional filter by patient ids (supports flat or MultiIndex metadata)
    if include_patient_ids is not None:
        if ('Patient ID' not in df.columns) and ('Patient ID', '') not in df.columns:
            raise KeyError("Patient ID column not found for filtering.")
        pid_col = ('Patient ID', '') if ('Patient ID', '') in df.columns else 'Patient ID'
        df = df[df[pid_col].isin(include_patient_ids)].copy()
        if df.empty:
            raise ValueError("Patient filter returned no rows.")

    # Resolve group cols for counting (flat or MultiIndex)
    def _col(key_flat: str):
        return (key_flat, '') if (key_flat, '') in df.columns else key_flat

    pid_c, bxi_c, vox_c = map(_col, ['Patient ID', 'Bx index', 'Voxel index'])
    n_biopsies = (
        df[[pid_c, bxi_c]].drop_duplicates().shape[0]
        if all(c in df.columns for c in [pid_c, bxi_c]) else None
    )
    n_group_voxels = (
        df[[pid_c, bxi_c, vox_c]].drop_duplicates().shape[0]
        if all(c in df.columns for c in [pid_c, bxi_c, vox_c]) else None
    )

    trial_col = 'MC trial num' if 'MC trial num' in df.columns else None
    n_trials_unique = int(df[trial_col].nunique()) if trial_col is not None else None

    # Collect signed + abs delta column keys per metric
    pairs = []
    missing = []
    for v in value_cols:
        signed_key = (f"{v} deltas", "nominal_minus_trial")
        abs_key    = (f"{v} abs deltas", "abs_nominal_minus_trial")
        if signed_key not in df.columns: missing.append(signed_key)
        if abs_key not in df.columns:    missing.append(abs_key)
        pairs.append((v, signed_key, abs_key))
    if missing:
        raise KeyError(
            "Missing expected delta columns. "
            "Did you run compute_mc_trial_deltas (updated version with abs columns)? "
            f"Missing: {missing}"
        )

    # Summarize helper (now includes ipr90 = q95 - q05)
    def _summarize_series(s: pd.Series) -> dict:
        s = pd.to_numeric(s, errors='coerce').dropna()
        if s.empty:
            return None
        q = s.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
        return {
            'n_rows': int(s.shape[0]),
            'n_trials_unique': n_trials_unique,
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
            'ipr90': q.loc[0.95] - q.loc[0.05],
        }

    rows = []
    for metric_name, signed_key, abs_key in pairs:
        stats_signed = _summarize_series(df[signed_key])
        if stats_signed is not None:
            rows.append({'metric': metric_name, **stats_signed})
        stats_abs = _summarize_series(df[abs_key])
        if stats_abs is not None:
            rows.append({'metric': f"{metric_name} (abs)", **stats_abs})

    summary = pd.DataFrame(rows)
    if not summary.empty:
        num_cols = [c for c in summary.columns if c not in ('metric',)]
        summary[num_cols] = summary[num_cols].round(decimals)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not csv_name.lower().endswith('.csv'):
        csv_name += '.csv'
    out_path = output_dir / csv_name
    summary.to_csv(out_path, index=False)
    return out_path



def save_mc_delta_grouped_csvs(
    mc_delta_df: pd.DataFrame,
    output_dir,
    *,
    base_name: str = "mc_trial_deltas",
    value_cols: tuple[str, ...] = ('Dose (Gy)', 'Dose grad (Gy/mm)'),
    decimals: int = 3
) -> tuple[Path, Path]:
    """
    Writes:
      - <base_name>_by_voxel.csv  (aggregate over MC trials; keep Patient ID, Bx index, Voxel index)
      - <base_name>_by_biopsy.csv (aggregate over voxels + MC trials; keep Patient ID, Bx index)
    Columns include: mean, std, sem, min, q05, q25, q50, q75, q95, max, iqr, ipr90.
    """

    df = mc_delta_df.copy()

    def _col(key_flat: str):
        return (key_flat, '') if (key_flat, '') in df.columns else key_flat

    metric_pairs = []
    missing = []
    for v in value_cols:
        signed_key = (f"{v} deltas", "nominal_minus_trial")
        abs_key    = (f"{v} abs deltas", "abs_nominal_minus_trial")
        if signed_key not in df.columns: missing.append(signed_key)
        if abs_key not in df.columns:    missing.append(abs_key)
        metric_pairs.append((v, signed_key, abs_key))
    if missing:
        raise KeyError(f"Missing expected delta columns for grouped summaries: {missing}")

    trial_col = 'MC trial num' if 'MC trial num' in df.columns else None

    # now includes ipr90
    def _summarize_series(s: pd.Series) -> dict | None:
        s = pd.to_numeric(s, errors='coerce').dropna()
        if s.empty:
            return None
        q = s.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
        return {
            'n_rows': int(s.shape[0]),
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
            'ipr90': q.loc[0.95] - q.loc[0.05],
        }

    def _constant_metadata(gdf: pd.DataFrame, exclude_cols: set) -> dict:
        out = {}
        for c in gdf.columns:
            if c in exclude_cols or c == trial_col:
                continue
            vals = gdf[c].drop_duplicates()
            if len(vals) == 1:
                out_key = c if isinstance(c, str) else (c[0] if c[1] == '' else c)
                out[out_key] = vals.iloc[0]
        return out

    def _build_grouped_summary(keys: list[str]) -> pd.DataFrame:
        rows = []
        for group_vals, gdf in df.groupby([_col(k) for k in keys], dropna=False):
            if not isinstance(group_vals, tuple):
                group_vals = (group_vals,)
            group_info = {k: v for k, v in zip(keys, group_vals)}

            n_trials_unique = int(gdf[trial_col].nunique()) if trial_col and trial_col in gdf.columns else None
            n_biopsies = int(gdf[[_col('Patient ID'), _col('Bx index')]].drop_duplicates().shape[0]) \
                         if all(_col(k) in gdf.columns for k in ['Patient ID','Bx index']) else None
            n_group_voxels = int(gdf[[_col('Patient ID'), _col('Bx index'), _col('Voxel index')]].drop_duplicates().shape[0]) \
                             if all(_col(k) in gdf.columns for k in ['Patient ID','Bx index','Voxel index']) else None

            exclude_cols = {trial_col}
            exclude_cols.update({sk for _, sk, _ in metric_pairs})
            exclude_cols.update({ak for _, _, ak in metric_pairs})
            const_meta = _constant_metadata(gdf, exclude_cols)

            for metric_name, signed_key, abs_key in metric_pairs:
                for which, colkey in (('signed', signed_key), ('abs', abs_key)):
                    stats = _summarize_series(gdf[colkey])
                    if stats is None:
                        continue
                    row = {
                        **group_info,
                        **const_meta,
                        'metric': metric_name if which == 'signed' else f"{metric_name} (abs)",
                        'n_trials_unique': n_trials_unique,
                        'n_biopsies': n_biopsies,
                        'n_group_voxels': n_group_voxels,
                        **stats,
                    }
                    rows.append(row)

        out = pd.DataFrame(rows)
        if not out.empty:
            id_like = set(keys) | {'metric'}
            num_cols = [c for c in out.columns if c not in id_like]
            out[num_cols] = out[num_cols].round(decimals)
            out = out.sort_values(keys + ['metric']).reset_index(drop=True)
        return out

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_voxel = _build_grouped_summary(['Patient ID', 'Bx index', 'Voxel index'])
    path_voxel = output_dir / f"{base_name}_by_voxel.csv"
    by_voxel.to_csv(path_voxel, index=False)

    by_biopsy = _build_grouped_summary(['Patient ID', 'Bx index'])
    path_biopsy = output_dir / f"{base_name}_by_biopsy.csv"
    by_biopsy.to_csv(path_biopsy, index=False)

    return path_voxel, path_biopsy




def save_nominal_delta_biopsy_stats(
    nominal_df: pd.DataFrame,
    output_dir,
    *,
    base_name: str = "nominal_deltas",
    value_blocks: tuple[str, ...] = ('Dose (Gy)', 'Dose grad (Gy/mm)'),
    decimals: int = 3,
    include_bx_id: bool = True,
) -> Path:
    """
    Aggregates per-biopsy (Patient ID, Bx index) across voxels for all nominal/abs delta kinds
    present in `nominal_df`.

    For each metric block in `value_blocks`, looks for:
      - (f"{metric} deltas", 'nominal_minus_mean'|'nominal_minus_mode'|'nominal_minus_q50')
      - (f"{metric} abs deltas", 'abs_nominal_minus_mean'|'abs_nominal_minus_mode'|'abs_nominal_minus_q50')

    Outputs one CSV:
      <base_name>_by_biopsy.csv

    Columns include: mean, std, sem, min, q05, q25, q50, q75, q95, max, iqr, ipr90, and counters.
    """
    df = nominal_df.copy()

    # --- helpers ---
    def _col_present(key):
        return key in df.columns

    def _summarize_series(s: pd.Series) -> dict | None:
        s = pd.to_numeric(s, errors='coerce').dropna()
        if s.empty:
            return None
        q = s.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
        return {
            'n_rows': int(s.shape[0]),
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
            'ipr90': q.loc[0.95] - q.loc[0.05],
        }

    # group keys (MultiIndex-safe)
    pid_key = ('Patient ID','') if isinstance(df.columns, pd.MultiIndex) and ('Patient ID','') in df.columns else 'Patient ID'
    bxi_key = ('Bx index','')   if isinstance(df.columns, pd.MultiIndex) and ('Bx index','') in df.columns   else 'Bx index'
    bxid_key= ('Bx ID','')      if isinstance(df.columns, pd.MultiIndex) and ('Bx ID','') in df.columns      else 'Bx ID'
    vx_key  = ('Voxel index','')if isinstance(df.columns, pd.MultiIndex) and ('Voxel index','') in df.columns else 'Voxel index'

    # build the list of metric column mappings actually present
    delta_kinds = (
        ('mean',   'nominal_minus_mean',      'abs_nominal_minus_mean'),
        ('mode',   'nominal_minus_mode',      'abs_nominal_minus_mode'),
        ('median', 'nominal_minus_q50',       'abs_nominal_minus_q50'),
    )

    metric_specs = []
    missing = []
    for metric in value_blocks:
        signed_block = f"{metric} deltas"
        abs_block    = f"{metric} abs deltas"
        for kind, signed_suffix, abs_suffix in delta_kinds:
            signed_key = (signed_block, signed_suffix)
            abs_key    = (abs_block, abs_suffix)
            # tolerate either proper MI keys or flattened strings (rare)
            ok_signed = _col_present(signed_key) or _col_present(f"{signed_block}_{signed_suffix}")
            ok_abs    = _col_present(abs_key)    or _col_present(f"{abs_block}_{abs_suffix}")
            if not ok_signed:
                missing.append(signed_key)
            if not ok_abs:
                missing.append(abs_key)
            metric_specs.append((metric, kind, signed_key, abs_key))
    if missing:
        # Only warn for those truly absent — but continue for present ones
        # You can raise instead if you want strictness:
        # raise KeyError(f"Missing columns: {missing}")
        pass

    rows = []
    group_keys = [pid_key, bxi_key]
    if include_bx_id and bxid_key in df.columns:
        group_keys.append(bxid_key)

    for group_vals, gdf in df.groupby(group_keys, dropna=False):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        group_info = { ('Patient ID' if k==pid_key else 'Bx index' if k==bxi_key else 'Bx ID'): v
                       for k, v in zip(group_keys, group_vals) }

        n_group_voxels = int(gdf[[pid_key, bxi_key, vx_key]].drop_duplicates().shape[0]) \
                         if vx_key in gdf.columns else None

        for metric, kind, signed_key, abs_key in metric_specs:
            # resolve keys if flattened
            if not _col_present(signed_key):
                signed_key = f"{signed_key[0]}_{signed_key[1]}"
            if not _col_present(abs_key):
                abs_key = f"{abs_key[0]}_{abs_key[1]}"

            if _col_present(signed_key):
                sstats = _summarize_series(gdf[signed_key])
                if sstats:
                    rows.append({
                        **group_info,
                        'metric': metric,
                        'delta_kind': kind,          # mean/mode/median (Q50)
                        'signed_or_abs': 'signed',
                        'n_group_voxels': n_group_voxels,
                        **sstats,
                    })
            if _col_present(abs_key):
                astats = _summarize_series(gdf[abs_key])
                if astats:
                    rows.append({
                        **group_info,
                        'metric': metric,
                        'delta_kind': kind,
                        'signed_or_abs': 'abs',
                        'n_group_voxels': n_group_voxels,
                        **astats,
                    })

    out = pd.DataFrame(rows)
    if not out.empty:
        num_cols = [c for c in out.columns if c not in {'Patient ID','Bx index','Bx ID','metric','delta_kind','signed_or_abs'}]
        out[num_cols] = out[num_cols].round(decimals)
        out = out.sort_values(['Patient ID','Bx index','Bx ID' if 'Bx ID' in out.columns else 'metric',
                               'metric','delta_kind','signed_or_abs']).reset_index(drop=True)

    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    path_biopsy = output_dir / f"{base_name}_by_biopsy.csv"
    out.to_csv(path_biopsy, index=False)
    return path_biopsy





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
    exclude_nominal_trial: bool = True,          # drop trial 0 rows if present
    decimals: int = 3,
    eps: float = 0.0,                             # tie tolerance for float comparisons
    # metadata handling
    biopsy_meta_candidates: Iterable[str] = (
        'Bx ID', 'Simulated bool', 'Simulated type'
    ),
    voxel_meta_candidates: Iterable[str] = (
        'Voxel begin (Z)', 'Voxel end (Z)',
        'X (Bx frame)', 'Y (Bx frame)', 'Z (Bx frame)', 'R (Bx frame)'
    ),
) -> Tuple[Path, pd.DataFrame, Path, pd.DataFrame]:
    """
    For each voxel, across all MC trials, compute how often nominal > trial (pos),
    nominal < trial (neg), and equal (zero) — for each metric in value_cols.

    Writes two CSVs:
      - {base_name}_per_voxel.csv : one row per (Patient ID, Bx index, Voxel index, metric) with proportions
                                    + voxel-level metadata (if present)
      - {base_name}_per_biopsy.csv: medians across voxels per biopsy + biopsy-level metadata (if present)

    Returns:
      (Path_to_voxel_csv, voxel_stats_df, Path_to_biopsy_csv, biopsy_stats_df)
    """
    # --- Helpers to robustly keep columns only if present ---
    def _present(cols: Iterable[str]) -> list:
        return [c for c in cols if c in mc_deltas.columns]

    # --- Build long table: keys + metric + diff ---
    parts = []
    for m in value_cols:
        col = (f"{m} deltas", "nominal_minus_trial")
        if col not in mc_deltas.columns:
            raise KeyError(f"Missing column {col}. Run compute_mc_trial_deltas for '{m}'.")
        tmp = mc_deltas[['Patient ID', 'Bx index', 'Voxel index', 'MC trial num']].copy()
        tmp['metric'] = m
        tmp['diff'] = pd.to_numeric(mc_deltas[col], errors='coerce')
        # attach optional metadata (voxel & biopsy)
        for extra in _present(voxel_meta_candidates):
            tmp[extra] = mc_deltas[extra]
        for extra in _present(biopsy_meta_candidates):
            tmp[extra] = mc_deltas[extra]
        parts.append(tmp)

    long = pd.concat(parts, ignore_index=True)

    # Optionally drop nominal trial rows
    if exclude_nominal_trial and 'MC trial num' in long.columns:
        long = long[long['MC trial num'] != 0]

    # proportions per voxel (across trials)
    # use tolerance eps for equality if provided
    if eps > 0:
        gt_mask = long['diff'] >  eps
        eq_mask = long['diff'].between(-eps, eps)
    else:
        gt_mask = long['diff'] >  0
        eq_mask = long['diff'] == 0

    long = long.assign(gt=gt_mask, eq=eq_mask)

    gb_keys_vox = ['Patient ID', 'Bx index', 'Voxel index', 'metric']
    voxel_stats = (
        long
        .groupby(gb_keys_vox, observed=True, sort=False)
        .agg(
            n_trials=('diff','count'),
            prop_nominal_gt_trial=('gt','mean'),
            prop_nominal_eq_trial=('eq','mean'),
        )
        .reset_index()
    )
    voxel_stats['prop_nominal_lt_trial'] = 1.0 - voxel_stats['prop_nominal_gt_trial'] - voxel_stats['prop_nominal_eq_trial']
    voxel_stats['CLES_strict'] = voxel_stats['prop_nominal_gt_trial']
    voxel_stats['CLES_ties']   = voxel_stats['prop_nominal_gt_trial'] + 0.5 * voxel_stats['prop_nominal_eq_trial']

    # attach voxel-level metadata (carry unique values per voxel)
    voxel_keep = _present(voxel_meta_candidates)
    if voxel_keep:
        # take first non-null per voxel/metric (should be constant across trials)
        firsts = (
            long
            .groupby(gb_keys_vox, observed=True, sort=False)[voxel_keep]
            .first()
            .reset_index()
        )
        voxel_stats = voxel_stats.merge(firsts, on=gb_keys_vox, how='left', validate='one_to_one')

    # per-biopsy medians across voxels (robust) + biopsy metadata
    gb_keys_bio = ['Patient ID', 'Bx index', 'metric']
    biopsy_stats = (
        voxel_stats
        .groupby(gb_keys_bio, observed=True, sort=False)
        .agg(
            n_voxels=('n_trials','size'),
            n_trials_median=('n_trials','median'),
            prop_nominal_gt_trial_median=('prop_nominal_gt_trial','median'),
            prop_nominal_eq_trial_median=('prop_nominal_eq_trial','median'),
            prop_nominal_lt_trial_median=('prop_nominal_lt_trial','median'),
            CLES_strict_median=('CLES_strict','median'),
            CLES_ties_median=('CLES_ties','median'),
            CLES_ties_IQR=('CLES_ties', lambda s: (s.quantile(0.75) - s.quantile(0.25))),
        )
        .reset_index()
    )

    # attach biopsy-level metadata (constant per biopsy)
    biopsy_keep = _present(biopsy_meta_candidates)
    if biopsy_keep:
        uniques = (
            long
            .drop_duplicates(subset=['Patient ID','Bx index'])
            [['Patient ID','Bx index'] + biopsy_keep]
        )
        biopsy_stats = biopsy_stats.merge(uniques, on=['Patient ID','Bx index'], how='left', validate='many_to_one')

    # --- rounding: only floats (keep counts as ints)
    def _round_floats(df: pd.DataFrame, nd: int) -> pd.DataFrame:
        fcols = df.select_dtypes(include='float').columns
        df[fcols] = df[fcols].round(nd)
        return df

    voxel_stats = _round_floats(voxel_stats, decimals)
    biopsy_stats = _round_floats(biopsy_stats, decimals)

    # --- save
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
    Accepts both the OLD schema (prop_gt/prop_eq/prop_lt) and the NEW schema
    (prop_nominal_gt_trial/prop_nominal_eq_trial/prop_nominal_lt_trial).
    """

    # Load if file paths were given
    if isinstance(voxel_stats, (str, Path)):
        voxel_stats = pd.read_csv(voxel_stats)
    if isinstance(biopsy_stats, (str, Path)):
        biopsy_stats = pd.read_csv(biopsy_stats)

    vs = voxel_stats.copy()
    bs = biopsy_stats.copy()

    # --- column resolver: supports old & new names
    def _pick(df: pd.DataFrame, *names, required=True):
        for n in names:
            if n in df.columns:
                return n
        if required:
            raise KeyError(f"None of {names} found in DataFrame columns: {list(df.columns)}")
        return None

    col_gt = _pick(vs, 'prop_nominal_gt_trial', 'prop_gt')
    col_eq = _pick(vs, 'prop_nominal_eq_trial', 'prop_eq')
    col_lt = _pick(vs, 'prop_nominal_lt_trial', 'prop_lt', required=False)

    # derive lt if missing
    if col_lt is None:
        vs['__prop_lt__'] = 1.0 - vs[col_gt] - vs[col_eq]
        col_lt = '__prop_lt__'

    # ensure CLES_ties present
    if 'CLES_ties' not in vs.columns:
        vs['CLES_ties'] = vs[col_gt] + 0.5 * vs[col_eq]

    # metrics to summarize (safe intersection in case one file lacks a metric)
    metrics = sorted(set(vs[metrics_col].unique()).intersection(set(bs[metrics_col].unique())))

    rows = []
    for m in metrics:
        vsm = vs[vs[metrics_col] == m].copy()
        bsm = bs[bs[metrics_col] == m].copy()
        if vsm.empty or bsm.empty:
            continue

        # Weights = number of trials contributing to that voxel
        if 'n_trials' not in vsm.columns:
            raise KeyError("Expected column 'n_trials' not found in voxel_stats.")
        w = pd.to_numeric(vsm['n_trials'], errors='coerce').astype(float)
        W = w.sum()
        if (not np.isfinite(W)) or (W <= 0):
            pooled_prop_gt = float('nan')
            pooled_prop_eq = float('nan')
        else:
            pooled_prop_gt = (pd.to_numeric(vsm[col_gt], errors='coerce') * w).sum() / W
            pooled_prop_eq = (pd.to_numeric(vsm[col_eq], errors='coerce') * w).sum() / W
        pooled_prop_lt = 1.0 - pooled_prop_gt - pooled_prop_eq if pd.notna(pooled_prop_gt) else float('nan')

        # Biopsy-level distribution (each biopsy counts once via its per-biopsy median CLES)
        if 'CLES_ties_median' not in bsm.columns:
            raise KeyError("Expected column 'CLES_ties_median' not found in biopsy_stats.")
        cles_b = pd.to_numeric(bsm['CLES_ties_median'], errors='coerce')
        cohort_median = cles_b.median()
        cohort_q25    = cles_b.quantile(0.25)
        cohort_q75    = cles_b.quantile(0.75)

        rows.append({
            'metric': m,
            'n_voxels': int(vsm[['Patient ID','Bx index','Voxel index']].drop_duplicates().shape[0]),
            'n_biopsies': int(bsm[['Patient ID','Bx index']].drop_duplicates().shape[0]),
            'n_trials_total': int(pd.to_numeric(vsm['n_trials'], errors='coerce').sum()),
            'pooled_prop_gt': pooled_prop_gt,
            'pooled_prop_eq': pooled_prop_eq,
            'pooled_prop_lt': pooled_prop_lt,
            'pooled_CLES_strict': pooled_prop_gt,
            'pooled_CLES_ties': (pooled_prop_gt + 0.5 * pooled_prop_eq) if pd.notna(pooled_prop_gt) else float('nan'),
            'biopsy_level_CLES_median': cohort_median,
            'biopsy_level_CLES_q25': cohort_q25,
            'biopsy_level_CLES_q75': cohort_q75,
            'biopsy_level_CLES_IQR': (cohort_q75 - cohort_q25),
        })

    out = pd.DataFrame(rows)

    # Round only float columns (keep counts as ints)
    float_cols = out.select_dtypes(include='float').columns
    out[float_cols] = out[float_cols].round(decimals)

    # Write CSV
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













def summarize_voxel_distributions(df, output_dir, csv_name="voxel_summary.csv", target_col="Dose (Gy)"):
    """
    Summarize per-voxel distributions across Monte Carlo trials.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns including Patient ID, Bx index, Bx ID, Voxel index, MC trial num, and target_col.
    output_dir : str
        Directory to save results.
    csv_name : str
        Name of the output CSV.
    target_col : str
        Column to summarize (default: 'Dose (Gy)').
    """
    os.makedirs(output_dir, exist_ok=True)
    
    group_cols = ["Patient ID", "Bx index", "Bx ID", "Voxel index"]
    results = []

    for keys, sub_df in df.groupby(group_cols):
        values = sub_df[target_col].dropna().values
        if len(values) == 0:
            continue
        
        stats = {
            "Patient ID": keys[0],
            "Bx index": keys[1],
            "Bx ID": keys[2],
            "Voxel index": keys[3],
            "mean": np.mean(values),
            "std": np.std(values, ddof=1),
            "min": np.min(values),
            "max": np.max(values),
            "q05": np.quantile(values, 0.05),
            "q25": np.quantile(values, 0.25),
            "median": np.median(values),
            "q75": np.quantile(values, 0.75),
            "q95": np.quantile(values, 0.95),
        }

        # Gaussian KDE mode
        if np.unique(values).size < 2:
            mode = values[0]
        else:
            try:
                kde = gaussian_kde(values)
                xs = np.linspace(values.min(), values.max(), KDE_GRID_SIZE)
                mode = xs[np.argmax(kde(xs))]
            except LinAlgError:
                mode = stats["median"]
        stats["kde_mode"] = mode
        
        results.append(stats)

    summary_df = pd.DataFrame(results)
    output_path = os.path.join(output_dir, csv_name)
    summary_df.to_csv(output_path, index=False)
    print(f"Saved voxel-wise summary CSV to {output_path}")
    
    return summary_df
