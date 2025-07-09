import pandas as pd
import os
from scipy.stats import gaussian_kde
import numpy as np
from numpy.linalg import LinAlgError
from typing import List, Optional


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