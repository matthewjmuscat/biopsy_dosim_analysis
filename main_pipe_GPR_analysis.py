import matplotlib
matplotlib.use("Agg")  # force non-interactive backend to avoid popping windows

import pandas as pd 
import load_files
from pathlib import Path
import os 
import shape_and_radiomic_features
import misc_funcs
import biopsy_information
import uncertainties_analysis
import production_plots
import pickle
import pathlib # imported for navigating file system
import summary_statistics
import pyarrow # imported for loading parquet files, although not referenced it is required
import helper_funcs
import GPR_analysis_helpers
import GPR_calibration
import numpy as np 
import GPR_analysis_pipeline_functions
import GPR_production_plots
import GPR_kernel_sensitivity
import GPR_semivariogram
import GPR_blocked_cv


def main():
    def _print_section(title: str) -> None:
        line = "=" * 80
        print(f"\n{line}\n{title}\n{line}")


    # =========================================================================
    # Runtime configuration (single-source knobs; keep synchronized with docs)
    # =========================================================================

    # --- Pipeline lane switches ---
    run_semivariogram_plots = False  # if True, write per-biopsy semivariogram figures
    run_patient_plots = False  # if True, write per-biopsy and paired patient figures
    run_kernel_sensitivity_and_calibtration_flag = False  # if True, run full kernel sensitivity lane; if False, run baseline-only calibration outputs
    run_cohort_plots = False  # if True, write cohort-level production figures
    run_blocked_cv = True  # if True, run blocked_CV lane (fold-map + fit/predict stage below)
    run_blocked_cv_fit_predict = True  # if True, run blocked_CV all-kernel train-only fit + held-out predict stage
    run_blocked_cv_plots = True  # if True, run blocked_CV plot lane using in-memory fit/predict artifacts (no CSV rereads)

    # --- Cohort filtering / plot cohort selection ---
    simulated_types = ['Real']  # options: ['Real'], ['Centroid DIL'], ['Optimal DIL'], or mixed subsets
    patient_bx_list = [  # biopsy list used by grid plotters
        ("189 (F2)", 0),
        ("200 (F2)", 0),
        ("201 (F2)", 1),
        ("198 (F2)", 0),
    ]
    grid_ncols = 2  # number of columns for multi-biopsy grid plots
    grid_label_map = {
        ("189 (F2)", 0): "Biopsy 3",
        ("200 (F2)", 0): "Biopsy 4",
        ("201 (F2)", 1): "Biopsy 5",
        ("198 (F2)", 0): "Biopsy 6",
    }  # optional override labels for grid plots
    per_biopsy_label_map = {
        ("188 (F2)", 0): "Biopsy 1",
        ("201 (F2)", 0): "Biopsy 2",
    }  # optional override labels for per-biopsy and paired plots

    # --- GP core methodology ---
    # Recommended default: ordinary kriging on this cohort.
    # Rationale: biopsy-level baseline dose can be nonzero/unknown; zero-mean mode
    # can bias profile level downward when baseline shift exists.
    gp_mean_mode = "ordinary"  # options: "ordinary", "zero"
    gp_target_stat = "median"  # options: "median", "mean"; MC summary used as voxel target y
    gp_position_mode = "begin"  # options: "begin", "center"; voxel z-position used by GP/metrics/plots

    # Recommended default: pairwise semivariogram.
    # Rationale: robust with non-contiguous voxel subsets (blocked_CV train folds) and
    # physically meaningful lag binning in mm.
    semivariogram_method = "pairwise"  # options: "shift", "pairwise"
    run_semivariogram_method_parity_check = True  # if True, save shift-vs-pairwise parity CSVs for QC
    semivariogram_voxel_size_mm = 1.0  # lag axis spacing (mm) used by semivariogram computation
    semivariogram_pairwise_position_mode = gp_position_mode  # pairwise-only z-position mode; kept aligned with gp_position_mode by default
    semivariogram_pairwise_lag_bin_width_mm = None  # pairwise-only lag bin width (mm); None -> semivariogram_voxel_size_mm

    # --- Baseline kernel identity + kernel metadata ---
    # Recommended default from sensitivity runs on this cohort.
    BASE_KERNEL_SPEC = ("rbf", None)  # options: ("matern", 1.5), ("matern", 2.5), ("rbf", None), ("exp", None)
    _KERNEL_LABEL_MAP = {
        ("matern", 1.5): "matern_nu_1_5",
        ("matern", 2.5): "matern_nu_2_5",
        ("rbf", None): "rbf",
        ("exp", None): "exp",
    }
    if BASE_KERNEL_SPEC not in _KERNEL_LABEL_MAP:
        raise ValueError(f"Unsupported BASE_KERNEL_SPEC {BASE_KERNEL_SPEC}. Update _KERNEL_LABEL_MAP to include it.")
    BASE_KERNEL_LABEL = _KERNEL_LABEL_MAP[BASE_KERNEL_SPEC]
    KERNEL_COLOR_MAP = {
        "matern_nu_1_5": "#0b3b8a",  # blue
        "matern_nu_2_5": "#c75000",  # orange
        "rbf": "#2a9d8f",            # green
        "exp": "#7a5195",            # purple
    }  # consistent kernel colors across sensitivity/calibration plot families

    # --- Kernel sensitivity + baseline calibration controls ---
    # When full sensitivity is off, these still control baseline-only calibration outputs.
    calibration_mean_bounds = (-1.0, 1.0)  # heuristic acceptable range for mean standardized residual per biopsy
    calibration_sd_bounds = (0.5, 1.5)  # heuristic acceptable range for SD of standardized residual per biopsy
    calibration_modes_list = [("histogram",), ("histogram", "kde"), ("kde",)]  # output variants for calibration plots

    # --- blocked_CV controls ---
    blocked_cv_output_subdir = "blocked_CV"  # output_data_GPR_analysis subfolder for blocked_CV artifacts

    # Recommended default: fixed_mm with explicit block length.
    # Rationale: in spatial models, holdout difficulty is governed by physical
    # separation distance, not raw voxel count.
    blocked_cv_block_mode = "fixed_mm"  # options: "equal_voxels", "fixed_mm"
    blocked_cv_n_folds = 5  # equal_voxels: direct fold count; fixed_mm + block_length_mm=None: derive length from span / n_folds
    blocked_cv_min_derived_block_mm = 5.0  # fixed_mm + block_length_mm=None only: floor on derived block length
    blocked_cv_block_length_mm = 8.0  # fixed_mm only: explicit block length in mm; set None to derive from span / n_folds

    # fixed_mm cleanup controls
    blocked_cv_merge_tiny_tail_folds = True  # fixed_mm only: merge tiny remainder tail folds
    blocked_cv_min_test_voxels = 3  # fixed_mm cleanup threshold: minimum held-out voxel count per fold
    blocked_cv_min_test_block_mm = 5.0  # fixed_mm cleanup threshold: minimum held-out physical span (mm) per fold
    blocked_cv_min_effective_folds_after_merge = 2  # fixed_mm cleanup guard: never collapse below this effective fold count
    blocked_cv_rebalance_two_fold_splits = True  # fixed_mm only: if 2 folds remain and one violates min_test_* threshold, rebalance to contiguous n/n or n/(n+1) folds

    blocked_cv_target_stat = gp_target_stat  # options: "median", "mean"; blocked_CV target summary statistic
    blocked_cv_mean_mode = gp_mean_mode  # options: "ordinary", "zero"; blocked_CV mean-mode for GP posterior
    blocked_cv_primary_predictive_variance_mode = "observed_mc"  # options: "latent", "observed_mc", "observed_mc_plus_nugget"; canonical blocked_CV standardization mode
    # Recommended default: observed_mc for observed-target calibration reporting.
    # Rationale: denominator matches uncertainty of the observed MC summary target.
    blocked_cv_compare_variance_modes = True  # if True, also score additional variance modes on identical folds/predictions
    blocked_cv_variance_modes_to_compare = ["latent", "observed_mc"]  # each entry must be one of: "latent", "observed_mc", "observed_mc_plus_nugget"
    blocked_cv_kernel_specs = [
        ("matern", 1.5, "matern_nu_1_5"),
        ("matern", 2.5, "matern_nu_2_5"),
        ("rbf", None, "rbf"),
        ("exp", None, "exp"),
    ]  # kernel list for blocked_CV run loop
    # None -> run all blocked_cv_kernel_specs; or explicit subset list of labels (3rd tuple entry in blocked_cv_kernel_specs).
    # Valid labels with current defaults: ["matern_nu_1_5", "matern_nu_2_5", "rbf", "exp"].
    # Example: blocked_cv_kernel_labels_to_run = ["rbf"] or ["matern_nu_1_5", "rbf"].
    blocked_cv_kernel_labels_to_run = None

    # blocked_CV output toggles
    write_blocked_cv_eligible_views = True  # if True, also write *_eligible CSV views and eligibility exclusions table
    # Debug-volume toggle: report/repro tables are always produced; this controls only large raw debug tables.
    write_blocked_cv_debug_csvs = True  # if False, skip fold-map and point-level prediction CSVs

    # blocked_CV optional per-kernel slices (all are strict subsets of *_all)
    write_blocked_cv_per_kernel_predictions_csvs = False
    write_blocked_cv_per_kernel_fit_status_csvs = False
    write_blocked_cv_per_kernel_variance_compare_csvs = False
    write_blocked_cv_per_kernel_variance_summary_csvs = False

    # blocked_CV plotting controls (independent from baseline plotting knobs)
    blocked_cv_plot_patient_bx_list = [
        ("189 (F2)", 0),
        ("200 (F2)", 0),
        ("201 (F2)", 1),
        ("198 (F2)", 0),
    ]  # grid subset for blocked_CV figures; set None to skip grid subset restriction
    blocked_cv_plot_grid_ncols = 2  # number of columns for blocked_CV grid layouts
    blocked_cv_plot_grid_label_map = {
        ("189 (F2)", 0): "Biopsy 3",
        ("200 (F2)", 0): "Biopsy 4",
        ("201 (F2)", 1): "Biopsy 5",
        ("198 (F2)", 0): "Biopsy 6",
    }  # optional blocked_CV label overrides for grid figures
    blocked_cv_plot_fold_ids = None  # None -> all folds; or explicit list (e.g., [0, 1])
    blocked_cv_plot_max_folds_per_biopsy = None  # None -> no cap; otherwise limit folds shown per biopsy
    blocked_cv_plot_fold_sort_mode = "fold_id"  # options: "fold_id", "z_start_mm"; deterministic fold display order
    blocked_cv_plot_include_merged_tail_folds = True  # if False, exclude merged-tail folds from blocked_CV figure generation
    blocked_cv_plot_include_rebalanced_two_fold_splits = True  # if False, exclude rebalanced-two-fold cases from blocked_CV figure generation
    # None -> all kernels included in this blocked_CV run; or explicit label subset.
    # Labels must match kernel labels used by blocked_cv_kernel_specs (e.g., "matern_nu_1_5", "matern_nu_2_5", "rbf", "exp").
    blocked_cv_plot_kernel_labels = None
    blocked_cv_plot_variance_mode = "primary"  # options: "primary", "latent", "observed_mc", "observed_mc_plus_nugget"
    # Centralized blocked_CV plot gating to avoid one variable per figure type.
    # Implemented keys currently used: paired_semivariogram_profile, profile_grids, semivariogram_grids,
    # write_report_figures, write_diagnostic_figures.
    # Placeholder keys (for upcoming figure families): residuals, calibration, kernel_comparison.
    blocked_cv_plot_options = {
        "paired_semivariogram_profile": False,
        "profile_grids": True,
        "semivariogram_grids": True,
        "semivariogram_show_n_pairs": True,  # if True, annotate semivariogram points with faint 'n=' pair-count labels
        "semivariogram_n_pairs_fontsize": 5.0,  # fontsize for semivariogram n-pairs annotations
        "residuals": False,
        "calibration": False,
        "kernel_comparison": False,
        "write_report_figures": True,
        "write_diagnostic_figures": False,
    }

    # --- Plot presentation toggles ---
    include_kernel_legend_in_primary_histograms = True  # if True, append kernel label on primary single-kernel plot legends/axes where supported

    # --- CSV output toggles ---
    # Full CSV column definitions are in `GPR_CSV_DATA_DICTIONARY.md`.
    write_split_main_cohort_summary_csvs = False  # if True, write cohort summary split CSVs in addition to consolidated summary
    write_sensitivity_per_kernel_metrics_csvs = False  # if True, write per-kernel metric slices for sensitivity lane
    write_sensitivity_per_kernel_calibration_csvs = False  # if True, write per-kernel calibration slices for sensitivity lane





    # GPR pipeline phases:
    # 1. Load MC voxelwise dose table
    # 2. Per-biopsy extraction
    # 3. Per-biopsy semivariogram + Matérn fit
    # 4. Per-biopsy GP posterior
    # 5. Per-biopsy metrics
    # 6. Cohort aggregation and plots




    _print_section("GPR PIPELINE: DATA LOADING")
    ### Set main output path ###
    #main_output_path = Path("/home/matthew-muscat/Documents/UBC/Research/Data/Output data/MC_sim_out- Date-May-15-2025 Time-18,11,24")
    main_output_path = Path("/home/matthew-muscat/Documents/UBC/Research/Data/Output data/MC_sim_out- Date-Jan-04-2026 Time-11,55,49 -- 15 patients F2 only cohort with simulated centroid and optimal bxs - good for dosim or GPR analysis")  # Update this path to your specific output directory

    
    ### Load Dataframes 

    # Set csv directory
    csv_directory = main_output_path.joinpath("Output CSVs")
    cohort_csvs_directory = csv_directory.joinpath("Cohort")
    
    
    _print_section("LOAD: Voxel-wise dose tables")
    ### Load all individual bx csvs and concatenate ### (START)

    mc_sim_results_path = csv_directory.joinpath("MC simulation")  # Ensure the directory is a Path object



    ### 1. Voxel wise dose output by MC trial number
    all_paths_voxel_wise_dose_output = load_files.find_csv_files(mc_sim_results_path, ['Voxel-wise dose output by MC trial number.parquet'])
    # Load and concatenate
    # Loop through all the paths and load the csv files
    all_voxel_wise_dose_dfs_list = []
    for path in all_paths_voxel_wise_dose_output:
        # Load the csv file into a dataframe
        #df = load_files.load_csv_as_dataframe(path)
        df = load_files.load_parquet_as_dataframe(path)
        # Append the dataframe to the list
        all_voxel_wise_dose_dfs_list.append(df)
        del df
    # Concatenate all the dataframes into one dataframe
    all_voxel_wise_dose_df = pd.concat(all_voxel_wise_dose_dfs_list, ignore_index=True)
    """ NOTE: The columns of the dataframe are:
    print(all_voxel_wise_dose_df.columns)
        Index(['Voxel index', 'MC trial num', 'Dose (Gy)', 'Dose grad (Gy/mm)',
            'X (Bx frame)', 'Y (Bx frame)', 'Z (Bx frame)', 'R (Bx frame)',
            'Simulated bool', 'Simulated type', 'Bx refnum', 'Bx index', 'Bx ID',
            'Patient ID', 'Voxel begin (Z)', 'Voxel end (Z)'],
            dtype='object')
    """
    del all_voxel_wise_dose_dfs_list
    # Print the shape of the dataframe
    print(f"Shape of all voxel wise dose dataframe: {all_voxel_wise_dose_df.shape}")
    # Print the columns of the dataframe
    print(f"Columns of all voxel wise dose dataframe: {all_voxel_wise_dose_df.columns}")
    # Print the first 5 rows of the dataframe
    print(f"First 5 rows of all voxel wise dose dataframe:\n {all_voxel_wise_dose_df.head()}")
    # Print the last 5 rows of the dataframe
    print(f"Last 5 rows of all voxel wise dose dataframe:\n {all_voxel_wise_dose_df.tail()}")
    # voxel wise dose output by MC trial number (END)




    _print_section("LOAD: Cohort global dosimetry by voxel")
    # Cohort global dosimetry by voxel
    cohort_global_dosimetry_by_voxel_path = cohort_csvs_directory.joinpath("Cohort: Global dosimetry by voxel.csv")  # Ensure the directory is a Path object
    # this is a multiindex dataframe
    cohort_global_dosimetry_by_voxel_df = load_files.load_multiindex_csv(cohort_global_dosimetry_by_voxel_path, header_rows=[0, 1])  # Load the CSV file into a DataFrame
    # Note that each biopsy has unique voxels, ie no biopsy has more than one voxel i 
    """ NOTE: The columns of the dataframe are:
    print(cohort_global_dosimetry_by_voxel_df.columns)
	MultiIndex([(  'Voxel begin (Z)',               ''),
            (    'Voxel end (Z)',               ''),
            (      'Voxel index',               ''),
            (       'Patient ID',               ''),
            (            'Bx ID',               ''),
            (         'Bx index',               ''),
            (        'Bx refnum',               ''),
            (   'Simulated bool',               ''),
            (   'Simulated type',               ''),
            (        'Dose (Gy)', 'argmax_density'),
            (        'Dose (Gy)',       'kurtosis'),
            (        'Dose (Gy)',            'max'),
            (        'Dose (Gy)',           'mean'),
            (        'Dose (Gy)',            'min'),
            (        'Dose (Gy)',        'nominal'),
            (        'Dose (Gy)',    'quantile_05'),
            (        'Dose (Gy)',    'quantile_25'),
            (        'Dose (Gy)',    'quantile_50'),
            (        'Dose (Gy)',    'quantile_75'),
            (        'Dose (Gy)',    'quantile_95'),
            (        'Dose (Gy)',            'sem'),
            (        'Dose (Gy)',       'skewness'),
            (        'Dose (Gy)',            'std'),
            ('Dose grad (Gy/mm)', 'argmax_density'),
            ('Dose grad (Gy/mm)',       'kurtosis'),
            ('Dose grad (Gy/mm)',            'max'),
            ('Dose grad (Gy/mm)',           'mean'),
            ('Dose grad (Gy/mm)',            'min'),
            ('Dose grad (Gy/mm)',        'nominal'),
            ('Dose grad (Gy/mm)',    'quantile_05'),
            ('Dose grad (Gy/mm)',    'quantile_25'),
            ('Dose grad (Gy/mm)',    'quantile_50'),
            ('Dose grad (Gy/mm)',    'quantile_75'),
            ('Dose grad (Gy/mm)',    'quantile_95'),
            ('Dose grad (Gy/mm)',            'sem'),
            ('Dose grad (Gy/mm)',       'skewness'),
            ('Dose grad (Gy/mm)',            'std')],
           )
    """

    # Add IQR (Q75 - Q25) and 90% IPR (Q95 - Q05) for both Dose and Dose Gradient
    for col in ["Dose (Gy)", "Dose grad (Gy/mm)"]:
        cohort_global_dosimetry_by_voxel_df[(col, "IQR")] = (
            cohort_global_dosimetry_by_voxel_df[(col, "quantile_75")] 
            - cohort_global_dosimetry_by_voxel_df[(col, "quantile_25")]
        )
        cohort_global_dosimetry_by_voxel_df[(col, "IPR90")] = (
            cohort_global_dosimetry_by_voxel_df[(col, "quantile_95")] 
            - cohort_global_dosimetry_by_voxel_df[(col, "quantile_05")]
        )
    """ Now the columns of the dataframe are:
        print(cohort_global_dosimetry_by_voxel_df.columns)
        MultiIndex([(  'Voxel begin (Z)',               ''),
                    (    'Voxel end (Z)',               ''),
                    (      'Voxel index',               ''),
                    (       'Patient ID',               ''),
                    (            'Bx ID',               ''),
                    (         'Bx index',               ''),
                    (        'Bx refnum',               ''),
                    (   'Simulated bool',               ''),
                    (   'Simulated type',               ''),
                    (        'Dose (Gy)', 'argmax_density'),
                    (        'Dose (Gy)',       'kurtosis'),
                    (        'Dose (Gy)',            'max'),
                    (        'Dose (Gy)',           'mean'),
                    (        'Dose (Gy)',            'min'),
                    (        'Dose (Gy)',        'nominal'),
                    (        'Dose (Gy)',    'quantile_05'),
                    (        'Dose (Gy)',    'quantile_25'),
                    (        'Dose (Gy)',    'quantile_50'),
                    (        'Dose (Gy)',    'quantile_75'),
                    (        'Dose (Gy)',    'quantile_95'),
                    (        'Dose (Gy)',            'sem'),
                    (        'Dose (Gy)',       'skewness'),
                    (        'Dose (Gy)',            'std'),
                    ('Dose grad (Gy/mm)', 'argmax_density'),
                    ('Dose grad (Gy/mm)',       'kurtosis'),
                    ('Dose grad (Gy/mm)',            'max'),
                    ('Dose grad (Gy/mm)',           'mean'),
                    ('Dose grad (Gy/mm)',            'min'),
                    ('Dose grad (Gy/mm)',        'nominal'),
                    ('Dose grad (Gy/mm)',    'quantile_05'),
                    ('Dose grad (Gy/mm)',    'quantile_25'),
                    ('Dose grad (Gy/mm)',    'quantile_50'),
                    ('Dose grad (Gy/mm)',    'quantile_75'),
                    ('Dose grad (Gy/mm)',    'quantile_95'),
                    ('Dose grad (Gy/mm)',            'sem'),
                    ('Dose grad (Gy/mm)',       'skewness'),
                    ('Dose grad (Gy/mm)',            'std'),
                    (        'Dose (Gy)',            'IQR'),
                    (        'Dose (Gy)',          'IPR90'),
                    ('Dose grad (Gy/mm)',            'IQR'),
                    ('Dose grad (Gy/mm)',          'IPR90')],
                )
    """


    print(f"done loading cohort global dosimetry by voxel dataframe. Shape: {cohort_global_dosimetry_by_voxel_df.shape}")

    ### Load all individual bx csvs and concatenate ### (END)



    # ---------------------------------------    
    # # Filter dataframes by simulated types (optional)
    # ---------------------------------------

    _print_section("FILTER: Simulated types")
    # filter dataframes by simulated types
    if simulated_types is not None:
        all_voxel_wise_dose_df = all_voxel_wise_dose_df[
            all_voxel_wise_dose_df['Simulated type'].isin(simulated_types)
        ].reset_index(drop=True)

        cohort_global_dosimetry_by_voxel_df = cohort_global_dosimetry_by_voxel_df[
            cohort_global_dosimetry_by_voxel_df[('Simulated type','')].isin(simulated_types)
        ].reset_index(drop=True)

        print(f"Filtered dataframes by simulated types: {simulated_types}")
        print(f"Shape of filtered all voxel wise dose dataframe: {all_voxel_wise_dose_df.shape}")
        print(f"Shape of filtered cohort global dosimetry by voxel dataframe: {cohort_global_dosimetry_by_voxel_df.shape}")
        # To determine number of biopsies we need number of unique patient ID and bx index pairs 
        print(f"Number of biopsies after filtering: {cohort_global_dosimetry_by_voxel_df[['Patient ID','Bx index']].drop_duplicates().shape[0]}")
        print(f"Number of patients after filtering: {cohort_global_dosimetry_by_voxel_df['Patient ID'].nunique()}")




    # ---------------------------------------    
    # # Create output directories
    # ---------------------------------------

    ## Create output directory
    # Output directory 
    output_dir = Path(__file__).parents[0].joinpath("output_data_GPR_analysis")
    os.makedirs(output_dir, exist_ok=True)
    # make dirs
    output_fig_directory = output_dir.joinpath("figures")
    os.makedirs(output_fig_directory, exist_ok=True)
    cohort_output_figures_dir = output_fig_directory.joinpath("cohort_output_figures")
    os.makedirs(cohort_output_figures_dir, exist_ok=True)
    pt_sp_figures_dir = output_fig_directory.joinpath("patient_specific_output_figures")
    os.makedirs(pt_sp_figures_dir, exist_ok=True)











    # ---------------------------------------    
    # # Cross-check voxelwise stats vs cohort summary
    # ---------------------------------------

    _print_section("QC: Voxelwise stats cross-check")
    cross_summary_df, cross_mismatches_df = helper_funcs.cross_check_voxelwise_statistics(
        mc_voxel_df=all_voxel_wise_dose_df,
        cohort_voxel_df=cohort_global_dosimetry_by_voxel_df,
        value_cols=("Dose (Gy)", "Dose grad (Gy/mm)"),
        key_cols=("Patient ID", "Bx index", "Bx ID", "Voxel index"),
        nominal_trial_num=0,
        rtol=1e-3,
        atol=1e-4,
        verbose=True,
        max_mismatch_rows=50,
        compute_mode=False,
    )

    cross_summary_path = output_dir.joinpath("voxel_stats_cross_check_summary.csv")
    cross_summary_df.to_csv(cross_summary_path, index=False)
    print(f"Saved voxelwise stats cross-check summary to: {cross_summary_path}")

    if not cross_mismatches_df.empty:
        cross_mismatch_path = output_dir.joinpath("voxel_stats_cross_check_mismatches_sample.csv")
        cross_mismatches_df.to_csv(cross_mismatch_path, index=False)
        print(f"Saved sample mismatches to: {cross_mismatch_path}")
    else:
        print("Voxelwise stats cross-check: no mismatches beyond tolerance.")


    # Good they seems to match within tolerance, so I will continue to just calcualte desired stats throughout the piupeline from the raw data for now.











    # ---------------------------------------    
    # # Semivariogram analysis
    # ---------------------------------------


    _print_section("SEMIVARIOGRAM: Compute per-biopsy")
   

    semivariogram_df = GPR_analysis_helpers.semivariogram_by_biopsy(
        all_voxel_wise_dose_df,
        voxel_size_mm=semivariogram_voxel_size_mm,
        max_lag_voxels=None,
        method=semivariogram_method,
        position_mode=semivariogram_pairwise_position_mode,
        lag_bin_width_mm=semivariogram_pairwise_lag_bin_width_mm,
    )
    """NOTE: The columns of the dataframe are:
    Index(['lag_voxels', 'h_mm', 'semivariance', 'n_pairs', 'Patient ID',
       'Bx index', 'Simulated bool', 'Simulated type', 'Bx refnum', 'Bx ID',
       'n_trials', 'n_voxels'],
      dtype='object')"""
    print(semivariogram_df)

    if run_semivariogram_method_parity_check:
        parity_dir = output_dir.joinpath("semivariogram_method_parity")
        parity_dir.mkdir(parents=True, exist_ok=True)
        parity_summary_df, parity_detail_df = GPR_semivariogram.compare_semivariogram_methods_by_biopsy(
            all_voxel_wise_dose_df,
            voxel_size_mm=semivariogram_voxel_size_mm,
            max_lag_voxels=None,
            position_mode=semivariogram_pairwise_position_mode,
            lag_bin_width_mm=semivariogram_pairwise_lag_bin_width_mm,
        )
        parity_summary_path = parity_dir.joinpath("semivariogram_method_parity_summary.csv")
        parity_detail_path = parity_dir.joinpath("semivariogram_method_parity_differences.csv")
        parity_summary_df.to_csv(parity_summary_path, index=False)
        parity_detail_df.to_csv(parity_detail_path, index=False)
        print(f"[semivariogram parity] summary saved to: {parity_summary_path}")
        print(f"[semivariogram parity] details saved to: {parity_detail_path}")
        if not parity_summary_df.empty:
            max_row = parity_summary_df.iloc[0]
            print(
                "[semivariogram parity] "
                f"worst biopsy: Patient {max_row['Patient ID']}, Bx {max_row['Bx index']}, "
                f"max |Δ semivariance|={max_row['max_abs_diff_semivariance']:.6g}"
            )

    # Sanity check:
    # RMS difference from semivariogram
    #semivariogram_df['rms_diff'] = np.sqrt(2 * semivariogram_df['semivariance'])
    #print(semivariogram_df)
    #print('test')
    # Note that these differ a bit because the distributions are non gaussian




    # Plot semivariogram for each biopsy (optional can be skipped for speed, but useful for QC and visualization of spatial structure)
    if run_semivariogram_plots:
        _print_section("SEMIVARIOGRAM: Plot per-biopsy")
        for patient_id, bx_index in semivariogram_df.groupby(['Patient ID', 'Bx index']).groups.keys():
            print(f"Plotting semivariogram for Patient ID: {patient_id}, Bx index: {bx_index}")

            patient_dir = pt_sp_figures_dir.joinpath(patient_id)
            os.makedirs(patient_dir, exist_ok=True)
            sv_dir = patient_dir.joinpath("semivariograms")
            os.makedirs(sv_dir, exist_ok=True)

            # Plot the semivariogram for each biopsy (production-quality)
            GPR_production_plots.plot_variogram_from_df(
                semivariogram_df,
                patient_id,
                bx_index,
                overlay_df=None,  # optional precomputed overlay with columns ['h_mm', 'median_absdiff', 'mean_absdiff'] (any subset ok)
                include_title_meta=False,  # paper-ready: keep title off, caption will hold metadata
                label_fontsize=16,
                tick_labelsize=14,
                title_fontsize=16,
                legend_fontsize=13,
                save_path=sv_dir,     # directory for semivariograms per patient
                file_name=f"semivariogram_patient_{patient_id}_bx_{bx_index}",  # base name, extension handled by save_formats
                save_formats=("pdf", "svg"),     # defaults to vector formats for publication; add "png" if needed
                dpi=400,
            )
            print(f"Saved semivariogram plot for Patient ID: {patient_id}, Bx index: {bx_index} to {patient_dir}")
    else:
        _print_section("SEMIVARIOGRAM: Plot per-biopsy (skipped)")











    # ---------------------------------------    
    # # GP + metrics
    # ---------------------------------------


    _print_section("GP: Run per-biopsy + metrics")
    # Run per-biopsy Matérn GP on the filtered cohort, derive per-biopsy metrics,
    # and write cohort-level summary/rollup CSVs (metrics, summary numbers, patient rollups).
    nu_arg = BASE_KERNEL_SPEC[1] if BASE_KERNEL_SPEC[0] == "matern" else None
    results, metrics_df, cohort_summary_df, by_patient = GPR_analysis_helpers.run_gp_and_collect_metrics(
        all_voxel_wise_dose_df=all_voxel_wise_dose_df,
        semivariogram_df=semivariogram_df,
        output_dir=output_dir,
        target_stat=gp_target_stat,
        nu=nu_arg,
        kernel_spec=BASE_KERNEL_SPEC,
        kernel_label=BASE_KERNEL_LABEL,
        position_mode=gp_position_mode,
        mean_mode=gp_mean_mode,
        save_split_cohort_summary_csvs=write_split_main_cohort_summary_csvs,
    )










    # ---------------------------------------    
    # # kernel sensitivity and calibration
    # ---------------------------------------


    # Optional kernel sensitivity block (kept minimal to avoid clutter)

    # create kernel sensitivity output directories
    kernel_sens_dir = output_dir.joinpath("kernel_sensitivity")
    kernel_sens_dir.mkdir(parents=True, exist_ok=True)
    kernel_figs_dir = kernel_sens_dir.joinpath("figures")
    kernel_csv_dir = kernel_sens_dir.joinpath("csv")
    kernel_figs_dir.mkdir(parents=True, exist_ok=True)
    kernel_csv_dir.mkdir(parents=True, exist_ok=True)

    # I have created two paths here so be careful if changing code around calibration because depending on which path you take outputs could be different depending on whether youre
    # careful about changing code. in particular the save paths of the csvs! Also be careful about selecting which bounds to use for heurisitic acceptable band for standardized residuals std and mean 
    if run_kernel_sensitivity_and_calibtration_flag:
        _print_section("KERNEL SENSITIVITY ANALYSIS")
        GPR_kernel_sensitivity.run_kernel_sensitivity(
            all_voxel_wise_dose_df=all_voxel_wise_dose_df,
            semivariogram_df=semivariogram_df,
            output_dir=kernel_sens_dir,
            figs_dir=kernel_figs_dir,
            csv_dir=kernel_csv_dir,
            target_stat=gp_target_stat,
            position_mode=gp_position_mode,
            mean_mode=gp_mean_mode,
            kernel_color_map=KERNEL_COLOR_MAP,
            save_per_kernel_metrics_csvs=write_sensitivity_per_kernel_metrics_csvs,
            save_per_kernel_calibration_csvs=write_sensitivity_per_kernel_calibration_csvs,
        )
    else:
        # I created a second path because I need the calibration to run for at least the baseline kernel in order to have calibration metrics and plots for the paper, but I want to be able to skip the rest of the sensitivity analysis if I dont need it to save time. 
        # So this way I can skip the sensitivity analysis but still get the calibration outputs for the baseline kernel.
        _print_section("KERNEL SENSITIVITY ANALYSIS (calibration only for baseline kernel)")
        
        calib_df_base = GPR_calibration.build_calibration_metrics(
            results,
            mean_bounds=calibration_mean_bounds,
            sd_bounds=calibration_sd_bounds,
        )
        calib_csv_base = kernel_csv_dir / f"calibration_metrics_{BASE_KERNEL_LABEL}.csv"
        calib_df_base.to_csv(calib_csv_base, index=False)
        calib_fig_dir_base = kernel_figs_dir / f"calibration_{BASE_KERNEL_LABEL}"
        calib_fig_dir_base.mkdir(parents=True, exist_ok=True)
        GPR_production_plots.calibration_plots_production(
            calib_df=calib_df_base,
            save_dir=calib_fig_dir_base,
            save_formats=("pdf", "svg"),
            mean_bounds=calibration_mean_bounds,
            sd_bounds=calibration_sd_bounds,
            modes_list=calibration_modes_list,
            kernel_color_map=KERNEL_COLOR_MAP,
            kernel_suffix=BASE_KERNEL_LABEL,
        )
        print(f"[calibration] baseline kernel saved to {calib_csv_base} and {calib_fig_dir_base}")











    # ---------------------------------------
    # blocked_CV lane (fold mapping + optional all-kernel fit/predict)
    # ---------------------------------------
    if run_blocked_cv:
        _print_section("BLOCKED_CV: Fold Mapping")
        blocked_cv_root, blocked_cv_figs_dir, blocked_cv_csv_dir = GPR_blocked_cv.init_blocked_cv_dirs(
            output_dir, subdir_name=blocked_cv_output_subdir
        )
        blocked_cv_kernel_specs_use = list(blocked_cv_kernel_specs)
        if blocked_cv_kernel_labels_to_run is not None:
            allowed_kernel_labels = {str(k) for k in blocked_cv_kernel_labels_to_run}
            blocked_cv_kernel_specs_use = [spec for spec in blocked_cv_kernel_specs_use if str(spec[2]) in allowed_kernel_labels]
            if not blocked_cv_kernel_specs_use:
                raise ValueError(
                    "blocked_cv_kernel_labels_to_run filtered out all kernel specs. "
                    "Provide labels that exist in blocked_cv_kernel_specs."
                )
        blocked_cv_cfg = GPR_blocked_cv.BlockedCVConfig(
            block_mode=blocked_cv_block_mode,
            n_folds=blocked_cv_n_folds,
            block_length_mm=blocked_cv_block_length_mm,
            min_derived_block_mm=blocked_cv_min_derived_block_mm,
            merge_tiny_tail_folds=blocked_cv_merge_tiny_tail_folds,
            min_test_voxels=blocked_cv_min_test_voxels,
            min_test_block_mm=blocked_cv_min_test_block_mm,
            min_effective_folds_after_merge=blocked_cv_min_effective_folds_after_merge,
            rebalance_two_fold_splits=blocked_cv_rebalance_two_fold_splits,
            position_mode=semivariogram_pairwise_position_mode,
            target_stat=blocked_cv_target_stat,
            mean_mode=blocked_cv_mean_mode,
            primary_predictive_variance_mode=blocked_cv_primary_predictive_variance_mode,
            compare_variance_modes=blocked_cv_compare_variance_modes,
            variance_modes_to_compare=blocked_cv_variance_modes_to_compare,
            kernel_specs=blocked_cv_kernel_specs_use,
            semivariogram_voxel_size_mm=semivariogram_voxel_size_mm,
            semivariogram_lag_bin_width_mm=semivariogram_pairwise_lag_bin_width_mm,
            write_debug_csvs=write_blocked_cv_debug_csvs,
            write_eligible_views=write_blocked_cv_eligible_views,
            write_per_kernel_predictions_csvs=write_blocked_cv_per_kernel_predictions_csvs,
            write_per_kernel_fit_status_csvs=write_blocked_cv_per_kernel_fit_status_csvs,
            write_per_kernel_variance_compare_csvs=write_blocked_cv_per_kernel_variance_compare_csvs,
            write_per_kernel_variance_summary_csvs=write_blocked_cv_per_kernel_variance_summary_csvs,
            plot_patient_bx_list=blocked_cv_plot_patient_bx_list,
            plot_grid_ncols=blocked_cv_plot_grid_ncols,
            plot_grid_label_map=blocked_cv_plot_grid_label_map,
            plot_fold_ids=blocked_cv_plot_fold_ids,
            plot_max_folds_per_biopsy=blocked_cv_plot_max_folds_per_biopsy,
            plot_fold_sort_mode=blocked_cv_plot_fold_sort_mode,
            plot_include_merged_tail_folds=blocked_cv_plot_include_merged_tail_folds,
            plot_include_rebalanced_two_fold_splits=blocked_cv_plot_include_rebalanced_two_fold_splits,
            plot_kernel_labels=blocked_cv_plot_kernel_labels,
            plot_variance_mode=blocked_cv_plot_variance_mode,
            plot_make_paired_semivariogram_profile=bool(blocked_cv_plot_options.get("paired_semivariogram_profile", False)),
            plot_make_semivariogram_grids=bool(blocked_cv_plot_options.get("semivariogram_grids", False)),
            plot_make_profile_grids=bool(blocked_cv_plot_options.get("profile_grids", False)),
            plot_semivariogram_show_n_pairs=bool(blocked_cv_plot_options.get("semivariogram_show_n_pairs", False)),
            plot_semivariogram_n_pairs_fontsize=float(blocked_cv_plot_options.get("semivariogram_n_pairs_fontsize", 5.0)),
            plot_write_report_figures=bool(blocked_cv_plot_options.get("write_report_figures", True)),
            plot_write_diagnostic_figures=bool(blocked_cv_plot_options.get("write_diagnostic_figures", False)),
        )
        blocked_cv_status = GPR_blocked_cv.run_blocked_cv_phase3b(
            all_voxel_wise_dose_df=all_voxel_wise_dose_df,
            semivariogram_df=semivariogram_df,
            output_dir=blocked_cv_root,
            figs_dir=blocked_cv_figs_dir,
            csv_dir=blocked_cv_csv_dir,
            config=blocked_cv_cfg,
        )
        print(f"[blocked_CV] fold-mapping status: {blocked_cv_status}")
        if run_blocked_cv_fit_predict:
            _print_section("BLOCKED_CV: All-kernel Train-only Fit + Held-out Predict")
            blocked_cv_fit_predict_result = GPR_blocked_cv.run_blocked_cv_fit_predict(
                all_voxel_wise_dose_df=all_voxel_wise_dose_df,
                semivariogram_df=semivariogram_df,
                output_dir=blocked_cv_root,
                figs_dir=blocked_cv_figs_dir,
                csv_dir=blocked_cv_csv_dir,
                config=blocked_cv_cfg,
            )
            blocked_cv_fit_predict_status = blocked_cv_fit_predict_result["status"]
            blocked_cv_fit_predict_artifacts = blocked_cv_fit_predict_result["artifacts"]
            print(f"[blocked_CV] fit/predict status: {blocked_cv_fit_predict_status}")
            if run_blocked_cv_plots:
                _print_section("BLOCKED_CV: Plots")
                blocked_cv_plot_status = GPR_blocked_cv.run_blocked_cv_plots(
                    fit_predict_artifacts=blocked_cv_fit_predict_artifacts,
                    all_voxel_wise_dose_df=all_voxel_wise_dose_df,
                    output_dir=blocked_cv_root,
                    figs_dir=blocked_cv_figs_dir,
                    csv_dir=blocked_cv_csv_dir,
                    config=blocked_cv_cfg,
                )
                print(f"[blocked_CV] plot status: {blocked_cv_plot_status}")
    else:
        _print_section("BLOCKED_CV: Skipped")










    # ---------------------------------------    
    # # biopsy-level production plots
    # ---------------------------------------

    if run_patient_plots:
        _print_section("PLOTS: Patient-level figures")

        # Create grid directory
        grid_dir = pt_sp_figures_dir.joinpath("grids")
        grid_dir.mkdir(parents=True, exist_ok=True)

        # print banner about grids
        print("=" * 80)
        print("Generating grid plots across selected biopsies")

        # Grid figures across selected biopsies for GP profiles
        print(f"    [plots] GP profile grid for {patient_bx_list}")
        print(f"    [plots] Grid number is {len(patient_bx_list)}")
        
        GPR_production_plots.plot_gp_profiles_grid(
            gp_results=results,
            patient_bx_list=patient_bx_list,
            save_path=grid_dir / f"gp_profiles_grid_kernel_{BASE_KERNEL_LABEL}",
            ncols=grid_ncols,
            label_map=grid_label_map,
            metrics_df=metrics_df,
            save_formats=("pdf", "svg"),
            dpi=400,
            include_kernel_legend=include_kernel_legend_in_primary_histograms,
            kernel_legend_label=BASE_KERNEL_LABEL,
        )

        # Grid figures across selected biopsies for semivariogram overlays
        print(f"    [plots] Semivariogram with fits grid for {patient_bx_list}")
        print(f"    [plots] Grid number is {len(patient_bx_list)}")
        GPR_production_plots.plot_variogram_overlays_grid(
            semivariogram_df=semivariogram_df,
            gp_results=results,
            patient_bx_list=patient_bx_list,
            save_path=grid_dir / f"variogram_overlays_grid_kernel_{BASE_KERNEL_LABEL}",
            ncols=grid_ncols,
            label_map=grid_label_map,
            metrics_df=metrics_df,
            save_formats=("pdf", "svg"),
            dpi=400,
        )

        # print a small banner
        print("=" * 80)
        print("Generating individual patient/biopsy-level plots and pairs")
        print("=" * 80)
        print("\n")

        for patient_id, bx_index in semivariogram_df.groupby(['Patient ID','Bx index']).groups.keys():
            # print a small banner
            print("=" * 80)
            print(f"Generating patient-level plots for Patient ID: {patient_id}, Bx index: {bx_index}")

            patient_dir = pt_sp_figures_dir.joinpath(patient_id)
            os.makedirs(patient_dir, exist_ok=True)

            res = results[(patient_id, bx_index)]

            # Production-grade patient-level figures
            print(f"    [plots] GP profile/residuals/diagnostics for Patient {patient_id}, Bx {bx_index}")
            GPR_production_plots.make_patient_level_gpr_plots(
               all_voxel_wise_dose_df,
               semivariogram_df,
               patient_id,
               bx_index,
               res,
               save_dir=patient_dir,
               save_formats=("pdf", "svg"),
               show_titles=False,
               font_scale=1.0,
               title_label=per_biopsy_label_map.get((patient_id, bx_index)),
               kernel_suffix=BASE_KERNEL_LABEL,
               include_kernel_legend=include_kernel_legend_in_primary_histograms,
               kernel_legend_label=BASE_KERNEL_LABEL,
           )

            # Paired figures with aligned axes (semivariogram+profile, reduction+ratio)
            pair_dir = patient_dir.joinpath("paired_panels")
            os.makedirs(pair_dir, exist_ok=True)
            print(f"    [plots] Paired semivariogram+profile for Patient {patient_id}, Bx {bx_index}")
            GPR_production_plots.plot_variogram_and_profile_pair(
                semivariogram_df,
                patient_id,
                bx_index,
                res,
                save_dir=pair_dir,
                file_name_base=f"variogram_profile_pair_patient_{patient_id}_bx_{bx_index}_kernel_{BASE_KERNEL_LABEL}",
                save_formats=("pdf", "svg"),
                title_label=per_biopsy_label_map.get((patient_id, bx_index)),
                metrics_row=metrics_df[(metrics_df["Patient ID"] == patient_id) & (metrics_df["Bx index"] == bx_index)].iloc[0] if not metrics_df[(metrics_df["Patient ID"] == patient_id) & (metrics_df["Bx index"] == bx_index)].empty else None,
                include_kernel_legend=include_kernel_legend_in_primary_histograms,
                kernel_legend_label=BASE_KERNEL_LABEL,
            )
            print(f"    [plots] Paired uncertainty reduction/ratio for Patient {patient_id}, Bx {bx_index}")
            GPR_production_plots.plot_uncertainty_pair(
                res,
                patient_id,
                bx_index,
                save_dir=pair_dir,
                file_name_base=f"uncertainty_pair_patient_{patient_id}_bx_{bx_index}_kernel_{BASE_KERNEL_LABEL}",
                save_formats=("pdf", "svg"),
                title_label=per_biopsy_label_map.get((patient_id, bx_index)),
                metrics_row=metrics_df[(metrics_df["Patient ID"] == patient_id) & (metrics_df["Bx index"] == bx_index)].iloc[0] if not metrics_df[(metrics_df["Patient ID"] == patient_id) & (metrics_df["Bx index"] == bx_index)].empty else None,
                include_kernel_legend=include_kernel_legend_in_primary_histograms,
                kernel_legend_label=BASE_KERNEL_LABEL,
            )
            print(f"    [plots] Paired standardized residuals for Patient {patient_id}, Bx {bx_index}")
            GPR_production_plots.plot_residuals_pair(
                res,
                patient_id,
                bx_index,
                save_dir=pair_dir,
                file_name_base=f"residuals_pair_patient_{patient_id}_bx_{bx_index}_kernel_{BASE_KERNEL_LABEL}",
                save_formats=("pdf", "svg"),
                title_label=per_biopsy_label_map.get((patient_id, bx_index)),
                include_kernel_legend=include_kernel_legend_in_primary_histograms,
                kernel_legend_label=BASE_KERNEL_LABEL,
            )
            # give the list of plots produced in this print statement
            print(f"Saved all plots for Patient ID: {patient_id}, Bx index: {bx_index} to {patient_dir}")
            
            # print a small banner
            print("=" * 80)
    else:
        _print_section("PLOTS: Patient-level figures (skipped)")
        





    # ---------------------------------------    
    # # cohort-level production plots
    # ---------------------------------------


    if run_cohort_plots:
        _print_section("PLOTS: Cohort-level figures")
        # Cohort plots

        GPR_production_plots.cohort_plots_production(
            metrics_df,
            cohort_output_figures_dir,
            save_formats=("pdf","svg"),
            kernel_suffix=BASE_KERNEL_LABEL,
            include_kernel_legend=include_kernel_legend_in_primary_histograms,
        )


        # linear regression of paired SDs between methods
        stats_path = output_dir.joinpath("cohort_mean_sd_regression_stats.csv")
        reg_stats = GPR_analysis_pipeline_functions.fit_mean_sd_regressions(metrics_df, save_csv_path=stats_path)

        GPR_production_plots.plot_mean_sd_scatter_with_fits_production(
            metrics_df, reg_stats,
            save_dir=cohort_output_figures_dir,
            file_name_base=f"cohort_mean_sd_scatter_with_fits_kernel_{BASE_KERNEL_LABEL}",
            save_formats=("pdf","svg"),
            include_kernel_legend=include_kernel_legend_in_primary_histograms,
            kernel_legend_label=BASE_KERNEL_LABEL,
        )

        GPR_production_plots.plot_mean_sd_bland_altman_production(
            metrics_df,
            save_dir=cohort_output_figures_dir,
            file_name_base=f"cohort_mean_sd_bland_altman_kernel_{BASE_KERNEL_LABEL}",
            save_formats=("pdf","svg"),
            source_csv_path=output_dir.joinpath("gpr_per_biopsy_metrics.csv"),
            show_annotation=False,
            include_kernel_legend=include_kernel_legend_in_primary_histograms,
            kernel_legend_label=BASE_KERNEL_LABEL,
        )
    else:
        _print_section("PLOTS: Cohort-level figures (skipped)")






    _print_section("GPR PIPELINE COMPLETE")
    print('\n')
    # print programme complete and pause and wait for enter to exit
    input("GPR analysis pipeline complete. Press Enter to exit.")

if __name__ == "__main__":
    main()
    # Run the main function
