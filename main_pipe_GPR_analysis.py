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
import numpy as np 
import GPR_analysis_pipeline_functions
import GPR_production_plots
import GPR_kernel_sensitivity


def main():
    def _print_section(title: str) -> None:
        line = "=" * 80
        print(f"\n{line}\n{title}\n{line}")


    # filter by simulated types
    simulated_types = ['Real']  # options: 'Real', 'Centroid DIL' 'Optimal DIL'

    # plotting / analysis gates (speed control)
    run_semivariogram_plots = False
    run_patient_plots = True
    run_kernel_sensitivity_flag = True
    run_cohort_plots = True

    # Baseline kernel selection (change here to switch kernels globally)
    #   ("matern", 1.5) -> Matérn ν = 3/2 (default)
    #   ("matern", 2.5) -> Matérn ν = 5/2
    #   ("rbf", None)   -> RBF / squared-exponential
    #   ("exp", None)   -> Exponential (approximately Matérn ν = 0.5)
    BASE_KERNEL_SPEC = ("matern", 1.5)
    _KERNEL_LABEL_MAP = {
        ("matern", 1.5): "matern_nu_1_5",
        ("matern", 2.5): "matern_nu_2_5",
        ("rbf", None): "rbf",
        ("exp", None): "exp",
    }
    if BASE_KERNEL_SPEC not in _KERNEL_LABEL_MAP:
        raise ValueError(f"Unsupported BASE_KERNEL_SPEC {BASE_KERNEL_SPEC}. Update _KERNEL_LABEL_MAP to include it.")
    BASE_KERNEL_LABEL = _KERNEL_LABEL_MAP[BASE_KERNEL_SPEC]



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
    main_output_path = Path("/home/matthew-muscat/Documents/UBC/Research/Data/Output data/MC_sim_out- Date-Jan-04-2026 Time-11,55,49")

    
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




    # ---------------------------------------    # Create output directories

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











    _print_section("QC: Voxelwise stats cross-check")
    # ---------------------------------------    # Cross-check voxelwise stats vs cohort summary
    # ---------------------------------------
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






    _print_section("SEMIVARIOGRAM: Compute per-biopsy")
    # ---------------------------------------    # Semivariogram analysis
    # ---------------------------------------

    semivariogram_df = GPR_analysis_helpers.semivariogram_by_biopsy(
        all_voxel_wise_dose_df,
        voxel_size_mm=1.0,
        max_lag_voxels=None,
    )
    """NOTE: The columns of the dataframe are:
    Index(['lag_voxels', 'h_mm', 'semivariance', 'n_pairs', 'Patient ID',
       'Bx index', 'Simulated bool', 'Simulated type', 'Bx refnum', 'Bx ID',
       'n_trials', 'n_voxels'],
      dtype='object')"""
    print(semivariogram_df)

    # Sanity check:
    # RMS difference from semivariogram
    #semivariogram_df['rms_diff'] = np.sqrt(2 * semivariogram_df['semivariance'])
    #print(semivariogram_df)
    #print('test')
    # Note that these differ a bit because the distributions are non gaussian





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
    


    _print_section("GP: Run per-biopsy + metrics")
    # Run per-biopsy Matérn GP on the filtered cohort, derive per-biopsy metrics,
    # and write cohort-level summary/rollup CSVs (metrics, summary numbers, patient rollups).
    position_mode = "begin"  # use voxel begin positions for plotting/GP (options: "center", "begin")
    nu_arg = BASE_KERNEL_SPEC[1] if BASE_KERNEL_SPEC[0] == "matern" else None
    results, metrics_df, cohort_summary_df, by_patient = GPR_analysis_helpers.run_gp_and_collect_metrics(
        all_voxel_wise_dose_df=all_voxel_wise_dose_df,
        semivariogram_df=semivariogram_df,
        output_dir=output_dir,
        target_stat="median",  # or "mean"
        nu=nu_arg,
        kernel_spec=BASE_KERNEL_SPEC,
        kernel_label=BASE_KERNEL_LABEL,
        position_mode=position_mode,
    )

    # Optional kernel sensitivity block (kept minimal to avoid clutter)
    if run_kernel_sensitivity_flag:
        _print_section("KERNEL SENSITIVITY (optional)")
        kernel_sens_dir = output_dir.joinpath("kernel_sensitivity")
        kernel_sens_dir.mkdir(parents=True, exist_ok=True)
        GPR_kernel_sensitivity.run_kernel_sensitivity(
            all_voxel_wise_dose_df=all_voxel_wise_dose_df,
            semivariogram_df=semivariogram_df,
            output_dir=kernel_sens_dir,
            target_stat="median",
            position_mode=position_mode,
        )


    if run_patient_plots:
        _print_section("PLOTS: Patient-level figures")
        for patient_id, bx_index in semivariogram_df.groupby(['Patient ID','Bx index']).groups.keys():
            patient_dir = pt_sp_figures_dir.joinpath(patient_id)
            os.makedirs(patient_dir, exist_ok=True)

            res = results[(patient_id, bx_index)]

            # Production-grade patient-level figures
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
            )

            # Paired figures with aligned axes (semivariogram+profile, reduction+ratio)
            pair_dir = patient_dir.joinpath("paired_panels")
            os.makedirs(pair_dir, exist_ok=True)
            GPR_production_plots.plot_variogram_and_profile_pair(
                semivariogram_df,
                patient_id,
                bx_index,
                res,
                save_dir=pair_dir,
                file_name_base=f"variogram_profile_pair_patient_{patient_id}_bx_{bx_index}",
                save_formats=("pdf", "svg"),
            )
            GPR_production_plots.plot_uncertainty_pair(
                res,
                patient_id,
                bx_index,
                save_dir=pair_dir,
                file_name_base=f"uncertainty_pair_patient_{patient_id}_bx_{bx_index}",
                save_formats=("pdf", "svg"),
            )
            # give the list of plots produced in this print statement
            print(f"Saved all plots for Patient ID: {patient_id}, Bx index: {bx_index} to {patient_dir}")




    if run_cohort_plots:
        _print_section("PLOTS: Cohort-level figures")
        # Cohort plots

        GPR_production_plots.cohort_plots_production(metrics_df,
                     cohort_output_figures_dir, save_formats=("pdf","svg"))


        # linear regression of paired SDs between methods
        stats_path = output_dir.joinpath("cohort_mean_sd_regression_stats.csv")
        reg_stats = GPR_analysis_pipeline_functions.fit_mean_sd_regressions(metrics_df, save_csv_path=stats_path)

        GPR_production_plots.plot_mean_sd_scatter_with_fits_production(
            metrics_df, reg_stats,
            save_dir=cohort_output_figures_dir,
            file_name_base="cohort_mean_sd_scatter_with_fits",
            save_formats=("pdf","svg"),
        )

        GPR_production_plots.plot_mean_sd_bland_altman_production(
            metrics_df,
            save_dir=cohort_output_figures_dir,
            file_name_base="cohort_mean_sd_bland_altman",
            save_formats=("pdf","svg"),
            source_csv_path=output_dir.joinpath("gpr_per_biopsy_metrics.csv"),
            show_annotation=False,
        )


    # print programme complete and pause and wait for enter to exit
    input("GPR analysis pipeline complete. Press Enter to exit.")

if __name__ == "__main__":
    main()
    # Run the main function
