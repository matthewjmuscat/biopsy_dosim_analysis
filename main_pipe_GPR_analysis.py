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
import GPR_analysis_plotting_functions_manual_methods


def main():
    main_output_path = Path("/home/matthew-muscat/Documents/UBC/Research/Data/Output data/MC_sim_out- Date-May-15-2025 Time-18,11,24")


    
    ### Load Dataframes 

    # Set csv directory
    csv_directory = main_output_path.joinpath("Output CSVs")
    cohort_csvs_directory = csv_directory.joinpath("Cohort")
    
    
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
    print(f"First 5 rows of all voxel wise dose dataframe: {all_voxel_wise_dose_df.head()}")
    # Print the last 5 rows of the dataframe
    print(f"Last 5 rows of all voxel wise dose dataframe: {all_voxel_wise_dose_df.tail()}")
    # voxel wise dose output by MC trial number (END)


    ### Load all individual bx csvs and concatenate ### (END)





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





    # ---------------------------------------    # Semivariogram analysis
    # ---------------------------------------

    semivariogram_df = GPR_analysis_helpers.semivariogram_by_biopsy(all_voxel_wise_dose_df, voxel_size_mm=1.0, max_lag_voxels=None)
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





    for patient_id, bx_index in semivariogram_df.groupby(['Patient ID', 'Bx index']).groups.keys():
        print(f"Plotting semivariogram for Patient ID: {patient_id}, Bx index: {bx_index}")

        patient_dir = pt_sp_figures_dir.joinpath(patient_id)
        os.makedirs(patient_dir, exist_ok=True)
        

        # Plot the semivariogram for each biopsy
        GPR_analysis_helpers.plot_variogram_from_df(
            semivariogram_df,
            patient_id,
            bx_index,
            overlay_df = None,  # optional precomputed overlay with columns ['h_mm', 'median_absdiff', 'mean_absdiff'] (any subset ok)
            include_title_meta = True,
            save_path = patient_dir,     # directory or full file path
            file_name =f"semivariogram_patient_{patient_id}_bx_{bx_index}.png",    # if dir provided, use this file name (ext optional)
            return_path = False,               # return saved path for downstream use
        )
        print(f"Saved semivariogram plot for Patient ID: {patient_id}, Bx index: {bx_index} to {patient_dir}")
    
    print('test')


    results = {}
    for (pid, bx_idx), _ in all_voxel_wise_dose_df.groupby(["Patient ID","Bx index"]):
        res = GPR_analysis_pipeline_functions.run_gp_for_biopsy(
            all_voxel_wise_dose_df,
            semivariogram_df,
            patient_id=pid,
            bx_index=bx_idx,
            target_stat="median",   # or "mean"
            nu=1.5                  # try 1.5 and 2.5 in sensitivity
        )
        results[(pid, bx_idx)] = res
        print(f"Processed Patient ID: {pid}, Bx index: {bx_idx}")

    
    # Build a metrics dataframe
    rows = []
    for (pid, bx_idx), res in results.items():
        row = GPR_analysis_pipeline_functions.compute_per_biopsy_metrics(pid, bx_idx, res, semivariogram_df)
        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    print("Per-biopsy metrics (head):")
    print(metrics_df.head())

    # Save for reproducibility
    metrics_csv_path = output_dir.joinpath("cohort_per_biopsy_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Saved per-biopsy metrics to: {metrics_csv_path}")

    cohort_summary = {
        "n_biopsies": int(len(metrics_df)),
        "mean_uncertainty_ratio": float(metrics_df["mean_ratio"].mean()),
        "median_uncertainty_ratio": float(metrics_df["median_ratio"].median()),
        "mean_integrated_ratio": float(metrics_df["integ_ratio"].mean()),
        "pct_biopsies_ge20pct_reduction": float((metrics_df["pct_vox_ge_20"] > 50).mean() * 100.0),  # >50% of voxels get ≥20% reduction
        "pct_reduction_mean_sd_mean":  float(metrics_df["pct_reduction_mean_sd"].mean()),
        "pct_reduction_mean_sd_std":   float(metrics_df["pct_reduction_mean_sd"].std(ddof=1)),
        "pct_reduction_mean_sd_median":float(metrics_df["pct_reduction_mean_sd"].median()),
        "pct_reduction_mean_sd_iqr":   float(metrics_df["pct_reduction_mean_sd"].quantile(0.75)
                                            - metrics_df["pct_reduction_mean_sd"].quantile(0.25)),
        "pct_reduction_integ_sd_mean": float(metrics_df["pct_reduction_integ_sd"].mean()),
        "pct_reduction_integ_sd_std":  float(metrics_df["pct_reduction_integ_sd"].std(ddof=1)),
        "pct_reduction_integ_sd_median":float(metrics_df["pct_reduction_integ_sd"].median()),
        "pct_reduction_integ_sd_iqr":  float(metrics_df["pct_reduction_integ_sd"].quantile(0.75)
                                            - metrics_df["pct_reduction_integ_sd"].quantile(0.25)),
        "median_length_scale_mm": float(metrics_df["ell"].median()),
        "median_nugget": float(metrics_df["nugget"].median()),
        "median_sv_rmse": float(metrics_df["sv_rmse"].median()),
    }
    print("Cohort summary:", cohort_summary)
    pd.Series(cohort_summary).to_csv(output_dir.joinpath("cohort_summary_numbers.csv"))


    by_patient = (
        metrics_df
        .groupby("Patient ID")
        .agg(n_bx=("Bx index", "nunique"),
            mean_ratio_mean=("mean_ratio","mean"),
            mean_ratio_sd=("mean_ratio","std"),
            ell_median=("ell","median"))
        .reset_index()
    )
    by_patient.to_csv(output_dir.joinpath("patient_level_rollups.csv"), index=False)


    for patient_id, bx_index in semivariogram_df.groupby(['Patient ID','Bx index']).groups.keys():
        patient_dir = pt_sp_figures_dir.joinpath(patient_id)
        os.makedirs(patient_dir, exist_ok=True)

        res = results[(patient_id, bx_index)]

        # 1) GP profile
        GPR_analysis_plotting_functions_manual_methods.plot_gp_profile(all_voxel_wise_dose_df, semivariogram_df, patient_id, bx_index,
                        save_path=patient_dir,
                        file_name=f"gp_profile_patient_{patient_id}_bx_{bx_index}.svg", gp_res=res, ci_level="both")

        # 2) Noise profile
        GPR_analysis_plotting_functions_manual_methods.plot_noise_profile(all_voxel_wise_dose_df, patient_id, bx_index,
                        save_path=patient_dir,
                        file_name=f"noise_profile_patient_{patient_id}_bx_{bx_index}.svg")

        # 3) Uncertainty reduction
        GPR_analysis_plotting_functions_manual_methods.plot_uncertainty_reduction(all_voxel_wise_dose_df, semivariogram_df, patient_id, bx_index,
                                save_path=patient_dir,
                                file_name=f"uncertainty_reduction_patient_{patient_id}_bx_{bx_index}.svg", gp_res=res)
        
        GPR_analysis_plotting_functions_manual_methods.plot_uncertainty_ratio(
            all_voxel_wise_dose_df, semivariogram_df, patient_id, bx_index,
            save_path=patient_dir,
            file_name=f"uncertainty_ratio_patient_{patient_id}_bx_{bx_index}.svg",
            gp_res=res
        )

        # 4) Residuals
        GPR_analysis_plotting_functions_manual_methods.plot_residuals(all_voxel_wise_dose_df, semivariogram_df, patient_id, bx_index,
                    save_path=patient_dir,
                    file_name=f"residuals_patient_{patient_id}_bx_{bx_index}.svg", gp_res=res)

        # 5) Variogram overlay (need hyperparams; reuse from a run or call _predict_at_X)
        #out = GPR_analysis_plotting_functions_manual_methods.predict_at_X(all_voxel_wise_dose_df, semivariogram_df, patient_id, bx_index)
        GPR_analysis_plotting_functions_manual_methods.plot_variogram_overlay(semivariogram_df, patient_id, bx_index, res["hyperparams"],
                            save_path=patient_dir,
                            file_name=f"variogram_overlay_patient_{patient_id}_bx_{bx_index}.svg")
        
        print(f"Saved all plots for Patient ID: {patient_id}, Bx index: {bx_index} to {patient_dir}")

    print('test')



    # Cohort plots

    GPR_analysis_plotting_functions_manual_methods.cohort_plots(metrics_df,
                 cohort_output_figures_dir)

    print('test')

    # linear regression of paired SDs between methods
    stats_path = output_dir.joinpath("cohort_mean_sd_regression_stats.csv")
    reg_stats = GPR_analysis_pipeline_functions.fit_mean_sd_regressions(metrics_df, save_csv_path=stats_path)

    plot_path = cohort_output_figures_dir.joinpath("cohort_mean_sd_scatter_with_fits.svg")
    GPR_analysis_plotting_functions_manual_methods.plot_mean_sd_scatter_with_fits(metrics_df, reg_stats, save_svg_path=plot_path)


    print('test')


if __name__ == "__main__":
    main()
    # Run the main function