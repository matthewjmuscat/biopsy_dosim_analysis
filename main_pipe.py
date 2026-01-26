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
import numpy as np
import deltas_bias_analysis

def main():

    # this way to do is has been changed
    # ------------------------------------------------------------------
    # Default simulation filters for the whole script
    # ------------------------------------------------------------------
    # These are just defaults. You can override them per-call.
    #sim_type_filter_default = "Real"   # or None, "Simulated", ["Real", "Simulated"], ...
    #sim_bool_filter_default = None     # or True / False / None



    # This one is 10k containment and 10 (very low) dosim for speed, all vitesse patients, also 2.5^3 for dil, 2.5 only for OARs and bxs
    #main_output_path = Path("/home/matthew-muscat/Documents/UBC/Research/Data/Output data/MC_sim_out- Date-Apr-03-2025 Time-15,59,46")
    
    # This one is 10 (very low for speed) containment and 10k dosim, all vitesse patients, also 2.5^3 for dil, 2.5 only for OARs and bxs (not including variation in contouring - although this is negligible)
    #main_output_path = Path("/home/matthew-muscat/Documents/UBC/Research/Data/Output data/MC_sim_out- Date-May-15-2025 Time-18,11,24")
    

    main_output_path = Path("/home/matthew-muscat/Documents/UBC/Research/Data/Output data/MC_sim_out- Date-Jan-04-2026 Time-11,55,49")


    # This one was 10k for containment and distances, so i am pulling the distances dataframe from here, the same sigmas were used so its fine, also run on a larger dataset F1 +F2 and simulated bools True for both centroid and optimal but will only need non sim F2 biopsies 
    #output_for_high_distances_run_path = Path("/home/matthew-muscat/Documents/UBC/Research/Data/Output data/MC_sim_out- Date-Dec-31-2025 Time-19,28,20")

    # This one was 10k for containment and distances, so i am pulling the distances dataframe from here, the same sigmas were used so its fine, also run on a larger dataset F1 +F2 and simulated bools True for both centroid and optimal but will only need non sim F2 biopsies
    # after fixed sampling bug, however this also changed voxelization a bit so now may not match older dosim runs. may need to rerun dosim with this one later if significant changes seen
    #output_for_high_distances_run_path = Path("/home/matthew-muscat/Documents/UBC/Research/Data/Output data/MC_sim_out- Date-Jan-03-2026 Time-05,38,44")

    # since did a rerun with everything consolidated to one output folder, can just use the main output path for distances too
    output_for_high_distances_run_path = main_output_path


    ### Load master dicts results
    
    master_structure_info_dict_results = load_files.load_master_dict(main_output_path,
                                                        "master_structure_info_dict_results")
    

    ### Define mimic structs ref dict

    bx_ref = "Bx ref"



    ### Load Dataframes 

    # Set csv directory
    csv_directory = main_output_path.joinpath("Output CSVs")
    cohort_csvs_directory = csv_directory.joinpath("Cohort")

    csv_directory_for_distances = output_for_high_distances_run_path.joinpath("Output CSVs")
    cohort_csvs_directory_for_distances = csv_directory_for_distances.joinpath("Cohort")

    # Cohort 3d radiomic features all oar and dil structures
    cohort_3d_radiomic_features_all_oar_dil_path = cohort_csvs_directory.joinpath("Cohort: 3D radiomic features all OAR and DIL structures.csv")  # Ensure the directory is a Path object
    cohort_3d_radiomic_features_all_oar_dil_df = load_files.load_csv_as_dataframe(cohort_3d_radiomic_features_all_oar_dil_path)  # Load the CSV file into a DataFrame
    """ NOTE: The columns of the dataframe are:
    cohort_3d_radiomic_features_all_oar_dil_df.columns =
    Index(['Patient ID', 'Structure ID', 'Structure type', 'Structure refnum',
        'Volume', 'Surface area', 'Surface area to volume ratio', 'Sphericity',
        'Compactness 1', 'Compactness 2', 'Spherical disproportion',
        'Maximum 3D diameter', 'PCA major', 'PCA minor', 'PCA least',
        'PCA eigenvector major', 'PCA eigenvector minor',
        'PCA eigenvector least', 'Major axis (equivalent ellipse)',
        'Minor axis (equivalent ellipse)', 'Least axis (equivalent ellipse)',
        'Elongation', 'Flatness', 'L/R dimension at centroid',
        'A/P dimension at centroid', 'S/I dimension at centroid',
        'S/I arclength', 'DIL centroid (X, prostate frame)',
        'DIL centroid (Y, prostate frame)', 'DIL centroid (Z, prostate frame)',
        'DIL centroid distance (prostate frame)', 'DIL prostate sextant (LR)',
        'DIL prostate sextant (AP)', 'DIL prostate sextant (SI)'],
      dtype='object')
    """


    # biopsy basic spatial features
    cohort_biopsy_basic_spatial_features_path = cohort_csvs_directory.joinpath("Cohort: Biopsy basic spatial features dataframe.csv")  # Ensure the directory is a Path object
    cohort_biopsy_basic_spatial_features_df = load_files.load_csv_as_dataframe(cohort_biopsy_basic_spatial_features_path)  # Load the CSV file into a DataFrame
    """ NOTE: The columns of the dataframe are:
    cohort_biopsy_basic_spatial_features_df.columns =
    Index(['Patient ID', 'Bx ID', 'Simulated bool', 'Simulated type',
        'Struct type', 'Bx refnum', 'Bx index', 'Length (mm)', 'Volume (mm3)',
        'Voxel side length (mm)', 'Relative DIL ID', 'Relative DIL index',
        'BX to DIL centroid (X)', 'BX to DIL centroid (Y)',
        'BX to DIL centroid (Z)', 'BX to DIL centroid distance',
        'NN surface-surface distance', 'Relative prostate ID',
        'Relative prostate index', 'Bx position in prostate LR',
        'Bx position in prostate AP', 'Bx position in prostate SI'],
      dtype='object')
      """

    # cohort biopsy-level distances statistics to all other structures 
    cohort_biopsy_level_distances_statistics_path = cohort_csvs_directory_for_distances.joinpath("Cohort: Tissue class - distances global results.csv")  # Ensure the directory is a Path object
    cohort_biopsy_level_distances_statistics_df = load_files.load_multiindex_csv(cohort_biopsy_level_distances_statistics_path, header_rows=[0, 1])  # Load the CSV file into a DataFrame
    """ NOTE: The columns of the dataframe are:
    print(cohort_biopsy_level_distances_statistics_df.columns)
MultiIndex([(                   'Patient ID',      ''),
            (                        'Bx ID',      ''),
            (                     'Bx index',      ''),
            (       'Relative structure ROI',      ''),
            (      'Relative structure type',      ''),
            (     'Relative structure index',      ''),
            (    'Struct. boundary NN dist.', 'count'),
            (    'Struct. boundary NN dist.',  'mean'),
            (    'Struct. boundary NN dist.',   'std'),
            (    'Struct. boundary NN dist.',   'min'),
            (    'Struct. boundary NN dist.',    '5%'),
            (    'Struct. boundary NN dist.',   '25%'),
            (    'Struct. boundary NN dist.',   '50%'),
            (    'Struct. boundary NN dist.',   '75%'),
            (    'Struct. boundary NN dist.',   '95%'),
            (    'Struct. boundary NN dist.',   'max'),
            (  'Dist. from struct. centroid', 'count'),
            (  'Dist. from struct. centroid',  'mean'),
            (  'Dist. from struct. centroid',   'std'),
            (  'Dist. from struct. centroid',   'min'),
            (  'Dist. from struct. centroid',    '5%'),
            (  'Dist. from struct. centroid',   '25%'),
            (  'Dist. from struct. centroid',   '50%'),
            (  'Dist. from struct. centroid',   '75%'),
            (  'Dist. from struct. centroid',   '95%'),
            (  'Dist. from struct. centroid',   'max'),
            ('Dist. from struct. centroid X', 'count'),
            ('Dist. from struct. centroid X',  'mean'),
            ('Dist. from struct. centroid X',   'std'),
            ('Dist. from struct. centroid X',   'min'),
            ('Dist. from struct. centroid X',    '5%'),
            ('Dist. from struct. centroid X',   '25%'),
            ('Dist. from struct. centroid X',   '50%'),
            ('Dist. from struct. centroid X',   '75%'),
            ('Dist. from struct. centroid X',   '95%'),
            ('Dist. from struct. centroid X',   'max'),
            ('Dist. from struct. centroid Y', 'count'),
            ('Dist. from struct. centroid Y',  'mean'),
            ('Dist. from struct. centroid Y',   'std'),
            ('Dist. from struct. centroid Y',   'min'),
            ('Dist. from struct. centroid Y',    '5%'),
            ('Dist. from struct. centroid Y',   '25%'),
            ('Dist. from struct. centroid Y',   '50%'),
            ('Dist. from struct. centroid Y',   '75%'),
            ('Dist. from struct. centroid Y',   '95%'),
            ('Dist. from struct. centroid Y',   'max'),
            ('Dist. from struct. centroid Z', 'count'),
            ('Dist. from struct. centroid Z',  'mean'),
            ('Dist. from struct. centroid Z',   'std'),
            ('Dist. from struct. centroid Z',   'min'),
            ('Dist. from struct. centroid Z',    '5%'),
            ('Dist. from struct. centroid Z',   '25%'),
            ('Dist. from struct. centroid Z',   '50%'),
            ('Dist. from struct. centroid Z',   '75%'),
            ('Dist. from struct. centroid Z',   '95%'),
            ('Dist. from struct. centroid Z',   'max')],
           )
      """
    # filter only the biopsies that are present in the main output (non simulated F2 biopsies)
    # get unique combinations of patients IDs and bx index from biopsy basic spatial features df 
    # keys we want to keep (from the main F2 non-sim run)
    valid_bx_keys = (
        cohort_biopsy_basic_spatial_features_df[['Patient ID', 'Bx index']]
        .drop_duplicates()
    )

    # build Index of valid (Patient ID, Bx index) pairs
    valid_index = valid_bx_keys.set_index(['Patient ID', 'Bx index']).index

    # build Index of (Patient ID, Bx index) pairs from the distances df
    dist_index = cohort_biopsy_level_distances_statistics_df.set_index(
        [('Patient ID', ''), ('Bx index', '')]
    ).index

    # boolean mask: True for rows we want to keep
    mask = dist_index.isin(valid_index)

    # filtered distances df, same MultiIndex columns as before
    cohort_biopsy_level_distances_statistics_filtered_df = (
        cohort_biopsy_level_distances_statistics_df.loc[mask].copy()
    )
    """
    print(cohort_biopsy_level_distances_statistics_filtered_df.columns)
MultiIndex([(                   'Patient ID',      ''),
            (                        'Bx ID',      ''),
            (                     'Bx index',      ''),
            (       'Relative structure ROI',      ''),
            (      'Relative structure type',      ''),
            (     'Relative structure index',      ''),
            (    'Struct. boundary NN dist.', 'count'),
            (    'Struct. boundary NN dist.',  'mean'),
            (    'Struct. boundary NN dist.',   'std'),
            (    'Struct. boundary NN dist.',   'min'),
            (    'Struct. boundary NN dist.',    '5%'),
            (    'Struct. boundary NN dist.',   '25%'),
            (    'Struct. boundary NN dist.',   '50%'),
            (    'Struct. boundary NN dist.',   '75%'),
            (    'Struct. boundary NN dist.',   '95%'),
            (    'Struct. boundary NN dist.',   'max'),
            (  'Dist. from struct. centroid', 'count'),
            (  'Dist. from struct. centroid',  'mean'),
            (  'Dist. from struct. centroid',   'std'),
            (  'Dist. from struct. centroid',   'min'),
            (  'Dist. from struct. centroid',    '5%'),
            (  'Dist. from struct. centroid',   '25%'),
            (  'Dist. from struct. centroid',   '50%'),
            (  'Dist. from struct. centroid',   '75%'),
            (  'Dist. from struct. centroid',   '95%'),
            (  'Dist. from struct. centroid',   'max'),
            ('Dist. from struct. centroid X', 'count'),
            ('Dist. from struct. centroid X',  'mean'),
            ('Dist. from struct. centroid X',   'std'),
            ('Dist. from struct. centroid X',   'min'),
            ('Dist. from struct. centroid X',    '5%'),
            ('Dist. from struct. centroid X',   '25%'),
            ('Dist. from struct. centroid X',   '50%'),
            ('Dist. from struct. centroid X',   '75%'),
            ('Dist. from struct. centroid X',   '95%'),
            ('Dist. from struct. centroid X',   'max'),
            ('Dist. from struct. centroid Y', 'count'),
            ('Dist. from struct. centroid Y',  'mean'),
            ('Dist. from struct. centroid Y',   'std'),
            ('Dist. from struct. centroid Y',   'min'),
            ('Dist. from struct. centroid Y',    '5%'),
            ('Dist. from struct. centroid Y',   '25%'),
            ('Dist. from struct. centroid Y',   '50%'),
            ('Dist. from struct. centroid Y',   '75%'),
            ('Dist. from struct. centroid Y',   '95%'),
            ('Dist. from struct. centroid Y',   'max'),
            ('Dist. from struct. centroid Z', 'count'),
            ('Dist. from struct. centroid Z',  'mean'),
            ('Dist. from struct. centroid Z',   'std'),
            ('Dist. from struct. centroid Z',   'min'),
            ('Dist. from struct. centroid Z',    '5%'),
            ('Dist. from struct. centroid Z',   '25%'),
            ('Dist. from struct. centroid Z',   '50%'),
            ('Dist. from struct. centroid Z',   '75%'),
            ('Dist. from struct. centroid Z',   '95%'),
            ('Dist. from struct. centroid Z',   'max')],
           )
    """




    # cohort biopsy-level distances statistics to all other structures
    cohort_voxel_level_distances_statistics_path = cohort_csvs_directory_for_distances.joinpath("Cohort: Tissue class - distances voxel-wise results.csv")  # Ensure the directory is a Path object
    cohort_voxel_level_distances_statistics_df = load_files.load_multiindex_csv(cohort_voxel_level_distances_statistics_path, header_rows=[0, 1])  # Load the CSV file into a DataFrame
    """ NOTE: The columns of the dataframe are:
    print(cohort_voxel_level_distances_statistics_df.columns)
    MultiIndex([(                   'Patient ID',      ''),
                (                        'Bx ID',      ''),
                (                     'Bx index',      ''),
                (       'Relative structure ROI',      ''),
                (      'Relative structure type',      ''),
                (     'Relative structure index',      ''),
                (                  'Voxel index',      ''),
                (              'Voxel begin (Z)',      ''),
                (                'Voxel end (Z)',      ''),
                (    'Struct. boundary NN dist.', 'count'),
                (    'Struct. boundary NN dist.',  'mean'),
                (    'Struct. boundary NN dist.',   'std'),
                (    'Struct. boundary NN dist.',   'min'),
                (    'Struct. boundary NN dist.',    '5%'),
                (    'Struct. boundary NN dist.',   '25%'),
                (    'Struct. boundary NN dist.',   '50%'),
                (    'Struct. boundary NN dist.',   '75%'),
                (    'Struct. boundary NN dist.',   '95%'),
                (    'Struct. boundary NN dist.',   'max'),
                (  'Dist. from struct. centroid', 'count'),
                (  'Dist. from struct. centroid',  'mean'),
                (  'Dist. from struct. centroid',   'std'),
                (  'Dist. from struct. centroid',   'min'),
                (  'Dist. from struct. centroid',    '5%'),
                (  'Dist. from struct. centroid',   '25%'),
                (  'Dist. from struct. centroid',   '50%'),
                (  'Dist. from struct. centroid',   '75%'),
                (  'Dist. from struct. centroid',   '95%'),
                (  'Dist. from struct. centroid',   'max'),
                ('Dist. from struct. centroid X', 'count'),
                ('Dist. from struct. centroid X',  'mean'),
                ('Dist. from struct. centroid X',   'std'),
                ('Dist. from struct. centroid X',   'min'),
                ('Dist. from struct. centroid X',    '5%'),
                ('Dist. from struct. centroid X',   '25%'),
                ('Dist. from struct. centroid X',   '50%'),
                ('Dist. from struct. centroid X',   '75%'),
                ('Dist. from struct. centroid X',   '95%'),
                ('Dist. from struct. centroid X',   'max'),
                ('Dist. from struct. centroid Y', 'count'),
                ('Dist. from struct. centroid Y',  'mean'),
                ('Dist. from struct. centroid Y',   'std'),
                ('Dist. from struct. centroid Y',   'min'),
                ('Dist. from struct. centroid Y',    '5%'),
                ('Dist. from struct. centroid Y',   '25%'),
                ('Dist. from struct. centroid Y',   '50%'),
                ('Dist. from struct. centroid Y',   '75%'),
                ('Dist. from struct. centroid Y',   '95%'),
                ('Dist. from struct. centroid Y',   'max'),
                ('Dist. from struct. centroid Z', 'count'),
                ('Dist. from struct. centroid Z',  'mean'),
                ('Dist. from struct. centroid Z',   'std'),
                ('Dist. from struct. centroid Z',   'min'),
                ('Dist. from struct. centroid Z',    '5%'),
                ('Dist. from struct. centroid Z',   '25%'),
                ('Dist. from struct. centroid Z',   '50%'),
                ('Dist. from struct. centroid Z',   '75%'),
                ('Dist. from struct. centroid Z',   '95%'),
                ('Dist. from struct. centroid Z',   'max')],
            )
    """
    # filter only the biopsies that are present in the main output (non simulated F2 biopsies)
    # get unique combinations of patients IDs and bx index from biopsy basic spatial features df 
    # keys we want to keep (from the main F2 non-sim run)
    valid_bx_keys = (
        cohort_biopsy_basic_spatial_features_df[['Patient ID', 'Bx index']]
        .drop_duplicates()
    )

    # build Index of valid (Patient ID, Bx index) pairs
    valid_index = valid_bx_keys.set_index(['Patient ID', 'Bx index']).index

    # build Index of (Patient ID, Bx index) pairs from the distances df
    dist_index = cohort_voxel_level_distances_statistics_df.set_index(
        [('Patient ID', ''), ('Bx index', '')]
    ).index

    # boolean mask: True for rows we want to keep
    mask = dist_index.isin(valid_index)

    # filtered distances df, same MultiIndex columns as before
    cohort_voxel_level_distances_statistics_filtered_df = (
        cohort_voxel_level_distances_statistics_df.loc[mask].copy()
    )
    """
    print(cohort_voxel_level_distances_statistics_filtered_df.columns)
    MultiIndex([(                   'Patient ID',      ''),
                (                        'Bx ID',      ''),
                (                     'Bx index',      ''),
                (       'Relative structure ROI',      ''),
                (      'Relative structure type',      ''),
                (     'Relative structure index',      ''),
                (                  'Voxel index',      ''),
                (              'Voxel begin (Z)',      ''),
                (                'Voxel end (Z)',      ''),
                (    'Struct. boundary NN dist.', 'count'),
                (    'Struct. boundary NN dist.',  'mean'),
                (    'Struct. boundary NN dist.',   'std'),
                (    'Struct. boundary NN dist.',   'min'),
                (    'Struct. boundary NN dist.',    '5%'),
                (    'Struct. boundary NN dist.',   '25%'),
                (    'Struct. boundary NN dist.',   '50%'),
                (    'Struct. boundary NN dist.',   '75%'),
                (    'Struct. boundary NN dist.',   '95%'),
                (    'Struct. boundary NN dist.',   'max'),
                (  'Dist. from struct. centroid', 'count'),
                (  'Dist. from struct. centroid',  'mean'),
                (  'Dist. from struct. centroid',   'std'),
                (  'Dist. from struct. centroid',   'min'),
                (  'Dist. from struct. centroid',    '5%'),
                (  'Dist. from struct. centroid',   '25%'),
                (  'Dist. from struct. centroid',   '50%'),
                (  'Dist. from struct. centroid',   '75%'),
                (  'Dist. from struct. centroid',   '95%'),
                (  'Dist. from struct. centroid',   'max'),
                ('Dist. from struct. centroid X', 'count'),
                ('Dist. from struct. centroid X',  'mean'),
                ('Dist. from struct. centroid X',   'std'),
                ('Dist. from struct. centroid X',   'min'),
                ('Dist. from struct. centroid X',    '5%'),
                ('Dist. from struct. centroid X',   '25%'),
                ('Dist. from struct. centroid X',   '50%'),
                ('Dist. from struct. centroid X',   '75%'),
                ('Dist. from struct. centroid X',   '95%'),
                ('Dist. from struct. centroid X',   'max'),
                ('Dist. from struct. centroid Y', 'count'),
                ('Dist. from struct. centroid Y',  'mean'),
                ('Dist. from struct. centroid Y',   'std'),
                ('Dist. from struct. centroid Y',   'min'),
                ('Dist. from struct. centroid Y',    '5%'),
                ('Dist. from struct. centroid Y',   '25%'),
                ('Dist. from struct. centroid Y',   '50%'),
                ('Dist. from struct. centroid Y',   '75%'),
                ('Dist. from struct. centroid Y',   '95%'),
                ('Dist. from struct. centroid Y',   'max'),
                ('Dist. from struct. centroid Z', 'count'),
                ('Dist. from struct. centroid Z',  'mean'),
                ('Dist. from struct. centroid Z',   'std'),
                ('Dist. from struct. centroid Z',   'min'),
                ('Dist. from struct. centroid Z',    '5%'),
                ('Dist. from struct. centroid Z',   '25%'),
                ('Dist. from struct. centroid Z',   '50%'),
                ('Dist. from struct. centroid Z',   '75%'),
                ('Dist. from struct. centroid Z',   '95%'),
                ('Dist. from struct. centroid Z',   'max')],
            )
    """




    # cohort voxel-level double sextant position within prostate for all biopsies
    cohort_voxel_level_double_sextant_positions_path = cohort_csvs_directory_for_distances.joinpath("Cohort: Per voxel prostate double sextant classification.csv")  # Ensure the directory is a Path object
    cohort_voxel_level_double_sextant_positions_df = load_files.load_csv_as_dataframe(cohort_voxel_level_double_sextant_positions_path)
    """
    print(cohort_voxel_level_double_sextant_positions_df.columns)
    Index(['Patient ID', 'Bx ID', 'Bx index', 'Voxel index', 'Simulated type',
        'Simulated bool', 'Bx refnum', 'Bx voxel prostate sextant (LR)',
        'Bx voxel prostate sextant (AP)', 'Bx voxel prostate sextant (SI)'],
        dtype='object')
    """
    cohort_voxel_level_double_sextant_positions_filtered_df = (
    cohort_voxel_level_double_sextant_positions_df.merge(
            valid_bx_keys,
            on=['Patient ID', 'Bx index'],
            how='inner'      # keep only valid pairs
        )
    )
    


    
    # Cohort global dosimetry
    cohort_global_dosimetry_path = cohort_csvs_directory.joinpath("Cohort: Global dosimetry (NEW).csv")  # Ensure the directory is a Path object
    # this is a multiindex dataframe
    cohort_global_dosimetry_df = load_files.load_multiindex_csv(cohort_global_dosimetry_path, header_rows=[0, 1])  # Load the CSV file into a DataFrame
    """ NOTE: The columns of the dataframe are:
    print(cohort_global_dosimetry_df.columns) = 
    MultiIndex([(            'Bx ID',                          ''),
            (       'Patient ID',                          ''),
            (         'Bx index',                          ''),
            (        'Bx refnum',                          ''),
            (   'Simulated bool',                          ''),
            (   'Simulated type',                          ''),
            (        'Dose (Gy)',            'argmax_density'),
            (        'Dose (Gy)',                  'kurtosis'),
            (        'Dose (Gy)',                       'max'),
            (        'Dose (Gy)',                      'mean'),
            (        'Dose (Gy)',                       'min'),
            (        'Dose (Gy)', 'nominal (spatial average)'),
            (        'Dose (Gy)',               'quantile_05'),
            (        'Dose (Gy)',               'quantile_25'),
            (        'Dose (Gy)',               'quantile_50'),
            (        'Dose (Gy)',               'quantile_75'),
            (        'Dose (Gy)',               'quantile_95'),
            (        'Dose (Gy)',                       'sem'),
            (        'Dose (Gy)',                  'skewness'),
            (        'Dose (Gy)',                       'std'),
            ('Dose grad (Gy/mm)',            'argmax_density'),
            ('Dose grad (Gy/mm)',                  'kurtosis'),
            ('Dose grad (Gy/mm)',                       'max'),
            ('Dose grad (Gy/mm)',                      'mean'),
            ('Dose grad (Gy/mm)',                       'min'),
            ('Dose grad (Gy/mm)', 'nominal (spatial average)'),
            ('Dose grad (Gy/mm)',               'quantile_05'),
            ('Dose grad (Gy/mm)',               'quantile_25'),
            ('Dose grad (Gy/mm)',               'quantile_50'),
            ('Dose grad (Gy/mm)',               'quantile_75'),
            ('Dose grad (Gy/mm)',               'quantile_95'),
            ('Dose grad (Gy/mm)',                       'sem'),
            ('Dose grad (Gy/mm)',                  'skewness'),
            ('Dose grad (Gy/mm)',                       'std')],
           )
	"""

    # Add IQR (Q75 - Q25) and IPR90 (Q95 - Q05) to the biopsy-level dataframe
    for col in ["Dose (Gy)", "Dose grad (Gy/mm)"]:
        cohort_global_dosimetry_df[(col, "IQR")] = (
            cohort_global_dosimetry_df[(col, "quantile_75")] 
            - cohort_global_dosimetry_df[(col, "quantile_25")]
        )
        cohort_global_dosimetry_df[(col, "IPR90")] = (
            cohort_global_dosimetry_df[(col, "quantile_95")] 
            - cohort_global_dosimetry_df[(col, "quantile_05")]
        )
    """ Now the columns of the dataframe are:
        print(cohort_global_dosimetry_df.columns)
        MultiIndex([(            'Bx ID',                          ''),
                    (       'Patient ID',                          ''),
                    (         'Bx index',                          ''),
                    (        'Bx refnum',                          ''),
                    (   'Simulated bool',                          ''),
                    (   'Simulated type',                          ''),
                    (        'Dose (Gy)',            'argmax_density'),
                    (        'Dose (Gy)',                  'kurtosis'),
                    (        'Dose (Gy)',                       'max'),
                    (        'Dose (Gy)',                      'mean'),
                    (        'Dose (Gy)',                       'min'),
                    (        'Dose (Gy)', 'nominal (spatial average)'),
                    (        'Dose (Gy)',               'quantile_05'),
                    (        'Dose (Gy)',               'quantile_25'),
                    (        'Dose (Gy)',               'quantile_50'),
                    (        'Dose (Gy)',               'quantile_75'),
                    (        'Dose (Gy)',               'quantile_95'),
                    (        'Dose (Gy)',                       'sem'),
                    (        'Dose (Gy)',                  'skewness'),
                    (        'Dose (Gy)',                       'std'),
                    ('Dose grad (Gy/mm)',            'argmax_density'),
                    ('Dose grad (Gy/mm)',                  'kurtosis'),
                    ('Dose grad (Gy/mm)',                       'max'),
                    ('Dose grad (Gy/mm)',                      'mean'),
                    ('Dose grad (Gy/mm)',                       'min'),
                    ('Dose grad (Gy/mm)', 'nominal (spatial average)'),
                    ('Dose grad (Gy/mm)',               'quantile_05'),
                    ('Dose grad (Gy/mm)',               'quantile_25'),
                    ('Dose grad (Gy/mm)',               'quantile_50'),
                    ('Dose grad (Gy/mm)',               'quantile_75'),
                    ('Dose grad (Gy/mm)',               'quantile_95'),
                    ('Dose grad (Gy/mm)',                       'sem'),
                    ('Dose grad (Gy/mm)',                  'skewness'),
                    ('Dose grad (Gy/mm)',                       'std'),
                    (        'Dose (Gy)',                       'IQR'),
                    (        'Dose (Gy)',                     'IPR90'),
                    ('Dose grad (Gy/mm)',                       'IQR'),
                    ('Dose grad (Gy/mm)',                     'IPR90')],
                )
    """



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

    #### WARNING THESE DVH METRICS ARE LIKELY INCORRECTLY CALCULATED IN THE MAIN ALGO!
    # Cohort bx dvh metrics
    """
    cohort_global_dosimetry_dvh_metrics_path = cohort_csvs_directory.joinpath("Cohort: Bx DVH metrics (generalized).csv")  # Ensure the directory is a Path object
    cohort_global_dosimetry_dvh_metrics_df = load_files.load_csv_as_dataframe(cohort_global_dosimetry_dvh_metrics_path)
    """
    """
    NOTE: The columns of the dataframe are:
    print(cohort_global_dosimetry_dvh_metrics_df.columns)
	Index(['Patient ID', 'Metric', 'Bx ID', 'Struct type', 
		'Simulated bool', 'Simulated type', 'Struct index', 'Mean', 'STD',
		'SEM', 'Max', 'Min', 'Skewness', 'Kurtosis', 'Q05', 'Q25', 'Q50', 'Q75',
		'Q95'],
		dtype='object')
    """
    """
    
    # ensure numeric (your df is dtype object by default)
    for col in ["Q05", "Q25", "Q50", "Q75", "Q95"]:
        cohort_global_dosimetry_dvh_metrics_df[col] = pd.to_numeric(
            cohort_global_dosimetry_dvh_metrics_df[col], errors="coerce"
        )

    # add IQR and IPR90
    cohort_global_dosimetry_dvh_metrics_df["IQR"] = (
        cohort_global_dosimetry_dvh_metrics_df["Q75"] - cohort_global_dosimetry_dvh_metrics_df["Q25"]
    )
    cohort_global_dosimetry_dvh_metrics_df["IPR90"] = (
        cohort_global_dosimetry_dvh_metrics_df["Q95"] - cohort_global_dosimetry_dvh_metrics_df["Q05"]
    )
    """
    """ Now the columns of the dataframe are:
    print(cohort_global_dosimetry_dvh_metrics_df.columns)
    Index(['Patient ID', 'Metric', 'Bx ID', 'Struct type', 
        'Simulated bool', 'Simulated type', 'Struct index', 'Mean', 'STD',
        'SEM', 'Max', 'Min', 'Skewness', 'Kurtosis', 'Q05', 'Q25', 'Q50', 'Q75',
        'Q95', 'IQR', 'IPR90'],
        dtype='object')
    """











    ### Load all individual bx csvs and concatenate ### (START)

    mc_sim_results_path = csv_directory.joinpath("MC simulation")  # Ensure the directory is a Path object

    ### 1. Point wise dose output by mc trial number 
    # Point wise dose output by MC trial number (START)
    #all_paths_point_wise_dose_output = load_files.find_csv_files(mc_sim_results_path, ['Point-wise dose output by MC trial number.csv'])
    all_paths_point_wise_dose_output = load_files.find_csv_files(mc_sim_results_path, ['Point-wise dose output by MC trial number.parquet'])
    
    # Load and concatenate 
    # Loop through all the paths and load the csv files
    all_point_wise_dose_dfs_list = []
    for path in all_paths_point_wise_dose_output:
        # Load the csv file into a dataframe
        #df = load_files.load_csv_as_dataframe(path)
        df = load_files.load_parquet_as_dataframe(path) # parquet?
        
        # Append the dataframe to the list
        all_point_wise_dose_dfs_list.append(df)

        del df
    # Concatenate all the dataframes into one dataframe
    all_point_wise_dose_df = pd.concat(all_point_wise_dose_dfs_list, ignore_index=True)
    """ NOTE: The columns of the dataframe are:
    print(all_point_wise_dose_df.columns)
	Index(['Original pt index', 'Dose (Gy)', 'Dose grad (Gy/mm)', 'MC trial num',
		'X (Bx frame)', 'Y (Bx frame)', 'Z (Bx frame)', 'R (Bx frame)',
		'Simulated bool', 'Simulated type', 'Bx refnum', 'Bx index', 'Bx ID',
		'Patient ID', 'Voxel index', 'Voxel begin (Z)', 'Voxel end (Z)'],
		dtype='object')
    """
    del all_point_wise_dose_dfs_list
    # Print the shape of the dataframe
    print(f"Shape of all point wise dose dataframe: {all_point_wise_dose_df.shape}")
    # Print the columns of the dataframe
    print(f"Columns of all point wise dose dataframe: {all_point_wise_dose_df.columns}")
    # Print the first 5 rows of the dataframe
    print(f"First 5 rows of all point wise dose dataframe: {all_point_wise_dose_df.head()}")
    # Print the last 5 rows of the dataframe
    print(f"Last 5 rows of all point wise dose dataframe: {all_point_wise_dose_df.tail()}")
    # point wise dose output by MC trial number (END)






    ### 2. Voxel wise dose output by MC trial number
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






    ### 3. all MC structure transformation values
    all_paths_mc_structure_transformation = load_files.find_csv_files(mc_sim_results_path, ['All MC structure transformation values.csv'])
    # Load and concatenate
    # Loop through all the paths and load the csv files
    all_mc_structure_transformation_dfs_list = []
    for path in all_paths_mc_structure_transformation:
        # Load the csv file into a dataframe
        df = load_files.load_csv_as_dataframe(path)
        # Append the dataframe to the list
        all_mc_structure_transformation_dfs_list.append(df)

        del df
    # Concatenate all the dataframes into one dataframe
    all_mc_structure_transformation_df = pd.concat(all_mc_structure_transformation_dfs_list, ignore_index=True)
    """ NOTE: The columns of the dataframe are:
    print(all_mc_structure_transformation_df.columns)
        Index(['Patient ID', 'Structure ID', 'Simulated bool', 'Simulated type',
            'Structure type', 'Structure ref num', 'Structure index',
            'Dilation (XY)', 'Dilation (Z)', 'Rotation (X)', 'Rotation (Y)',
            'Rotation (Z)', 'Shift (X)', 'Shift (Y)', 'Shift (Z)',
            'Shift (z_needle)', 'Trial'],
            dtype='object')
    """
    del all_mc_structure_transformation_dfs_list
    # Print the shape of the dataframe
    print(f"Shape of all MC structure transformation dataframe: {all_mc_structure_transformation_df.shape}")
    # Print the columns of the dataframe
    print(f"Columns of all MC structure transformation dataframe: {all_mc_structure_transformation_df.columns}")
    # Print the first 5 rows of the dataframe
    print(f"First 5 rows of all MC structure transformation dataframe: {all_mc_structure_transformation_df.head()}")
    # Print the last 5 rows of the dataframe
    print(f"Last 5 rows of all MC structure transformation dataframe: {all_mc_structure_transformation_df.tail()}")
    # all MC structure transformation values (END)

    
    # wrote a function to calculate these from the voxel wise dose dataframe
    if False:
        ### 4. cumulative dvh by mc trial number
        all_paths_cumulative_dvh_by_mc_trial_number = load_files.find_csv_files(mc_sim_results_path, ['Cumulative DVH by MC trial.parquet'])
        # Load and concatenate
        # Loop through all the paths and load the csv files
        all_cumulative_dvh_by_mc_trial_number_dfs_list = []
        for path in all_paths_cumulative_dvh_by_mc_trial_number:
            # Load the csv file into a dataframe
            #df = load_files.load_csv_as_dataframe(path)
            df = load_files.load_parquet_as_dataframe(path)
            # Append the dataframe to the list
            all_cumulative_dvh_by_mc_trial_number_dfs_list.append(df)

            del df
        # Concatenate all the dataframes into one dataframe
        all_cumulative_dvh_by_mc_trial_number_df = pd.concat(all_cumulative_dvh_by_mc_trial_number_dfs_list, ignore_index=True)
        """ NOTE: The columns of the dataframe are:
        print(all_cumulative_dvh_by_mc_trial_number_df.columns)
            Index(['Patient ID', 'Bx ID', 'Bx index', 'Simulated bool', 'Simulated type',
                'Percent volume', 'Dose (Gy)', 'MC trial'],
                dtype='object')
        """
        del all_cumulative_dvh_by_mc_trial_number_dfs_list
        # Print the shape of the dataframe
        print(f"Shape of all cumulative dvh by mc trial number dataframe: {all_cumulative_dvh_by_mc_trial_number_df.shape}")
        # Print the columns of the dataframe
        print(f"Columns of all cumulative dvh by mc trial number dataframe: {all_cumulative_dvh_by_mc_trial_number_df.columns}")
        # Print the first 5 rows of the dataframe
        print(f"First 5 rows of all cumulative dvh by mc trial number dataframe: {all_cumulative_dvh_by_mc_trial_number_df.head()}")
        # Print the last 5 rows of the dataframe
        print(f"Last 5 rows of all cumulative dvh by mc trial number dataframe: {all_cumulative_dvh_by_mc_trial_number_df.tail()}")
        # cumulative dvh by mc trial number (END)

    
    ### 5. Differential DVH by MC trial number
    all_paths_differential_dvh_by_mc_trial_number = load_files.find_csv_files(mc_sim_results_path, ['Differential DVH by MC trial.parquet'])
    # Load and concatenate
    # Loop through all the paths and load the csv files
    all_differential_dvh_by_mc_trial_number_dfs_list = []
    for path in all_paths_differential_dvh_by_mc_trial_number:
        # Load the csv file into a dataframe
        #df = load_files.load_csv_as_dataframe(path)
        df = load_files.load_parquet_as_dataframe(path)
        # Append the dataframe to the list
        all_differential_dvh_by_mc_trial_number_dfs_list.append(df)

        del df
    # Concatenate all the dataframes into one dataframe
    all_differential_dvh_by_mc_trial_number_df = pd.concat(all_differential_dvh_by_mc_trial_number_dfs_list, ignore_index=True)
    """ NOTE: The columns of the dataframe are:
    print(all_differential_dvh_by_mc_trial_number_df.columns)
        Index(['Patient ID', 'Bx ID', 'Bx index', 'Simulated bool', 'Simulated type',
            'Percent volume', 'Dose bin edge (left) (Gy)',
            'Dose bin edge (right) (Gy)', 'Dose bin center (Gy)',
            'Dose bin width (Gy)', 'Dose bin number', 'MC trial'],
            dtype='object')
    """
    del all_differential_dvh_by_mc_trial_number_dfs_list
    # Print the shape of the dataframe
    print(f"Shape of all differential dvh by mc trial number dataframe: {all_differential_dvh_by_mc_trial_number_df.shape}")
    # Print the columns of the dataframe
    print(f"Columns of all differential dvh by mc trial number dataframe: {all_differential_dvh_by_mc_trial_number_df.columns}")
    # Print the first 5 rows of the dataframe
    print(f"First 5 rows of all differential dvh by mc trial number dataframe: {all_differential_dvh_by_mc_trial_number_df.head()}")
    # Print the last 5 rows of the dataframe
    print(f"Last 5 rows of all differential dvh by mc trial number dataframe: {all_differential_dvh_by_mc_trial_number_df.tail()}")
    # differential dvh by mc trial number (END)





    ### Load all individual bx csvs and concatenate ### (END)






    # Load uncertainties csv
    #uncertainties_path = main_output_path.joinpath("uncertainties_file_auto_generated Date-Apr-02-2025 Time-03,45,24.csv")  # Ensure the directory is a Path object

    # Assuming main_output_path is already a Path object
    # Adjust the pattern as needed if the prefix should be "uncertainities"
    csv_files = list(main_output_path.glob("uncertainties*.csv"))
    if csv_files:
        # grab the first one 
        uncertainties_path = csv_files[0]
        uncertainties_df = load_files.load_csv_as_dataframe(uncertainties_path)
    else:
        raise FileNotFoundError("No uncertainties CSV file found in the directory.")
    



    ### Apply simulation filtering manually because the below down locals() method is casuing problems :


    ############ APPLY GLOBAL FILTERS TO ALL DATAFRAMES IF APPLICABLE ############
    # This filters dataframes based on sim_type_filter_default and sim_bool_filter_default variables set at the top of this script
    # Apply global filters to all DataFrames in this module
    ########### LOADING COMPLETE


    # ------------------------------------------------------------------
    # FILTER: keep only real cores (Simulated type == 'Real')
    # ------------------------------------------------------------------

    # ---------------------------------------------------------
    # Global simulation filtering
    #   - sim_type_filter: "Real" / "Simulated" / list / None
    #   - sim_bool_filter: True / False / None
    # ---------------------------------------------------------
    sim_type_filter = "Real"   # <- your default
    sim_bool_filter = None     # <- don't filter on Simulated bool by default

    sim_filter_targets: list[tuple[str, pd.DataFrame]] = [
        ("cohort_biopsy_basic_spatial_features_df", cohort_biopsy_basic_spatial_features_df),
        ("cohort_voxel_level_double_sextant_positions_df", cohort_voxel_level_double_sextant_positions_df),
        ("cohort_global_dosimetry_df", cohort_global_dosimetry_df),
        ("cohort_global_dosimetry_by_voxel_df", cohort_global_dosimetry_by_voxel_df),
        ("all_point_wise_dose_df", all_point_wise_dose_df),
        ("all_voxel_wise_dose_df", all_voxel_wise_dose_df),
        ("all_mc_structure_transformation_df", all_mc_structure_transformation_df),
        ("all_differential_dvh_by_mc_trial_number_df", all_differential_dvh_by_mc_trial_number_df),
        ]

    for name, df in sim_filter_targets:
        helper_funcs.filter_df_by_sim_inplace(
            df,
            name=name,
            sim_type_filter=sim_type_filter,
            sim_bool_filter=sim_bool_filter,
        )


    


    # ------------------------------------------------------------------
    # Apply simulation filtering to all locally defined DataFrames
    # ------------------------------------------------------------------
    
    """
    if sim_type_filter_default is not None or sim_bool_filter_default is not None:
        _ns = locals()
        for _name, _obj in list(_ns.items()):
            if isinstance(_obj, pd.DataFrame):
                df = _obj

                # Only touch DFs that actually have sim columns
                flat_cols = [str(c) for c in df.columns.to_flat_index()]
                has_sim_col = any(
                    ("Simulated type" in c) or ("Simulated bool" in c)
                    for c in flat_cols
                )

                if not has_sim_col:
                    # e.g. cohort_3d_radiomic_features_all_oar_dil_df
                    continue
                helper_funcs.filter_df_by_sim_inplace(
                    df,
                    name=_name,
                    sim_type_filter=sim_type_filter_default,
                    sim_bool_filter=sim_bool_filter_default,
                )



    # ------------------------------------------------------------------
    # Scan all local DataFrames for NaNs and report them
    # ------------------------------------------------------------------
    _ns = locals()
    for _name, _obj in list(_ns.items()):
        if isinstance(_obj, pd.DataFrame):
            df = _obj
            if df.isna().values.any():
                total_cells = df.shape[0] * df.shape[1]
                total_nans = int(df.isna().sum().sum())
                print(f"[NAN CHECK] {_name}: {total_nans} NaNs out of {total_cells} cells")

                # Optional: show per-column NaN counts (only those with NaNs)
                col_nans = df.isna().sum()
                col_nans = col_nans[col_nans > 0]
                if not col_nans.empty:
                    print(f"    Columns with NaNs:")
                    for col, n in col_nans.items():
                        print(f"        {col}: {n}")


    """



    ### DROP REFNUM COLUMN FROM ALL DATAFRAMES 

    for df in [cohort_biopsy_basic_spatial_features_df,
                cohort_voxel_level_double_sextant_positions_df,
                cohort_global_dosimetry_df,
                cohort_global_dosimetry_by_voxel_df,
                all_point_wise_dose_df,
                all_voxel_wise_dose_df,
               ]:
       df = helper_funcs.drop_bx_refnum(df)
    ########### LOADING COMPLETE









    ## Create output directory
    # Output directory 
    output_dir = Path(__file__).parents[0].joinpath("output_data")
    os.makedirs(output_dir, exist_ok=True)



    #### Create figures directories
    # make dirs
    output_fig_directory = output_dir.joinpath("figures")
    os.makedirs(output_fig_directory, exist_ok=True)
    cohort_output_figures_dir = output_fig_directory.joinpath("cohort_output_figures")
    os.makedirs(cohort_output_figures_dir, exist_ok=True)
    pt_sp_figures_dir = output_fig_directory.joinpath("patient_specific_output_figures")
    os.makedirs(pt_sp_figures_dir, exist_ok=True)


    ### Get unqiue patient IDs
    # Get ALL unique patient IDs from the cohort_3d_radiomic_features_all_oar_dil_df DataFrame
    unique_patient_ids_all = cohort_biopsy_basic_spatial_features_df['Patient ID'].unique().tolist()
    # Print the unique patient IDs
    print("Unique Patient IDs (ALL):")
    print(unique_patient_ids_all)
    # Print the number of unique patient IDs
    print(f"Number of unique patient IDs ALL: {len(unique_patient_ids_all)}")


    # Get unique patient IDs, however the patient IDs actually include the patient ID (F#) where F# indicates the fraction but its actually the same patient, so I want to take only F1, if F1 isnt present for a particular ID then I want to take F2
    # Get the unique patient IDs from the cohort_3d_radiomic_features_all_oar_dil_df DataFrame
    unique_patient_ids_f1_prioritized = misc_funcs.get_unique_patient_ids_fraction_prioritize(cohort_biopsy_basic_spatial_features_df,patient_id_col='Patient ID', priority_fraction='F1')
    # Print the unique patient IDs
    print("Unique Patient IDs (F1) prioritized:")
    print(unique_patient_ids_f1_prioritized)
    # Print the number of unique patient IDs
    print(f"Number of unique patient IDs (F1) prioritized: {len(unique_patient_ids_f1_prioritized)}")

    ### Get unique patient IDs for F1
    # Get the unique patient IDs from the cohort_3d_radiomic_features_all_oar_dil_df DataFrame
    unique_patient_ids_f1 = misc_funcs.get_unique_patient_ids_fraction_specific(cohort_biopsy_basic_spatial_features_df, patient_id_col='Patient ID',fraction='F1')
    # Print the unique patient IDs
    print("Unique Patient IDs (F1) ONLY:")
    print(unique_patient_ids_f1)
    # Print the number of unique patient IDs
    print(f"Number of unique patient IDs F1 ONLY: {len(unique_patient_ids_f1)}")

    ### Get unique patient IDs for F2
    # Get the unique patient IDs from the cohort_3d_radiomic_features_all_oar_dil_df DataFrame
    unique_patient_ids_f2 = misc_funcs.get_unique_patient_ids_fraction_specific(cohort_biopsy_basic_spatial_features_df, patient_id_col='Patient ID',fraction='F2')
    # Print the unique patient IDs
    print("Unique Patient IDs (F2) only:")
    print(unique_patient_ids_f2)
    # Print the number of unique patient IDs
    print(f"Number of unique patient IDs F2 ONLY: {len(unique_patient_ids_f2)}")


    ### Uncertainties analysis (START)
    # Create output directory for uncertainties analysis
    uncertainties_analysis_dir = output_dir.joinpath("uncertainties_analysis")
    os.makedirs(uncertainties_analysis_dir, exist_ok=True)
    # Output filename
    output_filename = 'uncertainties_analysis_statistics_all_patients.csv'
    # Get uncertainties analysis statistics
    uncertainties_analysis_statistics_df = uncertainties_analysis.compute_statistics_by_structure_type(uncertainties_df,
                                                                                            columns=['mu (X)', 'mu (Y)', 'mu (Z)', 'sigma (X)', 'sigma (Y)', 'sigma (Z)', 'Dilations mu (XY)', 'Dilations mu (Z)', 'Dilations sigma (XY)', 'Dilations sigma (Z)', 'Rotations mu (X)', 'Rotations mu (Y)', 'Rotations mu (Z)', 'Rotations sigma (X)', 'Rotations sigma (Y)', 'Rotations sigma (Z)'], 
                                                                                            patient_uids=unique_patient_ids_f2)
    # Save the statistics to a CSV file
    uncertainties_analysis_statistics_df.to_csv(uncertainties_analysis_dir.joinpath(output_filename), index=True)
    # Print the statistics
    print(uncertainties_analysis_statistics_df)
    ### Uncertainties analysis (END)





	### Radiomic features analysis (START)
    # Create output directory for radiomic features
    radiomic_features_dir = output_dir.joinpath("radiomic_features")
    os.makedirs(radiomic_features_dir, exist_ok=True)
    # Output filename
    output_filename = 'radiomic_features_statistics_all_patients.csv'
    # Get radiomic statistics
    radiomic_statistics_df = shape_and_radiomic_features.get_radiomic_statistics(cohort_3d_radiomic_features_all_oar_dil_df, 
                                                                                 patient_id= unique_patient_ids_f2, 
                                                                                 exclude_columns=['Patient ID', 'Structure ID', 'Structure type', 'Structure refnum','PCA eigenvector major', 'PCA eigenvector minor',	'PCA eigenvector least', 'DIL centroid (X, prostate frame)', 'DIL centroid (Y, prostate frame)', 'DIL centroid (Z, prostate frame)', 'DIL centroid distance (prostate frame)', 'DIL prostate sextant (LR)', 'DIL prostate sextant (AP)', 'DIL prostate sextant (SI)'])

    # Save the statistics to a CSV file
    radiomic_statistics_df.to_csv(radiomic_features_dir.joinpath(output_filename), index=True)
    # Print the statistics
    print(radiomic_statistics_df)
    ### Radiomic features analysis (END)





	### cumulative dil volume stats (START)
    mean_cumulative_dil_vol, std_cumulative_dil_vol = shape_and_radiomic_features.cumulative_dil_volume_stats(unique_patient_ids_f2, cohort_3d_radiomic_features_all_oar_dil_df)
    # Print the statistics
    print(f"Mean cumulative DIL volume: {mean_cumulative_dil_vol}")
    print(f"Standard deviation of cumulative DIL volume: {std_cumulative_dil_vol}")
    # Save the statistics to a CSV file
    output_filename = 'cumulative_dil_volume_statistics_all_patients.csv'
    # Create output directory for DIL information, use radiomic_features_dir
    os.makedirs(radiomic_features_dir, exist_ok=True)
    # Save the statistics to a CSV file
    cumulative_dil_volume_statistics_df = pd.DataFrame({'Mean cumulative DIL volume (mm^3)': [mean_cumulative_dil_vol], 'Standard deviation of cumulative DIL volume (mm^3)': [std_cumulative_dil_vol]})
    cumulative_dil_volume_statistics_df.to_csv(radiomic_features_dir.joinpath(output_filename), index=False)
    ### cumulatiove dil volume stats (END)
    




	### Find structure counts (START)
    # Create output directory for DIL information
    radiomic_features_dir = output_dir.joinpath("radiomic_features")
    os.makedirs(radiomic_features_dir, exist_ok=True)
    # Output filename
    output_filename = 'structure_counts_all_patients.csv'
    # Get structure counts
    structure_counts_df, structure_counts_statistics_df = shape_and_radiomic_features.calculate_structure_counts_and_stats(cohort_3d_radiomic_features_all_oar_dil_df, patient_id=unique_patient_ids_f2, structure_types=None)
    # Save the statistics to a CSV file
    structure_counts_df.to_csv(radiomic_features_dir.joinpath(output_filename), index=True)
    # Print the statistics
    print(structure_counts_df)
    # Save the statistics to a CSV file
    output_filename = 'structure_counts_statistics_all_patients.csv'
    # Save the statistics to a CSV file
    structure_counts_statistics_df.to_csv(radiomic_features_dir.joinpath(output_filename), index=True)
    # Print the statistics
    print(structure_counts_statistics_df)
    ### Find structure counts (END)





    ### Biopsy information analysis (START)
    # Create output directory for biopsy information
    biopsy_information_dir = output_dir.joinpath("biopsy_information")
    os.makedirs(biopsy_information_dir, exist_ok=True)
    # Output filename
    output_filename = 'biopsy_information_statistics_all_patients.csv'
    # Get biopsy information statistics
    biopsy_information_statistics_df = biopsy_information.get_filtered_statistics(cohort_biopsy_basic_spatial_features_df, 
                                                                                 columns=['Length (mm)', 
                                                                                          'Volume (mm3)', 
                                                                                          'Voxel side length (mm)',  
                                                                                          'BX to DIL centroid (X)', 
                                                                                          'BX to DIL centroid (Y)',
                                                                                          'BX to DIL centroid (Z)', 
                                                                                          'BX to DIL centroid distance', 
                                                                                          'NN surface-surface distance'], 
                                                                                 patient_id=unique_patient_ids_f2,
                                                                                 simulated_type='Real')

    # Save the statistics to a CSV file
    biopsy_information_statistics_df.to_csv(biopsy_information_dir.joinpath(output_filename), index=True)
    # Print the statistics
    print(biopsy_information_statistics_df)
    ### Biopsy information analysis (END)




    ### COMPUTE DVH METRICS PER TRIAL (START)
    # define the output directory for the dvh metrics
    dvh_metrics_dir = output_dir.joinpath("dvh_metrics")
    os.makedirs(dvh_metrics_dir, exist_ok=True)  # create the directory if it doesn't exist
    output_filename = "Cohort: DVH metrics per trial.csv"

    # IMPORTANT: This function is only valid if the fed dataframe only has 1 sample point per voxel index per trial, otherwise overweighting can occur.
    calculated_dvh_metrics_per_trial_df = helper_funcs.compute_dvh_metrics_per_trial_vectorized(
        all_voxel_wise_dose_df,
        d_perc_list = [2,50,98],
        v_perc_list = [100,125,150,175,200,300],
        # How to define the reference dose for V_Y% thresholds:
        # - EITHER pass a single float (same ref for all groups),
        # - OR pass the name of a column in df with the per-voxel ref dose (it must be constant within each group),
        # - OR pass a dict {(patient_id, bx_index): ref_dose_gy}.
        ref_dose_gy = 13.5,  # single float ref dose for all groups
        ref_dose_col = None,
        ref_dose_map = None,
        # I/O
        output_dir= dvh_metrics_dir,
        csv_name = output_filename
    )
    """print(calculated_dvh_metrics_per_trial_df.columns)=
    Index(['Patient ID', 'Bx index', 'MC trial num', 'Simulated bool',
       'Simulated type', 'Bx ID', 'D_2% (Gy)', 'D_50% (Gy)', 'D_98% (Gy)',
       'V_100% (%)', 'V_125% (%)', 'V_150% (%)', 'V_175% (%)', 'V_200% (%)',
       'V_300% (%)'],
      dtype='object')
    """
    print(f"DVH metrics per trial csv saved to file: {dvh_metrics_dir.joinpath(output_filename)}")
    ### COMPUTE DVH METRICS PER TRIAL (END)






    ### COMPUTE DVH METRICS STATISTICS PER BIOPSY (START)


    dvh_metrics_dir = output_dir.joinpath("dvh_metrics")
    os.makedirs(dvh_metrics_dir, exist_ok=True)  # create the directory if it doesn't exist
    output_filename = "Cohort_DVH_metrics_stats_per_biopsy.csv"

    cohort_global_dosimetry_dvh_metrics_df = helper_funcs.build_dvh_summary_one_row_per_biopsy(calculated_dvh_metrics_per_trial_df)
    """
    print(cohort_global_dosimetry_dvh_metrics_df.columns) = 
        Index(['Patient ID', 'Metric', 'Bx ID', 'Struct type',
            'Simulated bool', 'Simulated type', 'Struct index', 'Mean', 'STD',
            'SEM', 'Max', 'Min', 'Skewness', 'Kurtosis', 'Q05', 'Q25', 'Q50', 'Q75',
            'Q95', 'IQR', 'IPR90', 'Nominal'],
            dtype='object')
    """
    # Save cohort summary (DVH metrics) to file using simple to csv
    
    output_path = dvh_metrics_dir / output_filename
    cohort_global_dosimetry_dvh_metrics_df.to_csv(output_path, index=False)
    print(f'dvh summary csv saved to file: {output_path}')
    
    print('dvh summary csv saved to file')

    ### COMPUTE DVH METRICS STATISTICS PER BIOPSY (END)






     # --- NEW: Path 1 QA analysis on DVH metrics ---
    from qa_path1_thresholds import (
        ThresholdConfig,
        compute_biopsy_threshold_probabilities,
        attach_margin_z_scores_from_trials,   # NEW
        drop_influential_d2_biopsy_by_cooks
    )

    qa_dir = output_dir.joinpath("qa_path1")
    os.makedirs(qa_dir, exist_ok=True)

    qa_1_core = qa_dir.joinpath("1_core")
    os.makedirs(qa_1_core, exist_ok=True)

    qa_2_logit_margin = qa_dir.joinpath("2_logit_margin")
    os.makedirs(qa_2_logit_margin, exist_ok=True)

    qa_3_logit_grad = qa_dir.joinpath("3_logit_grad")
    os.makedirs(qa_3_logit_grad, exist_ok=True)

    qa_4_design = qa_dir.joinpath("4_design")
    os.makedirs(qa_4_design, exist_ok=True)

    qa_5_correlations = qa_dir.joinpath("5_correlations")
    os.makedirs(qa_5_correlations, exist_ok=True)

    qa_6_secondary_scan = qa_dir.joinpath("6_secondary_scan")
    os.makedirs(qa_6_secondary_scan, exist_ok=True)

    



    # Example threshold choices (we can tweak these, but the pipeline is general):
    threshold_configs = [
        # Cold-spot style: near-minimum dose per biopsy per fraction
        ThresholdConfig(
            metric_col="D_98% (Gy)",
            threshold=20.0,         # ~150% of Rx (13.5), around cohort median
            comparison="ge",
            label="D98 ≥ 20 Gy"
        ),
        # Median dose adequacy within the targeted core
        ThresholdConfig(
            metric_col="D_50% (Gy)",
            threshold=27.0,         # ~185% of Rx, in the middle of your D50 distribution
            comparison="ge",
            label="D50 ≥ 27 Gy"
        ),
        ThresholdConfig(
            metric_col="D_2% (Gy)",
            threshold=32.0,
            comparison="ge",       # pass if D2 >= 32 Gy
            label="D2 ≥ 32 Gy"
        ),

        # High-dose volume fraction inside the core
        ThresholdConfig(metric_col="V_150% (%)", threshold=50.0, comparison="ge",
                        label="V150 ≥ 50%"),
    ]

    path1_results_df = compute_biopsy_threshold_probabilities(
        calculated_dvh_metrics_per_trial_df,
        configs=threshold_configs,
        # group_cols default is ("Patient ID", "Bx index", "Bx ID")
        prob_pass_cutoffs=(0.05, 0.95),
    )


    # REMOVING ONE OUTLIER POINT FOR THE D2%, BECAUSE IT MAKES THE LOGISTIC FIT QUITE A BIT WORSE!
    if False: # remove outlier manually
        df = path1_results_df.copy()

        # find the worst outlier in the D2% panel
        is_d2 = df["metric"] == "D_2% (Gy)"
        idx_outlier = (
            df.loc[is_d2]
            .sort_values("distance_from_threshold_nominal")
            .iloc[-1]          # largest margin
            .name
        )

        # drop it
        path1_results_df = df.drop(index=idx_outlier)

        # print the droped index meta information and probability and distance from threshold margin values
        outlier_row = df.loc[idx_outlier]
        print("Dropped outlier index:", idx_outlier)
        print(outlier_row[["Patient ID", "Bx index", "Bx ID", "metric", "threshold",
                           "distance_from_threshold_nominal", "p_pass"]])

    else:  # use Cook's D to find outlier

        path1_results_df, d2_out_idx, cooks_max, cooks_thr = \
            drop_influential_d2_biopsy_by_cooks(path1_results_df)
        print("D2% Cook’s D max:", cooks_max, "threshold:", cooks_thr, "dropped index:", d2_out_idx)

    


    # Attach z = margin / STD(metric across trials)
    path1_results_df = attach_margin_z_scores_from_trials(
        path1_results_df,
        calculated_dvh_metrics_per_trial_df,
    )
    """
    print(path1_results_df.columns)
    Index(['Patient ID', 'Bx index', 'Bx ID', 'metric', 'threshold',
        'comparison', 'label', 'p_pass', 'n_pass', 'n_trials', 'qa_class',
        'nominal_value', 'nominal_pass', 'distance_from_threshold_nominal',
        'misclassified', 'misclassification_type', 'metric_std', 'z_margin'],
        dtype='object')
      """
    
    # save the enriched version
    
    path1_results_df.to_csv(
        qa_1_core.joinpath("p1_core_01_biopsy_mc_probs_z.csv"),
        index=False,
    )
    
    # --- END NEW BLOCK ---






    # --- NEW: enrich + summarize + model ---
    from qa_path1_thresholds import (
        compute_nominal_core_averages_from_voxels,
        attach_core_nominal_predictors,
        summarize_path1_by_threshold_v2,
        fit_path1_logit_per_threshold,
        compare_path1_logit_models,
        compute_margin_correlations_by_threshold,
        summarize_margin_by_categorical_predictors,

    )

    # 1) predictors (nominal core mean dose/grad) from voxel-wise df at trial 0
    core_nominal_df = compute_nominal_core_averages_from_voxels(all_voxel_wise_dose_df)

    path1_enriched_df = attach_core_nominal_predictors(path1_results_df, core_nominal_df)
    path1_enriched_df.to_csv(qa_1_core / "p1_core_02_biopsy_probs_plus_nominal_predictors.csv", index=False)
    """
    print(path1_enriched_df.columns)
    Index(['Patient ID', 'Bx index', 'Bx ID', 'metric', 'threshold',
        'comparison', 'label', 'p_pass', 'n_pass', 'n_trials', 'qa_class',
        'nominal_value', 'nominal_pass', 'distance_from_threshold_nominal',
        'misclassified', 'misclassification_type', 'metric_std', 'z_margin',
        'nominal_core_mean_dose_gy', 'nominal_core_mean_grad_gy_per_mm'],
        dtype='object')
    """

    # 2) cohort summary table (deliverable)
    summary_df = summarize_path1_by_threshold_v2(path1_enriched_df)
    summary_df.to_csv(qa_1_core / "p1_core_03_threshold_summary_by_rule.csv", index=False)
    """print(summary_df.columns)
    Index(['label', 'n_rows', 'n_conf_pass', 'n_borderline', 'n_conf_fail',
       'n_conf_only', 'n_misclassified_conf_only', 'misclass_rate_conf_only',
       'n_nominal_overestimates_conf_only',
       'n_nominal_underestimates_conf_only', 'p_pass_n', 'p_pass_mean',
       'p_pass_std', 'p_pass_min', 'p_pass_q05', 'p_pass_q25', 'p_pass_median',
       'p_pass_q75', 'p_pass_q95', 'p_pass_max', 'p_pass_iqr', 'p_pass_ipr90',
       'margin_n', 'margin_mean', 'margin_std', 'margin_min', 'margin_q05',
       'margin_q25', 'margin_median', 'margin_q75', 'margin_q95', 'margin_max',
       'margin_iqr', 'margin_ipr90', 'nominal_value_n', 'nominal_value_mean',
       'nominal_value_std', 'nominal_value_min', 'nominal_value_q05',
       'nominal_value_q25', 'nominal_value_median', 'nominal_value_q75',
       'nominal_value_q95', 'nominal_value_max', 'nominal_value_iqr',
       'nominal_value_ipr90', 'nominal_core_mean_dose_gy_n',
       'nominal_core_mean_dose_gy_mean', 'nominal_core_mean_dose_gy_std',
       'nominal_core_mean_dose_gy_min', 'nominal_core_mean_dose_gy_q05',
       'nominal_core_mean_dose_gy_q25', 'nominal_core_mean_dose_gy_median',
       'nominal_core_mean_dose_gy_q75', 'nominal_core_mean_dose_gy_q95',
       'nominal_core_mean_dose_gy_max', 'nominal_core_mean_dose_gy_iqr',
       'nominal_core_mean_dose_gy_ipr90', 'nominal_core_mean_grad_gy_per_mm_n',
       'nominal_core_mean_grad_gy_per_mm_mean',
       'nominal_core_mean_grad_gy_per_mm_std',
       'nominal_core_mean_grad_gy_per_mm_min',
       'nominal_core_mean_grad_gy_per_mm_q05',
       'nominal_core_mean_grad_gy_per_mm_q25',
       'nominal_core_mean_grad_gy_per_mm_median',
       'nominal_core_mean_grad_gy_per_mm_q75',
       'nominal_core_mean_grad_gy_per_mm_q95',
       'nominal_core_mean_grad_gy_per_mm_max',
       'nominal_core_mean_grad_gy_per_mm_iqr',
       'nominal_core_mean_grad_gy_per_mm_ipr90'],
      dtype='object')
    """

    # 3) logistic models (deliverables)
    # Primary: margin only
    coef1_df, pred1_df = fit_path1_logit_per_threshold(
        path1_enriched_df,
        predictors=("distance_from_threshold_nominal",),
    )
    coef1_df.to_csv(qa_2_logit_margin / "p1_logit_margin_01_coef.csv", index=False)
    pred1_df.to_csv(qa_2_logit_margin / "p1_logit_margin_02_predictions.csv", index=False)
    """
    print(coef1_df.columns)
    Index(['label', 'metric', 'threshold', 'predictors', 'n_biopsies',
       'n_trials_mean', 'll_eff', 'll_null_eff', 'aic', 'mcfadden_r2',
       'brier_w', 'rmse_prob_w', 'k_params', 'b_const',
       'b_distance_from_threshold_nominal', 'margin_at_p50', 'margin_at_p95'],
      dtype='object')
    """
    """
    print(pred1_df.columns)
    Index(['label', 'metric', 'threshold', 'p_pass', 'n_pass', 'n_trials',
       'distance_from_threshold_nominal', 'qa_class', 'p_hat_model'],
      dtype='object')
    """

    # Secondary: margin + core averaged nominal gradient
    coef2_df, pred2_df = fit_path1_logit_per_threshold(
        path1_enriched_df,
        predictors=("distance_from_threshold_nominal", "nominal_core_mean_grad_gy_per_mm"),
    )
    coef2_df.to_csv(qa_3_logit_grad / "p1_logit_grad_01_coef.csv", index=False)
    pred2_df.to_csv(qa_3_logit_grad / "p1_logit_grad_02_predictions.csv", index=False)
    """
    print(coef2_df.columns)
    Index(['label', 'metric', 'threshold', 'predictors', 'n_biopsies',
        'n_trials_mean', 'll_eff', 'll_null_eff', 'aic', 'mcfadden_r2',
        'brier_w', 'rmse_prob_w', 'k_params', 'b_const',
        'b_distance_from_threshold_nominal',
        'b_nominal_core_mean_grad_gy_per_mm', 'margin_at_p50_ref',
        'margin_at_p95_ref', 'nominal_core_mean_grad_gy_per_mm_ref_median'],
        dtype='object')
    """
    """
    print(pred2_df.columns)
    Index(['label', 'metric', 'threshold', 'p_pass', 'n_pass', 'n_trials',
        'distance_from_threshold_nominal', 'nominal_core_mean_grad_gy_per_mm',
        'qa_class', 'p_hat_model'],
        dtype='object')
    """

    # NEW: compare 1D vs 2D (nested) fits per threshold label
    model_compare_df = compare_path1_logit_models(coef1_df, coef2_df)

    model_compare_path = qa_3_logit_grad / "p1_logit_grad_03_model_compare_1d_vs_2d.csv"
    model_compare_df.to_csv(model_compare_path, index=False)
    print(f"Saved {model_compare_path}")
    """
    print(model_compare_df.columns)
    Index(['label', 'metric', 'threshold', 'n_biopsies', 'aic_model1',
        'aic_model2', 'delta_aic', 'brier_model1', 'brier_model2',
        'delta_brier_w', 'rmse_model1', 'rmse_model2', 'delta_rmse_prob_w',
        'lr_stat', 'lr_df', 'lr_pvalue'],
        dtype='object')
    """
    # --- END NEW ---


    # --- Margin + geometry + radiomics merged dataset (for correlation analysis) ---
    margin_predictors_df = helper_funcs.build_path1_margin_with_spatial_and_radiomics(
        path1_enriched_df=path1_enriched_df,
        biopsy_basic_df=cohort_biopsy_basic_spatial_features_df,
        radiomics_df=cohort_3d_radiomic_features_all_oar_dil_df,
    )

    qa_geom_rad_path = qa_4_design / "p1_design_01_margin_geom_radiomics_basic.csv"
    margin_predictors_df.to_csv(qa_geom_rad_path, index=False)
    print(f"Saved merged QA + geometry + radiomics DF to: {qa_geom_rad_path}")
    """
    print(margin_predictors_df.columns)
    Index(['Patient ID', 'Bx index', 'Bx ID', 'metric', 'threshold',
       'comparison', 'label', 'p_pass', 'n_pass', 'n_trials', 'qa_class',
       'nominal_value', 'nominal_pass', 'distance_from_threshold_nominal',
       'misclassified', 'misclassification_type', 'metric_std', 'z_margin',
       'nominal_core_mean_dose_gy', 'nominal_core_mean_grad_gy_per_mm',
       'Simulated bool', 'Simulated type', 'Length (mm)', 'Volume (mm3)',
       'Voxel side length (mm)', 'Relative DIL ID', 'Relative DIL index',
       'BX to DIL centroid (X)', 'BX to DIL centroid (Y)',
       'BX to DIL centroid (Z)', 'BX to DIL centroid distance',
       'NN surface-surface distance', 'Relative prostate ID',
       'Relative prostate index', 'Bx position in prostate LR',
       'Bx position in prostate AP', 'Bx position in prostate SI',
       'DIL Volume', 'DIL Surface area', 'DIL Surface area to volume ratio',
       'DIL Sphericity', 'DIL Compactness 1', 'DIL Compactness 2',
       'DIL Spherical disproportion', 'DIL Maximum 3D diameter',
       'DIL PCA major', 'DIL PCA minor', 'DIL PCA least',
       'DIL PCA eigenvector major', 'DIL PCA eigenvector minor',
       'DIL PCA eigenvector least', 'DIL Major axis (equivalent ellipse)',
       'DIL Minor axis (equivalent ellipse)',
       'DIL Least axis (equivalent ellipse)', 'DIL Elongation', 'DIL Flatness',
       'DIL L/R dimension at centroid', 'DIL A/P dimension at centroid',
       'DIL S/I dimension at centroid', 'DIL S/I arclength',
       'DIL DIL centroid (X, prostate frame)',
       'DIL DIL centroid (Y, prostate frame)',
       'DIL DIL centroid (Z, prostate frame)',
       'DIL DIL centroid distance (prostate frame)',
       'DIL DIL prostate sextant (LR)', 'DIL DIL prostate sextant (AP)',
       'DIL DIL prostate sextant (SI)', 'Prostate Volume',
       'Prostate Surface area', 'Prostate Surface area to volume ratio',
       'Prostate Sphericity', 'Prostate Compactness 1',
       'Prostate Compactness 2', 'Prostate Spherical disproportion',
       'Prostate Maximum 3D diameter', 'Prostate PCA major',
       'Prostate PCA minor', 'Prostate PCA least',
       'Prostate PCA eigenvector major', 'Prostate PCA eigenvector minor',
       'Prostate PCA eigenvector least',
       'Prostate Major axis (equivalent ellipse)',
       'Prostate Minor axis (equivalent ellipse)',
       'Prostate Least axis (equivalent ellipse)', 'Prostate Elongation',
       'Prostate Flatness', 'Prostate L/R dimension at centroid',
       'Prostate A/P dimension at centroid',
       'Prostate S/I dimension at centroid', 'Prostate S/I arclength',
       'Prostate DIL centroid (X, prostate frame)',
       'Prostate DIL centroid (Y, prostate frame)',
       'Prostate DIL centroid (Z, prostate frame)',
       'Prostate DIL centroid distance (prostate frame)',
       'Prostate DIL prostate sextant (LR)',
       'Prostate DIL prostate sextant (AP)',
       'Prostate DIL prostate sextant (SI)',
       'BX_to_DIL_centroid_distance_norm_SI',
       'NN_surface_surface_distance_norm_SI'],
      dtype='object')
    """


    # giant merged dataset with margins enriched spatial radiomics + distances
    margin_predictors_with_radiomics_and_distances_df = helper_funcs.build_path1_margin_with_spatial_radiomics_and_distances(
        path1_enriched_df=path1_enriched_df,
        biopsy_basic_df=cohort_biopsy_basic_spatial_features_df,
        radiomics_df=cohort_3d_radiomic_features_all_oar_dil_df,
        distances_df=cohort_biopsy_level_distances_statistics_filtered_df,
        # radiomics_feature_cols=None,  # or pass a list if you want to restrict
    )

    out_path = qa_4_design / "p1_design_02_margin_spatial_radiomics_distances.csv"
    margin_predictors_with_radiomics_and_distances_df.to_csv(out_path, index=False)
    print(f"Saved QA + spatial + radiomics + distances design matrix to: {out_path}")
    """
    print(margin_predictors_with_radiomics_and_distances_df.columns.tolist())
    ['Patient ID', 'Bx index', 'Bx ID', 'metric', 'threshold', 'comparison', 'label', 'p_pass', 'n_pass', 'n_trials', 'qa_class', 'nominal_value', 'nominal_pass', 'distance_from_threshold_nominal', 'misclassified', 'misclassification_type', 'metric_std', 'z_margin', 'nominal_core_mean_dose_gy', 'nominal_core_mean_grad_gy_per_mm', 'Simulated bool', 'Simulated type', 'Length (mm)', 'Volume (mm3)', 'Voxel side length (mm)', 'Relative DIL ID', 'Relative DIL index', 'Relative prostate ID', 'Relative prostate index', 'Bx position in prostate LR', 'Bx position in prostate AP', 'Bx position in prostate SI', 'DIL Volume', 'DIL Surface area', 'DIL Surface area to volume ratio', 'DIL Sphericity', 'DIL Compactness 1', 'DIL Compactness 2', 'DIL Spherical disproportion', 'DIL Maximum 3D diameter', 'DIL PCA major', 'DIL PCA minor', 'DIL PCA least', 'DIL PCA eigenvector major', 'DIL PCA eigenvector minor', 'DIL PCA eigenvector least', 'DIL Major axis (equivalent ellipse)', 'DIL Minor axis (equivalent ellipse)', 'DIL Least axis (equivalent ellipse)', 'DIL Elongation', 'DIL Flatness', 'DIL L/R dimension at centroid', 'DIL A/P dimension at centroid', 'DIL S/I dimension at centroid', 'DIL S/I arclength', 'DIL DIL centroid (X, prostate frame)', 'DIL DIL centroid (Y, prostate frame)', 'DIL DIL centroid (Z, prostate frame)', 'DIL DIL centroid distance (prostate frame)', 'DIL DIL prostate sextant (LR)', 'DIL DIL prostate sextant (AP)', 'DIL DIL prostate sextant (SI)', 'Prostate Volume', 'Prostate Surface area', 'Prostate Surface area to volume ratio', 'Prostate Sphericity', 'Prostate Compactness 1', 'Prostate Compactness 2', 'Prostate Spherical disproportion', 'Prostate Maximum 3D diameter', 'Prostate PCA major', 'Prostate PCA minor', 'Prostate PCA least', 'Prostate PCA eigenvector major', 'Prostate PCA eigenvector minor', 'Prostate PCA eigenvector least', 'Prostate Major axis (equivalent ellipse)', 'Prostate Minor axis (equivalent ellipse)', 'Prostate Least axis (equivalent ellipse)', 'Prostate Elongation', 'Prostate Flatness', 'Prostate L/R dimension at centroid', 'Prostate A/P dimension at centroid', 'Prostate S/I dimension at centroid', 'Prostate S/I arclength', 'Prostate DIL centroid (X, prostate frame)', 'Prostate DIL centroid (Y, prostate frame)', 'Prostate DIL centroid (Z, prostate frame)', 'Prostate DIL centroid distance (prostate frame)', 'Prostate DIL prostate sextant (LR)', 'Prostate DIL prostate sextant (AP)', 'Prostate DIL prostate sextant (SI)', 'DIL NN dist mean', 'DIL centroid dist mean', 'Prostate NN dist mean', 'Prostate centroid dist mean', 'Rectum NN dist mean', 'Rectum centroid dist mean', 'Urethra NN dist mean', 'Urethra centroid dist mean', 'Prostate mean dimension at centroid', 'BX_to_prostate_centroid_distance_norm_mean_dim']
    """


    # -------------------------------------------------------------------------
    # PATH 1: Correlations between margin and spatial / radiomic predictors
    # -------------------------------------------------------------------------

    # Optional filter: real biopsies only
    design_for_corr = margin_predictors_with_radiomics_and_distances_df.copy()
    # If you want *only* real biopsies (not simulated), uncomment:
    # design_for_corr = design_for_corr[design_for_corr["Simulated bool"] == False].copy()

    # Explicit include-list of predictors we want to scan.
    # You can trim/extend this list as you like.
    margin_corr_predictors: list[str] = [
        # -- dose 
        #"nominal_core_mean_dose_gy",
        "nominal_core_mean_grad_gy_per_mm",

        # --- spatial distances (from distances_df) ---
        "BX_to_prostate_centroid_distance_norm_mean_dim",  # normalized biopsy–prostate centroid distance
        "DIL centroid dist mean",
        "Prostate centroid dist mean",
        "Rectum centroid dist mean",
        "Urethra centroid dist mean",
        "DIL NN dist mean",
        "Prostate NN dist mean",
        "Rectum NN dist mean",
        "Urethra NN dist mean",

        # --- basic biopsy geometry ---
        "Length (mm)",
        #"Volume (mm3)",
        #"NN surface-surface distance",
        #"BX to DIL centroid distance",

        # --- DIL radiomics (shape/size) ---
        "DIL Volume",
        "DIL Surface area",
        "DIL Surface area to volume ratio",
        "DIL Sphericity",
        "DIL Compactness 1",
        "DIL Compactness 2",
        "DIL Spherical disproportion",
        "DIL Maximum 3D diameter",
        "DIL PCA major",
        "DIL PCA minor",
        "DIL PCA least",

        # --- Prostate radiomics (shape/size) ---
        "Prostate Volume",
        "Prostate Surface area",
        "Prostate Surface area to volume ratio",
        "Prostate Sphericity",
        "Prostate Compactness 1",
        "Prostate Compactness 2",
        "Prostate Spherical disproportion",
        "Prostate Maximum 3D diameter",
        "Prostate PCA major",
        "Prostate PCA minor",
        "Prostate PCA least",
    ]

    # Compute correlations per DVH rule / threshold
    margin_corr_df = compute_margin_correlations_by_threshold(
        design_df=design_for_corr,
        predictor_cols=margin_corr_predictors,
        target_col="distance_from_threshold_nominal",
        label_col="label",  # path1 uses 'label' for the rule T label
    )

    corr_out_path = qa_5_correlations / "p1_corr_01_margin_vs_predictors_by_threshold.csv"
    margin_corr_df.to_csv(corr_out_path, index=False)
    print(f"Saved margin–predictor correlation table to: {corr_out_path}")
    """
    print(margin_corr_df.columns)  
    Index(['label', 'predictor', 'N', 'pearson_r', 'pearson_p', 'spearman_rho',
        'spearman_p', 'abs_pearson_r', 'abs_spearman_rho'],
        dtype='object')
    """


    # 2) categorical summaries
    categorical_predictors = [
        "Bx position in prostate LR",
        "Bx position in prostate AP",
        "Bx position in prostate SI",
        "DIL DIL prostate sextant (LR)",
        "DIL DIL prostate sextant (AP)",
        "DIL DIL prostate sextant (SI)",
    ]

    cat_summary_df = summarize_margin_by_categorical_predictors(
        design_df=design_for_corr,
        categorical_cols=categorical_predictors,
        margin_col="distance_from_threshold_nominal",
        rule_col="label",
    )

    cat_out = qa_5_correlations / "p1_corr_02_margin_categorical_summaries.csv"
    cat_summary_df.to_csv(cat_out, index=False)
    print(f"Saved categorical margin summaries to: {cat_out}")
    """
    print(cat_summary_df.columns)
    Index(['Rule', 'Predictor', 'Level', 'N', 'Margin mean', 'Margin std',
        'Margin median', 'Margin Q25', 'Margin Q75'],
        dtype='object')
    """











    # -------------------------------------------------------------------------
    # PATH 1: scan margin + X 2D logit models for a shortlist of predictors
    # -------------------------------------------------------------------------

    # Shortlist of secondary predictors to test in 2D logit (margin + X)
    # Adjust this list by inspecting the correlation CSV.
    secondary_predictors_for_scan: list[str] = [
        # -- dose 
        #"nominal_core_mean_dose_gy",
        "nominal_core_mean_grad_gy_per_mm",

        # --- spatial distances (from distances_df) ---
        "BX_to_prostate_centroid_distance_norm_mean_dim",  # normalized biopsy–prostate centroid distance
        "DIL centroid dist mean",
        "Prostate centroid dist mean",
        "Rectum centroid dist mean",
        "Urethra centroid dist mean",
        "DIL NN dist mean",
        "Prostate NN dist mean",
        "Rectum NN dist mean",
        "Urethra NN dist mean",

        # --- basic biopsy geometry ---
        "Length (mm)",
        #"Volume (mm3)",
        #"NN surface-surface distance",
        #"BX to DIL centroid distance",

        # --- DIL radiomics (shape/size) ---
        "DIL Volume",
        "DIL Surface area",
        "DIL Surface area to volume ratio",
        "DIL Sphericity",
        "DIL Compactness 1",
        "DIL Compactness 2",
        "DIL Spherical disproportion",
        "DIL Maximum 3D diameter",
        "DIL PCA major",
        "DIL PCA minor",
        "DIL PCA least",

        # --- Prostate radiomics (shape/size) ---
        "Prostate Volume",
        "Prostate Surface area",
        "Prostate Surface area to volume ratio",
        "Prostate Sphericity",
        "Prostate Compactness 1",
        "Prostate Compactness 2",
        "Prostate Spherical disproportion",
        "Prostate Maximum 3D diameter",
        "Prostate PCA major",
        "Prostate PCA minor",
        "Prostate PCA least",
    ]




    # Design matrix with margin + spatial + radiomics + distances
    # (cheap to build; if you also build it later for correlations,
    # you can share or keep this and remove the later duplicate.)
    design_for_models = helper_funcs.build_path1_margin_with_spatial_radiomics_and_distances(
        path1_enriched_df=path1_enriched_df,
        biopsy_basic_df=cohort_biopsy_basic_spatial_features_df,
        radiomics_df=cohort_3d_radiomic_features_all_oar_dil_df,
        distances_df=cohort_biopsy_level_distances_statistics_filtered_df,
        # radiomics_feature_cols=None,
    )  
    """
    print(design_for_models.columns.tolist())
    ['Patient ID', 'Bx index', 'Bx ID', 'metric', 'threshold', 'comparison', 'label', 'p_pass', 'n_pass', 'n_trials', 'qa_class', 'nominal_value', 'nominal_pass', 'distance_from_threshold_nominal', 'misclassified', 'misclassification_type', 'metric_std', 'z_margin', 'nominal_core_mean_dose_gy', 'nominal_core_mean_grad_gy_per_mm', 'Simulated bool', 'Simulated type', 'Length (mm)', 'Volume (mm3)', 'Voxel side length (mm)', 'Relative DIL ID', 'Relative DIL index', 'Relative prostate ID', 'Relative prostate index', 'Bx position in prostate LR', 'Bx position in prostate AP', 'Bx position in prostate SI', 'DIL Volume', 'DIL Surface area', 'DIL Surface area to volume ratio', 'DIL Sphericity', 'DIL Compactness 1', 'DIL Compactness 2', 'DIL Spherical disproportion', 'DIL Maximum 3D diameter', 'DIL PCA major', 'DIL PCA minor', 'DIL PCA least', 'DIL PCA eigenvector major', 'DIL PCA eigenvector minor', 'DIL PCA eigenvector least', 'DIL Major axis (equivalent ellipse)', 'DIL Minor axis (equivalent ellipse)', 'DIL Least axis (equivalent ellipse)', 'DIL Elongation', 'DIL Flatness', 'DIL L/R dimension at centroid', 'DIL A/P dimension at centroid', 'DIL S/I dimension at centroid', 'DIL S/I arclength', 'DIL DIL centroid (X, prostate frame)', 'DIL DIL centroid (Y, prostate frame)', 'DIL DIL centroid (Z, prostate frame)', 'DIL DIL centroid distance (prostate frame)', 'DIL DIL prostate sextant (LR)', 'DIL DIL prostate sextant (AP)', 'DIL DIL prostate sextant (SI)', 'Prostate Volume', 'Prostate Surface area', 'Prostate Surface area to volume ratio', 'Prostate Sphericity', 'Prostate Compactness 1', 'Prostate Compactness 2', 'Prostate Spherical disproportion', 'Prostate Maximum 3D diameter', 'Prostate PCA major', 'Prostate PCA minor', 'Prostate PCA least', 'Prostate PCA eigenvector major', 'Prostate PCA eigenvector minor', 'Prostate PCA eigenvector least', 'Prostate Major axis (equivalent ellipse)', 'Prostate Minor axis (equivalent ellipse)', 'Prostate Least axis (equivalent ellipse)', 'Prostate Elongation', 'Prostate Flatness', 'Prostate L/R dimension at centroid', 'Prostate A/P dimension at centroid', 'Prostate S/I dimension at centroid', 'Prostate S/I arclength', 'Prostate DIL centroid (X, prostate frame)', 'Prostate DIL centroid (Y, prostate frame)', 'Prostate DIL centroid (Z, prostate frame)', 'Prostate DIL centroid distance (prostate frame)', 'Prostate DIL prostate sextant (LR)', 'Prostate DIL prostate sextant (AP)', 'Prostate DIL prostate sextant (SI)', 'DIL NN dist mean', 'DIL centroid dist mean', 'Prostate NN dist mean', 'Prostate centroid dist mean', 'Rectum NN dist mean', 'Rectum centroid dist mean', 'Urethra NN dist mean', 'Urethra centroid dist mean', 'Prostate mean dimension at centroid', 'BX_to_prostate_centroid_distance_norm_mean_dim']
    """

    # If you want ONLY real biopsies (not simulated) in these models, uncomment:
    # design_for_models = design_for_models[design_for_models["Simulated bool"] == False].copy()

    base_margin_col = "distance_from_threshold_nominal"
    required_cols = [base_margin_col, 'metric', 'threshold', 'label', 'p_pass', 'n_pass', 'n_trials', 'qa_class'] + secondary_predictors_for_scan

    missing_cols = [c for c in required_cols if c not in design_for_models.columns]
    if missing_cols:
        raise KeyError(
            f"Design df is missing required columns for 2D logit scan: {missing_cols}"
        )

    # Complete-case subset for fair comparison (same biopsies for 1D and every 2D)
    design_cc = design_for_models.dropna(subset=required_cols).copy()
    print(f"PATH1 2D scan: using {len(design_cc)} rows after complete-case filtering.")
    """
    print(design_cc.columns.tolist())
    ['Patient ID', 'Bx index', 'Bx ID', 'metric', 'threshold', 'comparison', 'label', 'p_pass', 'n_pass', 'n_trials', 'qa_class', 'nominal_value', 'nominal_pass', 'distance_from_threshold_nominal', 'misclassified', 'misclassification_type', 'metric_std', 'z_margin', 'nominal_core_mean_dose_gy', 'nominal_core_mean_grad_gy_per_mm', 'Simulated bool', 'Simulated type', 'Length (mm)', 'Volume (mm3)', 'Voxel side length (mm)', 'Relative DIL ID', 'Relative DIL index', 'Relative prostate ID', 'Relative prostate index', 'Bx position in prostate LR', 'Bx position in prostate AP', 'Bx position in prostate SI', 'DIL Volume', 'DIL Surface area', 'DIL Surface area to volume ratio', 'DIL Sphericity', 'DIL Compactness 1', 'DIL Compactness 2', 'DIL Spherical disproportion', 'DIL Maximum 3D diameter', 'DIL PCA major', 'DIL PCA minor', 'DIL PCA least', 'DIL PCA eigenvector major', 'DIL PCA eigenvector minor', 'DIL PCA eigenvector least', 'DIL Major axis (equivalent ellipse)', 'DIL Minor axis (equivalent ellipse)', 'DIL Least axis (equivalent ellipse)', 'DIL Elongation', 'DIL Flatness', 'DIL L/R dimension at centroid', 'DIL A/P dimension at centroid', 'DIL S/I dimension at centroid', 'DIL S/I arclength', 'DIL DIL centroid (X, prostate frame)', 'DIL DIL centroid (Y, prostate frame)', 'DIL DIL centroid (Z, prostate frame)', 'DIL DIL centroid distance (prostate frame)', 'DIL DIL prostate sextant (LR)', 'DIL DIL prostate sextant (AP)', 'DIL DIL prostate sextant (SI)', 'Prostate Volume', 'Prostate Surface area', 'Prostate Surface area to volume ratio', 'Prostate Sphericity', 'Prostate Compactness 1', 'Prostate Compactness 2', 'Prostate Spherical disproportion', 'Prostate Maximum 3D diameter', 'Prostate PCA major', 'Prostate PCA minor', 'Prostate PCA least', 'Prostate PCA eigenvector major', 'Prostate PCA eigenvector minor', 'Prostate PCA eigenvector least', 'Prostate Major axis (equivalent ellipse)', 'Prostate Minor axis (equivalent ellipse)', 'Prostate Least axis (equivalent ellipse)', 'Prostate Elongation', 'Prostate Flatness', 'Prostate L/R dimension at centroid', 'Prostate A/P dimension at centroid', 'Prostate S/I dimension at centroid', 'Prostate S/I arclength', 'Prostate DIL centroid (X, prostate frame)', 'Prostate DIL centroid (Y, prostate frame)', 'Prostate DIL centroid (Z, prostate frame)', 'Prostate DIL centroid distance (prostate frame)', 'Prostate DIL prostate sextant (LR)', 'Prostate DIL prostate sextant (AP)', 'Prostate DIL prostate sextant (SI)', 'DIL NN dist mean', 'DIL centroid dist mean', 'Prostate NN dist mean', 'Prostate centroid dist mean', 'Rectum NN dist mean', 'Rectum centroid dist mean', 'Urethra NN dist mean', 'Urethra centroid dist mean', 'Prostate mean dimension at centroid', 'BX_to_prostate_centroid_distance_norm_mean_dim']
    """


    # --- 1D: margin-only model (on the same complete-case data) ---
    coef1_scan_df, pred1_scan_df = fit_path1_logit_per_threshold(
        design_cc,
        predictors=(base_margin_col,),
    )
    """
    print(coef1_scan_df.columns)
    Index(['label', 'metric', 'threshold', 'predictors', 'n_biopsies',
        'n_trials_mean', 'll_eff', 'll_null_eff', 'aic', 'mcfadden_r2',
        'brier_w', 'rmse_prob_w', 'k_params', 'b_const',
        'b_distance_from_threshold_nominal', 'margin_at_p50', 'margin_at_p95'],
        dtype='object')

    print(pred1_scan_df.columns)
    Index(['label', 'metric', 'threshold', 'p_pass', 'n_pass', 'n_trials',
        'distance_from_threshold_nominal', 'qa_class', 'p_hat_model'],
        dtype='object')
    """
    # (You can save these if you want; they’re the "margin-only" fits
    # restricted to the complete-case subset used for the 2D models.)

    # --- 2D: margin + X models for each secondary predictor in the shortlist ---
    all_coef2: list[pd.DataFrame] = []
    all_pred2: list[pd.DataFrame] = []
    all_compare: list[pd.DataFrame] = []

    for sec_col in secondary_predictors_for_scan:
        print(f"  Fitting 2D logit model with secondary predictor: {sec_col}")

        # 2D model: margin + sec_col
        coef2_df_alt, pred2_df_alt = fit_path1_logit_per_threshold(
            design_cc,
            predictors=(base_margin_col, sec_col),
        )

        # Tag with which secondary predictor we used
        coef2_df_alt["secondary_predictor"] = sec_col
        pred2_df_alt["secondary_predictor"] = sec_col

        all_coef2.append(coef2_df_alt)
        all_pred2.append(pred2_df_alt)

        # Compare 1D vs 2D for this secondary predictor
        cmp_df = compare_path1_logit_models(
            coef1_df=coef1_scan_df,
            coef2_df=coef2_df_alt,
        )
        cmp_df["secondary_predictor"] = sec_col
        all_compare.append(cmp_df)







    # Concatenate results across all secondary predictors
    if all_coef2:
        coef2_all_df = pd.concat(all_coef2, ignore_index=True)
        pred2_all_df = pd.concat(all_pred2, ignore_index=True)
        model_compare_all_df = pd.concat(all_compare, ignore_index=True)

        # Save to disk
        coef2_all_path = qa_6_secondary_scan / "p1_secscan_01_coef_margin_plus_all_secondaries.csv"
        pred2_all_path = qa_6_secondary_scan / "p1_secscan_02_predictions_margin_plus_all_secondaries.csv"
        cmp_all_path = qa_6_secondary_scan / "p1_secscan_03_model_compare_all_vs_margin_raw.csv"

        coef2_all_df.to_csv(coef2_all_path, index=False)
        pred2_all_df.to_csv(pred2_all_path, index=False)
        model_compare_all_df.to_csv(cmp_all_path, index=False)

        print(f"Saved 2D coef table (margin + secondary) to: {coef2_all_path}")
        print(f"Saved 2D prediction table (margin + secondary) to: {pred2_all_path}")
        print(f"Saved model comparison table (margin vs margin+secondary) to: {cmp_all_path}")
    
    
        # --- Summarize secondary-predictor scan ---

        # 1) Best secondary predictor per rule (metric/threshold/label),
        #    based primarily on ΔAIC (2D − 1D). More negative = better 2D model.
        sorted_cmp = model_compare_all_df.sort_values(
            by=["metric", "threshold", "label", "delta_aic", "delta_brier_w", "lr_pvalue"],
            ascending=[True,   True,        True,    True,       True,           True],
        )

        sorted_cmp_path = qa_6_secondary_scan / "p1_secscan_04_model_compare_all_vs_margin_sorted.csv"
        sorted_cmp.to_csv(sorted_cmp_path, index=False)
        print(f"Saved sorted best secondary predictor grouped by threshold to: {sorted_cmp_path}")

        best_per_rule = (
            sorted_cmp
            .groupby(["metric", "threshold", "label"], as_index=False)
            .first()
            .copy()
        )

        best_per_rule_path = qa_6_secondary_scan / "p1_secscan_05_best_secondary_per_threshold.csv"
        best_per_rule.to_csv(best_per_rule_path, index=False)
        print(f"Saved best secondary predictor per threshold to: {best_per_rule_path}")

        # 2) Overall ranking of secondary predictors across all rules.
        #    We look at median improvements and how often each predictor helps.
        predictor_summary = (
            model_compare_all_df
            .groupby("secondary_predictor")
            .agg(
                n_rules=("label", "nunique"),
                median_delta_aic=("delta_aic", "median"),
                median_delta_brier=("delta_brier_w", "median"),
                median_delta_rmse=("delta_rmse_prob_w", "median"),
                frac_aic_better=("delta_aic", lambda x: (x < 0.0).mean()),
                frac_brier_better=("delta_brier_w", lambda x: (x < 0.0).mean()),
                frac_rmse_better=("delta_rmse_prob_w", lambda x: (x < 0.0).mean()),
                frac_lr_sig=("lr_pvalue", lambda x: (x < 0.05).mean()),
            )
            .reset_index()
        )

        # Sort so "most helpful" (most negative median ΔAIC) appears first
        predictor_ranking = predictor_summary.sort_values(
            by="median_delta_aic",
            ascending=True,   # more negative = stronger improvement
        )

        predictor_ranking_path = qa_6_secondary_scan / "p1_secscan_06_secondary_ranking_overall.csv"
        predictor_ranking.to_csv(predictor_ranking_path, index=False)
        print(f"Saved secondary predictor ranking to: {predictor_ranking_path}")


        # 3) Δδ̂_0.95 vs best secondary predictor per threshold
        from qa_path1_thresholds import compute_delta95_vs_best_secondary_per_threshold

        delta95_effect_df = compute_delta95_vs_best_secondary_per_threshold(
            path1_results_df=path1_results_df,
            design_cc=design_cc,
            coef2_all_df=coef2_all_df,
            best_per_rule=best_per_rule,
        )

        delta95_effect_path = (
            qa_6_secondary_scan / "p1_secscan_07_delta95_vs_best_secondary_per_threshold.csv"
        )
        delta95_effect_df.to_csv(delta95_effect_path, index=False)
        print(
            "Saved Δδ̂_0.95 vs best secondary predictor summary to:\n"
            f"  {delta95_effect_path}"
        )



    
    else:
        print("No secondary predictors specified for 2D scan.")







    # --- NEW: Path-1 figures (production_plots) ---
    qa_fig_dir = qa_dir / "figures"

    # Path-1 QA figures for paper
    """
    production_plots.production_plot_path1_threshold_qa_summary(
        path1_results_df,
        save_dir=qa_fig_dir,
        file_name="Fig_Path1_threshold_QA_summary.png",
    )

    production_plots.production_plot_path1_p_pass_vs_margin_by_metric(
        path1_results_df,
        save_dir=qa_fig_dir,
        file_name="Fig_Path1_p_pass_vs_margin_by_metric.png",
    )

    production_plots.production_plot_path1_logit_margin_plus_grad_families(
        pred_df=pred2_df,
        coef_df=coef2_df,
        save_dir=qa_fig_dir,
        file_name="Fig_Path1_logit_margin_plus_grad_families.png",
        n_grad_levels=4,  # or tweak if you want more / fewer families
    )
    """

    # --- Path-1 QA summary figure (A: stacked bars, B: p_pass distributions) ---
    # What we want explicit on the plots:
    #   - percent labels in bar blocks + callouts for tiny blocks (default in our updated function)
    #   - legend moved outside so it never covers the bars
    production_plots.production_plot_path1_threshold_qa_summary(
        path1_results_df,
        save_dir=qa_fig_dir,
        file_name="Fig_Path1_threshold_QA_summary.png",
        legend_outside=True,          # move legend outside panel A
        annotate_percents=True,       # show % labels (and callouts when too small)
        percent_fmt="{:.0f}%",        # e.g., 37%
        min_count_inside=2,           # tiny counts -> callout
        min_frac_inside=0.08,         # tiny fractions -> callout
        show_title = False,

    )

    production_plots.production_plot_path1_threshold_qa_summary_v2(
        path1_results_df,
        save_dir=qa_fig_dir,
        file_name="Fig_Path1_threshold_QA_summary_v2.png",
        legend_outside=True,          # move legend outside panel A
        annotate_percents=True,       # show % labels (and callouts when too small)
        percent_fmt="{:.0f}%",        # e.g., 37%
        min_count_inside=2,           # tiny counts -> callout
        min_frac_inside=0.08,         # tiny fractions -> callout
        show_title = False,

    )

    # --- Single-predictor (margin-only) logit plots ---
    # What we want explicit on the plots:
    #   - goodness-of-fit metrics per panel (optional)
    #   - vertical "required margin" line for p_hat = 0.95 (optional) + numeric annotation

    

    production_plots.production_plot_path1_p_pass_vs_margin_by_metric(
        path1_results_df,
        save_dir=qa_fig_dir,
        file_name="Fig_Path1_p_pass_vs_margin_by_metric.png",
        add_logistic_fit=True,
        annotate_fit_stats=True,                 # print GOF per subplot
        fit_stats=("n", 
                   "mcfadden_r2", 
                   "wrmse", 
                   #"brier",
                   ),
        show_required_margin_line=True,
        required_prob=0.95,
        required_line_kwargs={                  # style of the δ̂ line
            "linestyle": "--",
            "linewidth": 1.2,
            "alpha": 0.9,
            "color": "black",
        },
    )

    # --- Two-predictor (margin + gradient) logit family plots ---
    # Show:
    #   - gradient families (chosen from pred2_df[grad_col])
    #   - comparative fit metrics vs 1D model (ΔAIC, ΔBrier, ΔWRMSE, LR p, etc.)
    production_plots.production_plot_path1_logit_margin_plus_grad_families(
        pred_df=pred2_df,
        coef_df=coef2_df,

        # comparison info (1D vs 2D)
        comparison_df=model_compare_df,          # has column "label", so default comparison_label_col is OK

        save_dir=qa_fig_dir,
        file_name="Fig_Path1_logit_margin_plus_grad_families.png",

        n_grad_levels=4,
        grad_quantiles=(0.10, 0.37, 0.63, 0.90),
        prob_pass_cutoffs=(0.05, 0.95),

        # show the comparison stats box (using columns from coef2_df + model_compare_df)
        annotate_fit_stats=True,
        fit_stats=(
            "n",           # N biopsies (from coef2_df['n_biopsies'])
            "delta_aic",   # AIC_2D − AIC_1D
            "delta_brier", # ΔwBrier (2D − 1D)
            #"delta_rmse",  # ΔwRMSE (2D − 1D)
            "lr_p",        # LR test p-value
        ),

        # overlay the margin-only (1D) curve
        overlay_1d_model=True,
        coef1_df=coef1_df,          # 1D coefficients; label column is also "label"

        overlay_1d_kwargs={         # optional styling for the 1D curve
            "color": "black",
            "linestyle": "--",
            "linewidth": 2.0,
            "alpha": 0.9,
        },
    )



    # --- Path-1: 2-predictor logit figure using best secondary per threshold ---

    # Restrict coef/pred tables to the chosen secondary predictor for each rule
    merge_cols = ["metric", "threshold", "label", "secondary_predictor"]

    coef2_best = coef2_all_df.merge(
        best_per_rule[merge_cols],
        on=merge_cols,
        how="inner",
    )
    pred2_best = pred2_all_df.merge(
        best_per_rule[merge_cols],
        on=merge_cols,
        how="inner",
    )

    # Map each rule label -> secondary predictor column used in that panel
    per_label_secondary = {
        row["label"]: row["secondary_predictor"]
        for _, row in best_per_rule.iterrows()
    }

    # Panel-specific legend labels: use generic \hat{g}_i for each rule
    per_label_legend_title = {
        "D2 ≥ 32 Gy":  r"Levels of $\hat{g}_1$",
        "D50 ≥ 27 Gy": r"Levels of $\hat{g}_2$",
        "D98 ≥ 20 Gy": r"Levels of $\hat{g}_3$",
        "V150 ≥ 50%":  r"Levels of $\hat{g}_4$",
    }

    per_label_grad_label_template = {
        "D2 ≥ 32 Gy":  r"$\hat{{g}}_1 = {value:.2f}\ {unit}$",
        "D50 ≥ 27 Gy": r"$\hat{{g}}_2 = {value:.2f}\ {unit}$",
        "D98 ≥ 20 Gy": r"$\hat{{g}}_3 = {value:.2f}\ {unit}$",
        "V150 ≥ 50%":  r"$\hat{{g}}_4 = {value:.2f}\ {unit}$",
    }

    per_label_secondary_unit = {
        "D2 ≥ 32 Gy":  r"\mathrm{Gy\ mm^{-1}}",
        "D50 ≥ 27 Gy": r"\mathrm{Dimless}",
        "D98 ≥ 20 Gy": r"\mathrm{Gy\ mm^{-1}}",
        "V150 ≥ 50%":  r"\mathrm{mm}",
    }

    """
    per_label_secondary_annotation = {
        "D2 ≥ 32 Gy":  r"Secondary predictor: $\hat{g}_1$ | Core nominal dose gradient ($\mathrm{Gy\ mm^{-1}}$)",
        "D50 ≥ 27 Gy": r"Secondary predictor: $\hat{g}_2$ | DIL Spherical disproportion ($\mathrm{Dimless}$)",
        "D98 ≥ 20 Gy": r"Secondary predictor: $\hat{g}_3$ | Core nominal dose gradient ($\mathrm{Gy\ mm^{-1}}$)",
        "V150 ≥ 50%":  r"Secondary predictor: $\hat{g}_4$ | Rectum mean NN distance ($\mathrm{mm}$)",
    }
    """
    per_label_secondary_annotation = {
        "D2 ≥ 32 Gy": (
            r"Secondary predictor: "
            r"$\hat{g}_1 = \overline{G}^{(0)}_b$" "\n"
            r"(core nominal dose gradient, $\mathrm{Gy\ mm^{-1}}$)"
        ),
        "D50 ≥ 27 Gy": (
            r"Secondary predictor: "
            r"$\hat{g}_2 = \mathrm{SphDisp}_{\mathrm{DIL}}$" "\n"
            r"(DIL spherical disproportion, dimensionless)"
        ),
        "D98 ≥ 20 Gy": (
            r"Secondary predictor: "
            r"$\hat{g}_3 = \overline{G}^{(0)}_b$" "\n"
            r"(core nominal dose gradient, $\mathrm{Gy\ mm^{-1}}$)"
        ),
        "V150 ≥ 50%": (
            r"Secondary predictor: "
            r"$\hat{g}_4 = \overline{d}^{\mathrm{NN}}_{\mathrm{R}}$" "\n"
            r"(rectum mean NN distance, $\mathrm{mm}$)"
        ),
    }



    """
    per_label_secondary_annotation = {
        "D2 ≥ 32 Gy":  r"Secondary predictor: $\hat{g}_1$ | REDACTED ($\mathrm{Gy\ mm^{-1}}$)",
        "D50 ≥ 27 Gy": r"Secondary predictor: $\hat{g}_2$ | REDACTED ($\mathrm{Dimless}$)",
        "D98 ≥ 20 Gy": r"Secondary predictor: $\hat{g}_3$ | REDACTED ($\mathrm{Gy\ mm^{-1}}$)",
        "V150 ≥ 50%":  r"Secondary predictor: $\hat{g}_4$ | REDACTED ($\mathrm{mm}$)",
    }
    """

    per_label_stats_box_corner = {
            #"V150 ≥ 50%": "top-left",   # move that box to top-left for panel D
            # you can add others later if needed
        }
    per_label_stats_box_xy = {
        "V150 ≥ 50%": (0.02, 0.60),   # axes fractions (x, y)
    }


    production_plots.production_plot_path1_logit_margin_plus_grad_families_generalized(
        pred_df=pred2_best,
        coef_df=coef2_best,

        metric_col="metric",
        threshold_col="threshold",
        label_col="label",
        margin_col="distance_from_threshold_nominal",
        grad_col="nominal_core_mean_grad_gy_per_mm",  # only for colour scale
        p_pass_col="p_pass",
        qa_class_col="qa_class",

        comparison_df=best_per_rule,
        comparison_label_col="label",

        overlay_1d_model=True,
        coef1_df=coef1_scan_df,
        coef1_label_col="label",

        per_label_secondary=per_label_secondary,
        per_label_grad_label_template=per_label_grad_label_template,
        per_label_legend_title=per_label_legend_title,

        per_label_secondary_unit=per_label_secondary_unit,
        per_label_secondary_annotation=per_label_secondary_annotation,

        save_dir=qa_fig_dir,
        file_name="Fig_Path1_logit_margin_plus_best_secondary_families.png",
        n_grad_levels=4,
        grad_quantiles=(0.10, 0.37, 0.63, 0.90),
        prob_pass_cutoffs=(0.05, 0.95),

        # NEW: show the improvement box, same style as the gradient-only fig
        annotate_fit_stats=True,
        fit_stats=(
            "n",            # N biopsies
            "delta_aic",    # 2D – 1D
            #"delta_brier",  # uses delta_brier_w under the hood
            "delta_rmse", # optional
            "lr_p",         # LR test p-value
        ),
        per_label_stats_box_corner = per_label_stats_box_corner,
        per_label_stats_box_xy = per_label_stats_box_xy,
    )






    # --- END NEW BLOCK ---











    ### COMPUTE CUMULATIVE DVH CURVES PER TRIAL (START)



    # Cumulative DVH (one row per unique dose per trial)
    all_cumulative_dvh_by_mc_trial_number_df = helper_funcs.build_cumulative_dvh_by_mc_trial_number_df(
        all_voxel_wise_dose_df
    )



    ### COMPUTE CUMULATIVE DVH CURVES PER TRIAL (END)







    ### voxel wise nominal-MC trial delta  (START)

    # Build per-trial deltas relative to nominal (trial 0)
    mc_deltas = summary_statistics.compute_mc_trial_deltas_with_abs(all_voxel_wise_dose_df)
    """print(mc_deltas.columns)
Index([                                               'Patient ID',
                                                        'Bx index',
                                                     'Voxel index',
                                                           'Bx ID',
                                                 'Voxel begin (Z)',
                                                   'Voxel end (Z)',
                                                  'Simulated bool',
                                                  'Simulated type',
                                                    'X (Bx frame)',
                                                    'Y (Bx frame)',
                                                    'Z (Bx frame)',
                                                    'R (Bx frame)',
                                                    'MC trial num',
                       ('Dose (Gy) deltas', 'nominal_minus_trial'),
               ('Dose (Gy) abs deltas', 'abs_nominal_minus_trial'),
               ('Dose grad (Gy/mm) deltas', 'nominal_minus_trial'),
       ('Dose grad (Gy/mm) abs deltas', 'abs_nominal_minus_trial')],
      dtype='object')"""

    print(f"Shape of mc_deltas dataframe: {mc_deltas.shape}")

    # Create output directory for voxel-wise nominal-mode nominal-mean nominal-q50 analysis
    voxel_wise_nominal_analysis_dir = output_dir.joinpath("voxel_wise_nominal_analysis")
    os.makedirs(voxel_wise_nominal_analysis_dir, exist_ok=True)


    # Summarizes the signed nominal–trial deltas across all voxels/trials for each metric
    # (n, mean, std, quantiles, etc.) and writes a single CSV per cohort.

    # existing overall summary (all groups pooled)
    summary_path = summary_statistics.save_mc_delta_summary_csv(
        mc_deltas,
        output_dir=voxel_wise_nominal_analysis_dir,
        csv_name="mc_trial_deltas_summary.csv",
        value_cols=('Dose (Gy)', 'Dose grad (Gy/mm)'),
        include_patient_ids=None,
        decimals=3
    )

    # new grouped summaries
    voxel_csv, biopsy_csv = summary_statistics.save_mc_delta_grouped_csvs(
        mc_deltas,
        output_dir=voxel_wise_nominal_analysis_dir,
        base_name="mc_trial_deltas",
        value_cols=('Dose (Gy)', 'Dose grad (Gy/mm)'),
        decimals=3
    )

    print("Saved:")
    print(" - overall:", summary_path)
    print(" - per-voxel:", voxel_csv)
    print(" - per-biopsy:", biopsy_csv)


    print('mc deltas summary csv saved to file')

    """
    csv_path = summary_statistics.save_paired_effect_sizes_by_trial_csv_fast(
    mc_deltas,
    output_dir=voxel_wise_nominal_analysis_dir,
    csv_name="paired_effect_sizes_by_trial.csv",
    value_cols=('Dose (Gy)', 'Dose grad (Gy/mm)'),
    decimals=3
    )
    print('mc paired effect sizes by trial csv saved to file')
    print(csv_path)
    """

    # For each voxel, estimates P(nominal > trial), P(=), P(<) across trials (CLES components),
    # writing both a per-voxel CSV and a per-biopsy (median across voxels) CSV.
    vox_path, cles_voxel_stats, bio_path, cles_biopsy_stats = summary_statistics.save_nominal_vs_trial_proportions_csv(
        mc_deltas,
        output_dir=voxel_wise_nominal_analysis_dir,
        base_name="paired_effect_sizes_by_trial",
        value_cols = ('Dose (Gy)', 'Dose grad (Gy/mm)'),
        exclude_nominal_trial = True,   # drop trial 0 rows if present
        decimals = 3,
    )
    print('mc nominal vs trial proportions csv saved to file')
    print(vox_path)
    print(bio_path)

    # Builds cohort-wide CLES summaries: pooled proportions (weighted by voxel trial counts)
    # and the distribution of biopsy-level CLES medians; writes one CSV per cohort.
    summary_statistics.save_cohort_cles_summary_csv(
        cles_voxel_stats,           # DataFrame OR path to "*_per_voxel.csv"
        cles_biopsy_stats,          # DataFrame OR path to "*_per_biopsy.csv"
        output_dir=voxel_wise_nominal_analysis_dir,
        csv_name = "cohort_cles_summary.csv",
        metrics_col = "metric",
        decimals = 3,
    )


    # Direct cohort-pooled CLES computed from all nominal–trial pairs (no voxel/biopsy grouping),
    # writing a compact CSV with n_pairs and pooled prop_gt/eq/lt, CLES_strict, CLES_ties per metric.
    cohort_csv_pooled_cles_from_mc_deltas, pooled = summary_statistics.save_cohort_pooled_cles_from_mc_deltas(
        mc_deltas,
        output_dir=voxel_wise_nominal_analysis_dir,
        csv_name="cohort_pooled_cles.csv",
        value_cols=('Dose (Gy)', 'Dose grad (Gy/mm)'),
    )
    print(cohort_csv_pooled_cles_from_mc_deltas)

    ### voxel-wise nominal-mode nominal-mean nominal-q50 analysis (START)
    
    """
    nominal_deltas_df = summary_statistics.compute_biopsy_nominal_deltas(cohort_global_dosimetry_by_voxel_df)


    nominal_gradient_deltas_df = summary_statistics.compute_biopsy_nominal_deltas(cohort_global_dosimetry_by_voxel_df,
                                                                                           zero_level_index_str='Dose grad (Gy/mm)')
    """

    nominal_deltas_df_with_abs = summary_statistics.compute_biopsy_nominal_deltas_with_abs(cohort_global_dosimetry_by_voxel_df)
    """
    print(nominal_deltas_df_with_abs.columns) = 
    MultiIndex([(     'Voxel begin (Z)',                       ''),
                (       'Voxel end (Z)',                       ''),
                (         'Voxel index',                       ''),
                (          'Patient ID',                       ''),
                (               'Bx ID',                       ''),
                (            'Bx index',                       ''),
                (      'Simulated bool',                       ''),
                (      'Simulated type',                       ''),
                (    'Dose (Gy) deltas',     'nominal_minus_mean'),
                (    'Dose (Gy) deltas',     'nominal_minus_mode'),
                (    'Dose (Gy) deltas',      'nominal_minus_q50'),
                ('Dose (Gy) abs deltas', 'abs_nominal_minus_mean'),
                ('Dose (Gy) abs deltas', 'abs_nominal_minus_mode'),
                ('Dose (Gy) abs deltas',  'abs_nominal_minus_q50')],
            )
    """
    nominal_gradient_deltas_df_with_abs = summary_statistics.compute_biopsy_nominal_deltas_with_abs(cohort_global_dosimetry_by_voxel_df,
                                                                                           zero_level_index_str='Dose grad (Gy/mm)')
    """
    print(nominal_gradient_deltas_df_with_abs.columns) =
    MultiIndex([(             'Voxel begin (Z)',                       ''),
                (               'Voxel end (Z)',                       ''),
                (                 'Voxel index',                       ''),
                (                  'Patient ID',                       ''),
                (                       'Bx ID',                       ''),
                (                    'Bx index',                       ''),
                (              'Simulated bool',                       ''),
                (              'Simulated type',                       ''),
                (    'Dose grad (Gy/mm) deltas',     'nominal_minus_mean'),
                (    'Dose grad (Gy/mm) deltas',     'nominal_minus_mode'),
                (    'Dose grad (Gy/mm) deltas',      'nominal_minus_q50'),
                ('Dose grad (Gy/mm) abs deltas', 'abs_nominal_minus_mean'),
                ('Dose grad (Gy/mm) abs deltas', 'abs_nominal_minus_mode'),
                ('Dose grad (Gy/mm) abs deltas',  'abs_nominal_minus_q50')],
            )
    """



    # Dose (Gy)
    dose_csv = summary_statistics.save_nominal_delta_biopsy_stats(
        nominal_deltas_df_with_abs,
        output_dir=voxel_wise_nominal_analysis_dir,
        base_name="nominal_deltas_dose",
        value_blocks=('Dose (Gy)',),
        decimals=3
    )

    # Dose grad (Gy/mm)
    grad_csv = summary_statistics.save_nominal_delta_biopsy_stats(
        nominal_gradient_deltas_df_with_abs,
        output_dir=voxel_wise_nominal_analysis_dir,
        base_name="nominal_deltas_grad",
        value_blocks=('Dose grad (Gy/mm)',),
        decimals=3
    )




    print('print to file')
    """
    csv_path = summary_statistics.save_delta_boxplot_summary_csv_with_absolute(
        nominal_deltas_df,
        output_dir=voxel_wise_nominal_analysis_dir,
        csv_name='dose_deltas_boxplot_summary.csv',
        zero_level_index_str='Dose (Gy)',
        include_patient_ids=None,   # or ['184','201']
        decimals=3
    )

    csv_path = summary_statistics.save_delta_boxplot_summary_csv_with_absolute(
        nominal_gradient_deltas_df,
        output_dir=voxel_wise_nominal_analysis_dir,
        csv_name='dose_gradient_deltas_boxplot_summary.csv',
        zero_level_index_str='Dose grad (Gy/mm)',
        include_patient_ids=None,   # or ['184','201']
        decimals=3
    )
    """




    # begin analysis of nominal bias versus predictors (spatial/radiomics/distances)


    delta_corr_dir = output_dir.joinpath("deltas_bias_correlations")
    os.makedirs(delta_corr_dir, exist_ok=True)

    

    delta_design_df = helper_funcs.build_deltas_with_spatial_radiomics_and_distances_v2(
        nominal_deltas_df_with_abs=nominal_deltas_df_with_abs,
        biopsy_basic_df=cohort_biopsy_basic_spatial_features_df,
        radiomics_df=cohort_3d_radiomic_features_all_oar_dil_df,
        distances_df=cohort_voxel_level_distances_statistics_filtered_df,
        all_voxel_wise_dose_df=all_voxel_wise_dose_df,
        radiomics_feature_cols=None,  # or a curated subset, as before
        bx_voxel_sextant_df = cohort_voxel_level_double_sextant_positions_filtered_df
    )
    """
    print(delta_design_df.columns.tolist())
        ['Voxel begin (Z)', 'Voxel end (Z)', 'Voxel index', 'Patient ID', 'Bx ID', 'Bx index', 'Dose (Gy) deltas nominal_minus_mean', 
        'Dose (Gy) deltas nominal_minus_mode', 'Dose (Gy) deltas nominal_minus_q50', 'Dose (Gy) abs deltas abs_nominal_minus_mean', 
        'Dose (Gy) abs deltas abs_nominal_minus_mode', 'Dose (Gy) abs deltas abs_nominal_minus_q50', 'Simulated bool', 'Simulated type', 'Length (mm)', 
        'Volume (mm3)', 'Voxel side length (mm)', 'Relative DIL ID', 'Relative DIL index', 'Relative prostate ID', 'Relative prostate index', 
        'Bx position in prostate LR', 'Bx position in prostate AP', 'Bx position in prostate SI', 'DIL Structure index', 'DIL Volume', 'DIL Surface area', 
        'DIL Surface area to volume ratio', 'DIL Sphericity', 'DIL Compactness 1', 'DIL Compactness 2', 'DIL Spherical disproportion', 'DIL Maximum 3D diameter', 
        'DIL PCA major', 'DIL PCA minor', 'DIL PCA least', 'DIL PCA eigenvector major', 'DIL PCA eigenvector minor', 'DIL PCA eigenvector least', 
        'DIL Major axis (equivalent ellipse)', 'DIL Minor axis (equivalent ellipse)', 'DIL Least axis (equivalent ellipse)', 'DIL Elongation', 'DIL Flatness', 
        'DIL L/R dimension at centroid', 'DIL A/P dimension at centroid', 'DIL S/I dimension at centroid', 'DIL S/I arclength', 'DIL DIL centroid (X, prostate frame)', 
        'DIL DIL centroid (Y, prostate frame)', 'DIL DIL centroid (Z, prostate frame)', 'DIL DIL centroid distance (prostate frame)', 'DIL DIL prostate sextant (LR)', 
        'DIL DIL prostate sextant (AP)', 'DIL DIL prostate sextant (SI)', 'Prostate Structure index', 'Prostate Volume', 'Prostate Surface area', 'Prostate Surface area to volume ratio', 
        'Prostate Sphericity', 'Prostate Compactness 1', 'Prostate Compactness 2', 'Prostate Spherical disproportion', 'Prostate Maximum 3D diameter', 'Prostate PCA major', 'Prostate PCA minor', 
        'Prostate PCA least', 'Prostate PCA eigenvector major', 'Prostate PCA eigenvector minor', 'Prostate PCA eigenvector least', 'Prostate Major axis (equivalent ellipse)', 
        'Prostate Minor axis (equivalent ellipse)', 'Prostate Least axis (equivalent ellipse)', 'Prostate Elongation', 'Prostate Flatness', 'Prostate L/R dimension at centroid', 
        'Prostate A/P dimension at centroid', 'Prostate S/I dimension at centroid', 'Prostate S/I arclength', 'Prostate DIL centroid (X, prostate frame)', 'Prostate DIL centroid (Y, prostate frame)', 
        'Prostate DIL centroid (Z, prostate frame)', 'Prostate DIL centroid distance (prostate frame)', 'Prostate DIL prostate sextant (LR)', 'Prostate DIL prostate sextant (AP)', 
        'Prostate DIL prostate sextant (SI)', 'DIL NN dist mean', 'DIL centroid dist mean', 'Prostate NN dist mean', 'Prostate centroid dist mean', 'Rectum NN dist mean', 
        'Rectum centroid dist mean', 'Urethra NN dist mean', 'Urethra centroid dist mean', 'Bx voxel prostate sextant (LR)', 'Bx voxel prostate sextant (AP)', 'Bx voxel prostate sextant (SI)', 
        'Prostate mean dimension at centroid', 'BX_to_prostate_centroid_distance_norm_mean_dim', 'DIL_centroid_distance_norm_mean_prostate_diameter', 'Nominal dose (Gy)', 'Nominal dose grad (Gy/mm)']
    """

    delta_design_csv = delta_corr_dir.joinpath(
        "deltas_00_per_voxel_deltas_and_predictors.csv"
    )
    delta_design_df.to_csv(delta_design_csv, index=False)



    # ------------------------------------------------------------------
    # Δ-bias vs predictors: build long design + correlations
    # ------------------------------------------------------------------

    # Optionally restrict to real biopsies only (if needed)
    # delta_design_df_for_corr = delta_design_df[delta_design_df["Simulated bool_x"] == False].copy()
    delta_design_df_for_corr = delta_design_df.copy()

    # 01) Long per-voxel design with Δ kind and log1p|Δ|
    delta_long_design = deltas_bias_analysis.make_long_delta_design_from_delta_design(
        delta_design_df_for_corr
    )

    """
    print(delta_long_design.columns)
    Index(['Voxel begin (Z)', 'Voxel end (Z)', 'Voxel index', 'Patient ID',
        'Bx ID', 'Bx index', 'Simulated bool', 'Simulated type',
        'Length (mm)', 'Volume (mm3)', 'Voxel side length (mm)',
        'Relative DIL ID', 'Relative DIL index', 'Relative prostate ID',
        'Relative prostate index', 'Bx position in prostate LR',
        'Bx position in prostate AP', 'Bx position in prostate SI',
        'DIL Volume', 'DIL Surface area', 'DIL Surface area to volume ratio',
        'DIL Sphericity', 'DIL Compactness 1', 'DIL Compactness 2',
        'DIL Spherical disproportion', 'DIL Maximum 3D diameter',
        'DIL PCA major', 'DIL PCA minor', 'DIL PCA least',
        'DIL PCA eigenvector major', 'DIL PCA eigenvector minor',
        'DIL PCA eigenvector least', 'DIL Major axis (equivalent ellipse)',
        'DIL Minor axis (equivalent ellipse)',
        'DIL Least axis (equivalent ellipse)', 'DIL Elongation', 'DIL Flatness',
        'DIL L/R dimension at centroid', 'DIL A/P dimension at centroid',
        'DIL S/I dimension at centroid', 'DIL S/I arclength',
        'DIL DIL centroid (X, prostate frame)',
        'DIL DIL centroid (Y, prostate frame)',
        'DIL DIL centroid (Z, prostate frame)',
        'DIL DIL centroid distance (prostate frame)',
        'DIL DIL prostate sextant (LR)', 'DIL DIL prostate sextant (AP)',
        'DIL DIL prostate sextant (SI)', 'Prostate Volume',
        'Prostate Surface area', 'Prostate Surface area to volume ratio',
        'Prostate Sphericity', 'Prostate Compactness 1',
        'Prostate Compactness 2', 'Prostate Spherical disproportion',
        'Prostate Maximum 3D diameter', 'Prostate PCA major',
        'Prostate PCA minor', 'Prostate PCA least',
        'Prostate PCA eigenvector major', 'Prostate PCA eigenvector minor',
        'Prostate PCA eigenvector least',
        'Prostate Major axis (equivalent ellipse)',
        'Prostate Minor axis (equivalent ellipse)',
        'Prostate Least axis (equivalent ellipse)', 'Prostate Elongation',
        'Prostate Flatness', 'Prostate L/R dimension at centroid',
        'Prostate A/P dimension at centroid',
        'Prostate S/I dimension at centroid', 'Prostate S/I arclength',
        'Prostate DIL centroid (X, prostate frame)',
        'Prostate DIL centroid (Y, prostate frame)',
        'Prostate DIL centroid (Z, prostate frame)',
        'Prostate DIL centroid distance (prostate frame)',
        'Prostate DIL prostate sextant (LR)',
        'Prostate DIL prostate sextant (AP)',
        'Prostate DIL prostate sextant (SI)', 'DIL NN dist mean',
        'DIL centroid dist mean', 'Prostate NN dist mean',
        'Prostate centroid dist mean', 'Rectum NN dist mean',
        'Rectum centroid dist mean', 'Urethra NN dist mean',
        'Urethra centroid dist mean', 'Bx voxel prostate sextant (LR)',
        'Bx voxel prostate sextant (AP)', 'Bx voxel prostate sextant (SI)',
        'Prostate mean dimension at centroid',
        'BX_to_prostate_centroid_distance_norm_mean_dim', 'Nominal dose (Gy)',
        'Nominal dose grad (Gy/mm)', 'Delta kind', 'Delta (Gy)', '|Delta|',
        'log1p|Delta|'],
        dtype='object')
    """

    delta_long_csv = delta_corr_dir.joinpath(
        "deltas_01_long_per_voxel_deltas_and_predictors.csv"
    )
    delta_long_design.to_csv(delta_long_csv, index=False)



    # 2a) Magnitude-focused (robust): log1p|Delta|
    delta_corr_maglog1p = deltas_bias_analysis.compute_delta_vs_predictor_correlation_generalized(
        design_df=delta_long_design,
        delta_col="log1p|Delta|",
        key_cols=("Patient ID", "Bx ID", "Bx index", "Voxel index"),
        group_col="Delta kind",
        min_n=30,
        rank_by = "pearson", 
        exclude_cols=[
            "Voxel begin (Z)", "Voxel end (Z)",
            "Prostate Structure index",
            "Prostate DIL centroid (X, prostate frame)",
            "Prostate DIL centroid (Y, prostate frame)",
            "Prostate DIL centroid (Z, prostate frame)",
            "Prostate DIL centroid distance (prostate frame)",
            "Prostate DIL prostate sextant (LR)",
            "Prostate DIL prostate sextant (AP)",
            "Prostate DIL prostate sextant (SI)",
            "DIL PCA eigenvector major",
            "DIL PCA eigenvector minor",
            "DIL PCA eigenvector least",
            "Simulated bool", "Simulated type",
            "Volume (mm3)", "Voxel side length (mm)",
            "Relative DIL ID", "Relative DIL index",
            "Relative prostate ID", "Relative prostate index",
            "Bx position in prostate LR",
            "Bx position in prostate AP",
            "Bx position in prostate SI",
            "DIL DIL prostate sextant (LR)",
            "DIL DIL prostate sextant (AP)",
            "DIL DIL prostate sextant (SI)",
        ],
    )
    """
    print(delta_corr_maglog1p.columns)
    Index(['Delta kind', 'predictor', 'n', 'r', 'p_value'], dtype='object')
    """

    delta_corr_maglog1p.to_csv(
        delta_corr_dir.joinpath(
            "deltas_02a_correlations_log1pDelta_by_delta_kind_and_predictor.csv"
        ),
        index=False,
    )


    # 2b) Directional: signed Delta (Gy)
    delta_corr_signed = deltas_bias_analysis.compute_delta_vs_predictor_correlation_generalized(
        design_df=delta_long_design,
        delta_col="Delta (Gy)",
        key_cols=("Patient ID", "Bx ID", "Bx index", "Voxel index"),
        group_col="Delta kind",
        min_n=30,
        rank_by = "pearson",
        exclude_cols=[
            "Voxel begin (Z)", "Voxel end (Z)",
            "Prostate Structure index",
            "Prostate DIL centroid (X, prostate frame)",
            "Prostate DIL centroid (Y, prostate frame)",
            "Prostate DIL centroid (Z, prostate frame)",
            "Prostate DIL centroid distance (prostate frame)",
            "Prostate DIL prostate sextant (LR)",
            "Prostate DIL prostate sextant (AP)",
            "Prostate DIL prostate sextant (SI)",
            "DIL PCA eigenvector major",
            "DIL PCA eigenvector minor",
            "DIL PCA eigenvector least",
            "Simulated bool", "Simulated type",
            "Volume (mm3)", "Voxel side length (mm)",
            "Relative DIL ID", "Relative DIL index",
            "Relative prostate ID", "Relative prostate index",
            "Bx position in prostate LR",
            "Bx position in prostate AP",
            "Bx position in prostate SI",
            "DIL DIL prostate sextant (LR)",
            "DIL DIL prostate sextant (AP)",
            "DIL DIL prostate sextant (SI)",
        ],
    )
    """
    print(delta_corr_signed.columns)
    Index(['Delta kind', 'predictor', 'n', 'r', 'p_value'], dtype='object')
    """

    delta_corr_signed.to_csv(
        delta_corr_dir.joinpath(
            "deltas_02b_correlations_signedDelta_by_delta_kind_and_predictor.csv"
        ),
        index=False,
    )


    # 2c) Magnitude-focused (robust): log1p|Delta|
    delta_corr_mag = deltas_bias_analysis.compute_delta_vs_predictor_correlation_generalized(
        design_df=delta_long_design,
        delta_col="|Delta|",
        key_cols=("Patient ID", "Bx ID", "Bx index", "Voxel index"),
        group_col="Delta kind",
        min_n=30,
        rank_by = "pearson",
        exclude_cols=[
            "Voxel begin (Z)", "Voxel end (Z)",
            "Prostate Structure index",
            "Prostate DIL centroid (X, prostate frame)",
            "Prostate DIL centroid (Y, prostate frame)",
            "Prostate DIL centroid (Z, prostate frame)",
            "Prostate DIL centroid distance (prostate frame)",
            "Prostate DIL prostate sextant (LR)",
            "Prostate DIL prostate sextant (AP)",
            "Prostate DIL prostate sextant (SI)",
            "DIL PCA eigenvector major",
            "DIL PCA eigenvector minor",
            "DIL PCA eigenvector least",
            "Simulated bool", "Simulated type",
            "Volume (mm3)", "Voxel side length (mm)",
            "Relative DIL ID", "Relative DIL index",
            "Relative prostate ID", "Relative prostate index",
            "Bx position in prostate LR",
            "Bx position in prostate AP",
            "Bx position in prostate SI",
            "DIL DIL prostate sextant (LR)",
            "DIL DIL prostate sextant (AP)",
            "DIL DIL prostate sextant (SI)",
        ],
    )
    """
    print(delta_corr_mag.columns)
    Index(['Delta kind', 'predictor', 'n', 'r', 'p_value'], dtype='object')
    """

    delta_corr_mag.to_csv(
        delta_corr_dir.joinpath(
            "deltas_02c_correlations_abs_Delta_by_delta_kind_and_predictor.csv"
        ),
        index=False,
    )



    # Choose your four predictors:
    #  - Nominal dose
    #  - Nominal dose gradient
    #  - Top distance-based predictor
    #  - Top radiomic predictor
    predictor_cols = [
        "Nominal dose (Gy)",                                        # highest correlated in all three kinds of delta
        "Nominal dose grad (Gy/mm)",                                # second highest correlated in all three kinds of delta
        "BX_to_prostate_centroid_distance_norm_mean_dim",          # biopsy to prostate centroid distance normalized by the prostate mean diameter (x,y,z) was third highest in all three kinds of delta
        "DIL Flatness",                                               # across all three kinds of delta, flatness was the most strongly correlated radiomic predictor
    ]

    # Optional: override labels for the distance/radiomic panels with LaTeX that matches the paper
    predictor_label_map = {
        # Nominal voxel dose: D_{b,v}^{(0)} (Gy)
        "Nominal dose (Gy)": r"$D_{b,v}^{(0)}\ \mathrm{(Gy)}$",

        # Nominal dose gradient magnitude: ||∇D||_{nom} (Gy mm^{-1})
        "Nominal dose grad (Gy/mm)": r"$G_{b,v}^{(0)}\ \mathrm{(Gy\ mm^{-1})}$",

        # Distance-based predictor normalized distance so its dimensionless
        #"DIL NN dist mean": r"$\overline{d}^{\mathrm{NN}}_{\mathrm{DIL}}~\mathrm{(mm)}$",
        "BX_to_prostate_centroid_distance_norm_mean_dim": r"$\overline{d}^{\mathrm{norm,cen}}_{\mathrm{P},v}$",

        # Radiomic predictor
        "DIL Flatness": r"$\mathrm{Flatness}_{\mathrm{DIL}}$",
    }

    delta_kind_label_map = {
        "Δ_median": r"$j = Q_{50}$",
        "Δ_mean": r"$j = mean$",
        "Δ_mode": r"$j = mode$",
    }

    """
    svg_path_log1p, png_path_log1p, stats_csv_log1p, stats_df_log1p = production_plots.plot_log1p_median_delta_vs_predictors_pkg(
        delta_long_design,
        save_dir=delta_corr_dir,
        file_prefix="03c_log1p_medianDelta_vs_top4_predictors",
        predictor_cols=predictor_cols,
        predictor_label_map=predictor_label_map,
        axes_label_fontsize=13,
        tick_label_fontsize=11,
        legend_fontsize=11,
        height=3.0,
        aspect=1.4,
        facet_cols=2,
        label_style="latex",
        j_symbol="Q50",
        scatter=True,
        scatter_sample=20000,
        scatter_alpha=0.15,
        scatter_size=10.0,
        annotate_stats=True,
        write_stats_csv=True,
        title=None,  # or set a LaTeX-style title if you want
    )
    """



    """
    svg, png, stats_csv_med, stats_df = production_plots.plot_delta_vs_predictors_pkg(
        delta_long_design,
        save_dir=delta_corr_dir,
        file_prefix="03c_abs_medianDelta_vs_top4_predictors",
        predictor_cols = predictor_cols,
        predictor_label_map=predictor_label_map,
        # NEW: general y control
        y_col = "|Delta|",
        delta_kind_label = "Δ_median",     # e.g. "Δ_median", "Δ_mean", "Δ_mode", or None for no filter
        delta_kind_col = "Delta kind",
        # scatter / regression options
        axes_label_fontsize=13,
        tick_label_fontsize=11,
        legend_fontsize=11,
        height=3.0,
        aspect=1.4,
        facet_cols=2,
        label_style="latex",
        idx_sub = ("b", "v"),
        j_symbol="Q50",
        scatter=True,
        scatter_sample=20000,
        scatter_alpha=0.15,
        scatter_size=10.0,
        annotate_stats=True,
        write_stats_csv=True,
        title=None
    )
    """ 

    # for setting the x limit min value for the plot to 0 for these specific predictors
    zero_x_predictors = {
        "Nominal dose (Gy)",
        "Nominal dose grad (Gy/mm)",
        # maybe distance, maybe not
    }

    svg_abs_delta, png_abs_delta, stats_csv_med_abs_delta, stats_df_abs_delta = production_plots.plot_delta_vs_predictors_pkg_generalized(
        delta_long_design,
        save_dir=delta_corr_dir,
        file_prefix="03c_abs_medianDelta_vs_top4_predictors",
        predictor_cols = predictor_cols,
        predictor_label_map=predictor_label_map,
        # NEW: general y control
        y_col = "|Delta|",
        delta_kind_label = ["Δ_median", "Δ_mean", "Δ_mode"],     # e.g. "Δ_median", "Δ_mean", "Δ_mode", or None for no filter
        delta_kind_col = "Delta kind",
        # scatter / regression options
        axes_label_fontsize=13,
        tick_label_fontsize=11,
        legend_fontsize=11,
        height=3.0,
        aspect=1.4,
        facet_cols=2,
        label_style="latex",
        idx_sub = ("b", "v"),
        j_symbol="(j)",
        scatter=True,
        scatter_sample=20000,
        scatter_alpha=0.3,
        scatter_size=10.0,
        annotate_stats=False,
        write_stats_csv=True,
        title=None,
        delta_kind_label_map = delta_kind_label_map,
        zero_x_predictors = zero_x_predictors
    )





















    # Path where you want to store sextant summaries
    sextant_summary_dir = delta_corr_dir / "sextant_summaries"
    sextant_summary_dir.mkdir(parents=True, exist_ok=True)

    sextant_cols = (
        "Bx voxel prostate sextant (LR)",
        "Bx voxel prostate sextant (AP)",
        "Bx voxel prostate sextant (SI)",
    )

    # 1) Dose and gradient across MC trials by voxel sextant
    sextant_mc_dose_grad = deltas_bias_analysis.summarize_mc_dose_grad_by_sextant(
        all_voxel_wise_dose_df=all_voxel_wise_dose_df,
        sextant_df=cohort_voxel_level_double_sextant_positions_filtered_df,
        sextant_cols=sextant_cols,
    )
    """
    print(sextant_mc_dose_grad.columns)
    Index(['Bx voxel prostate sextant (LR)', 'Bx voxel prostate sextant (AP)',
        'Bx voxel prostate sextant (SI)', 'Dose (Gy) count', 'Dose (Gy) mean',
        'Dose (Gy) std', 'Dose (Gy) median', 'Dose (Gy) q05', 'Dose (Gy) q25',
        'Dose (Gy) q75', 'Dose (Gy) q95', 'Dose (Gy) IQR', 'Dose (Gy) IPR90',
        'Dose grad (Gy/mm) count', 'Dose grad (Gy/mm) mean',
        'Dose grad (Gy/mm) std', 'Dose grad (Gy/mm) median',
        'Dose grad (Gy/mm) q05', 'Dose grad (Gy/mm) q25',
        'Dose grad (Gy/mm) q75', 'Dose grad (Gy/mm) q95',
        'Dose grad (Gy/mm) IQR', 'Dose grad (Gy/mm) IPR90', 'n_voxels',
        'n_trials', 'n_datapoints'],
        dtype='object')
    """
    sextant_mc_dose_grad.to_csv(
        sextant_summary_dir / "sextant_mc_dose_grad_summary.csv",
        index=False,
    )

    # 2) Trial wise deltas by voxel sextant
    sextant_mc_deltas = deltas_bias_analysis.summarize_mc_deltas_by_sextant(
        mc_deltas=mc_deltas,
        sextant_df=cohort_voxel_level_double_sextant_positions_filtered_df,
        sextant_cols=sextant_cols,
    )
    """
    print(sextant_mc_deltas.columns)
    Index(['Bx voxel prostate sextant (LR)', 'Bx voxel prostate sextant (AP)',
        'Bx voxel prostate sextant (SI)',
        'Dose (Gy) deltas nominal_minus_trial count',
        'Dose (Gy) deltas nominal_minus_trial mean',
        'Dose (Gy) deltas nominal_minus_trial std',
        'Dose (Gy) deltas nominal_minus_trial median',
        'Dose (Gy) deltas nominal_minus_trial q05',
        'Dose (Gy) deltas nominal_minus_trial q25',
        'Dose (Gy) deltas nominal_minus_trial q75',
        'Dose (Gy) deltas nominal_minus_trial q95',
        'Dose (Gy) deltas nominal_minus_trial IQR',
        'Dose (Gy) deltas nominal_minus_trial IPR90',
        'Dose (Gy) abs deltas abs_nominal_minus_trial count',
        'Dose (Gy) abs deltas abs_nominal_minus_trial mean',
        'Dose (Gy) abs deltas abs_nominal_minus_trial std',
        'Dose (Gy) abs deltas abs_nominal_minus_trial median',
        'Dose (Gy) abs deltas abs_nominal_minus_trial q05',
        'Dose (Gy) abs deltas abs_nominal_minus_trial q25',
        'Dose (Gy) abs deltas abs_nominal_minus_trial q75',
        'Dose (Gy) abs deltas abs_nominal_minus_trial q95',
        'Dose (Gy) abs deltas abs_nominal_minus_trial IQR',
        'Dose (Gy) abs deltas abs_nominal_minus_trial IPR90',
        'Dose grad (Gy/mm) deltas nominal_minus_trial count',
        'Dose grad (Gy/mm) deltas nominal_minus_trial mean',
        'Dose grad (Gy/mm) deltas nominal_minus_trial std',
        'Dose grad (Gy/mm) deltas nominal_minus_trial median',
        'Dose grad (Gy/mm) deltas nominal_minus_trial q05',
        'Dose grad (Gy/mm) deltas nominal_minus_trial q25',
        'Dose grad (Gy/mm) deltas nominal_minus_trial q75',
        'Dose grad (Gy/mm) deltas nominal_minus_trial q95',
        'Dose grad (Gy/mm) deltas nominal_minus_trial IQR',
        'Dose grad (Gy/mm) deltas nominal_minus_trial IPR90',
        'Dose grad (Gy/mm) abs deltas abs_nominal_minus_trial count',
        'Dose grad (Gy/mm) abs deltas abs_nominal_minus_trial mean',
        'Dose grad (Gy/mm) abs deltas abs_nominal_minus_trial std',
        'Dose grad (Gy/mm) abs deltas abs_nominal_minus_trial median',
        'Dose grad (Gy/mm) abs deltas abs_nominal_minus_trial q05',
        'Dose grad (Gy/mm) abs deltas abs_nominal_minus_trial q25',
        'Dose grad (Gy/mm) abs deltas abs_nominal_minus_trial q75',
        'Dose grad (Gy/mm) abs deltas abs_nominal_minus_trial q95',
        'Dose grad (Gy/mm) abs deltas abs_nominal_minus_trial IQR',
        'Dose grad (Gy/mm) abs deltas abs_nominal_minus_trial IPR90',
        'n_voxels', 'n_trials', 'n_datapoints'],
        dtype='object')
    """
    sextant_mc_deltas.to_csv(
        sextant_summary_dir / "sextant_mc_trial_deltas_summary.csv",
        index=False,
    )

    # 3) Nominal summary deltas (bias) by voxel sextant
    sextant_nominal_deltas = deltas_bias_analysis.summarize_nominal_deltas_by_sextant(
        nominal_deltas_df_with_abs=nominal_deltas_df_with_abs,
        sextant_df=cohort_voxel_level_double_sextant_positions_filtered_df,
        sextant_cols=sextant_cols,
    )
    """
    print(sextant_nominal_deltas.columns)
    Index(['Bx voxel prostate sextant (LR)', 'Bx voxel prostate sextant (AP)',
        'Bx voxel prostate sextant (SI)',
        'Dose (Gy) deltas nominal_minus_mean count',
        'Dose (Gy) deltas nominal_minus_mean mean',
        'Dose (Gy) deltas nominal_minus_mean std',
        'Dose (Gy) deltas nominal_minus_mean median',
        'Dose (Gy) deltas nominal_minus_mean q05',
        'Dose (Gy) deltas nominal_minus_mean q25',
        'Dose (Gy) deltas nominal_minus_mean q75',
        'Dose (Gy) deltas nominal_minus_mean q95',
        'Dose (Gy) deltas nominal_minus_mean IQR',
        'Dose (Gy) deltas nominal_minus_mean IPR90',
        'Dose (Gy) deltas nominal_minus_mode count',
        'Dose (Gy) deltas nominal_minus_mode mean',
        'Dose (Gy) deltas nominal_minus_mode std',
        'Dose (Gy) deltas nominal_minus_mode median',
        'Dose (Gy) deltas nominal_minus_mode q05',
        'Dose (Gy) deltas nominal_minus_mode q25',
        'Dose (Gy) deltas nominal_minus_mode q75',
        'Dose (Gy) deltas nominal_minus_mode q95',
        'Dose (Gy) deltas nominal_minus_mode IQR',
        'Dose (Gy) deltas nominal_minus_mode IPR90',
        'Dose (Gy) deltas nominal_minus_q50 count',
        'Dose (Gy) deltas nominal_minus_q50 mean',
        'Dose (Gy) deltas nominal_minus_q50 std',
        'Dose (Gy) deltas nominal_minus_q50 median',
        'Dose (Gy) deltas nominal_minus_q50 q05',
        'Dose (Gy) deltas nominal_minus_q50 q25',
        'Dose (Gy) deltas nominal_minus_q50 q75',
        'Dose (Gy) deltas nominal_minus_q50 q95',
        'Dose (Gy) deltas nominal_minus_q50 IQR',
        'Dose (Gy) deltas nominal_minus_q50 IPR90',
        'Dose (Gy) abs deltas abs_nominal_minus_mean count',
        'Dose (Gy) abs deltas abs_nominal_minus_mean mean',
        'Dose (Gy) abs deltas abs_nominal_minus_mean std',
        'Dose (Gy) abs deltas abs_nominal_minus_mean median',
        'Dose (Gy) abs deltas abs_nominal_minus_mean q05',
        'Dose (Gy) abs deltas abs_nominal_minus_mean q25',
        'Dose (Gy) abs deltas abs_nominal_minus_mean q75',
        'Dose (Gy) abs deltas abs_nominal_minus_mean q95',
        'Dose (Gy) abs deltas abs_nominal_minus_mean IQR',
        'Dose (Gy) abs deltas abs_nominal_minus_mean IPR90',
        'Dose (Gy) abs deltas abs_nominal_minus_mode count',
        'Dose (Gy) abs deltas abs_nominal_minus_mode mean',
        'Dose (Gy) abs deltas abs_nominal_minus_mode std',
        'Dose (Gy) abs deltas abs_nominal_minus_mode median',
        'Dose (Gy) abs deltas abs_nominal_minus_mode q05',
        'Dose (Gy) abs deltas abs_nominal_minus_mode q25',
        'Dose (Gy) abs deltas abs_nominal_minus_mode q75',
        'Dose (Gy) abs deltas abs_nominal_minus_mode q95',
        'Dose (Gy) abs deltas abs_nominal_minus_mode IQR',
        'Dose (Gy) abs deltas abs_nominal_minus_mode IPR90',
        'Dose (Gy) abs deltas abs_nominal_minus_q50 count',
        'Dose (Gy) abs deltas abs_nominal_minus_q50 mean',
        'Dose (Gy) abs deltas abs_nominal_minus_q50 std',
        'Dose (Gy) abs deltas abs_nominal_minus_q50 median',
        'Dose (Gy) abs deltas abs_nominal_minus_q50 q05',
        'Dose (Gy) abs deltas abs_nominal_minus_q50 q25',
        'Dose (Gy) abs deltas abs_nominal_minus_q50 q75',
        'Dose (Gy) abs deltas abs_nominal_minus_q50 q95',
        'Dose (Gy) abs deltas abs_nominal_minus_q50 IQR',
        'Dose (Gy) abs deltas abs_nominal_minus_q50 IPR90', 'n_voxels',
        'n_datapoints'],
        dtype='object')
    """
    sextant_nominal_deltas.to_csv(
        sextant_summary_dir / "sextant_nominal_bias_deltas_summary.csv",
        index=False,
    )



    sextant_fig_dir = delta_corr_dir / "sextant_figures"
    sextant_fig_dir.mkdir(parents=True, exist_ok=True)

    production_plots.plot_sextant_dose_bias_panels(
        sextant_mc_dose_grad=sextant_mc_dose_grad,
        sextant_mc_deltas=sextant_mc_deltas,
        sextant_nominal_deltas=sextant_nominal_deltas,
        output_dir=sextant_fig_dir,
        fig_scale=1.0,
        dpi = 300,
        show = False,
        fig_name = "Fig_sextant_dose_bias_by_double_sextant",
        # font controls
        title_fontsize = 16,
        cell_fontsize = 8,
        ytick_fontsize = 12,
        cbar_label_fontsize  = 16,
        cbar_tick_fontsize = 14,
    )



    























    # Uses precomputed abs columns; will raise if they’re missing
    csv_path = summary_statistics.save_delta_boxplot_summary_csv_with_absolute_no_recalc(
        nominal_deltas_df_with_abs,                      # <-- use the _with_abs version
        output_dir=voxel_wise_nominal_analysis_dir,
        csv_name='dose_deltas_boxplot_summary.csv',
        zero_level_index_str='Dose (Gy)',
        include_patient_ids=None,   # or ['184','201']
        decimals=3,
        require_precomputed_abs=True,     # default
    )

    csv_path = summary_statistics.save_delta_boxplot_summary_csv_with_absolute_no_recalc(
        nominal_gradient_deltas_df_with_abs,                      # <-- use the _with_abs version
        output_dir=voxel_wise_nominal_analysis_dir,
        csv_name='dose_gradient_deltas_boxplot_summary.csv',
        zero_level_index_str='Dose grad (Gy/mm)',
        include_patient_ids=None,   # or ['184','201']
        decimals=3,
        require_precomputed_abs=True,     # default
    )
    print(csv_path)





    # build deltas versus gradient dataframe
    """
    combined_deltas_plus_gradient_vals_wide, combined_deltas_plus_gradient_vals_long = helper_funcs.build_deltas_vs_gradient_df(
        nominal_deltas_df=nominal_deltas_df,
        cohort_by_voxel_df=cohort_global_dosimetry_by_voxel_df,
        gradient_top='Dose grad (Gy/mm)',   # default
        gradient_stat='nominal',            # or 'mean', 'quantile_50', etc., if you prefer
        return_long=True
    )
    """
    
    combined_wide_deltas_vs_gradient, combined_long_deltas_vs_gradient = helper_funcs.build_deltas_vs_gradient_df_with_abs(
        nominal_deltas_df=nominal_deltas_df_with_abs,
        cohort_by_voxel_df=cohort_global_dosimetry_by_voxel_df,
        zero_level_index_str='Dose (Gy)',
        gradient_top='Dose grad (Gy/mm)',
        gradient_stats=('nominal', 'median', 'mean', 'mode'),  # multiple stats
        gradient_stat=None,            # alias (ignored since gradient_stats provided)
        meta_keep=(
            'Voxel begin (Z)', 'Voxel end (Z)', 'Voxel index',
            'Patient ID', 'Bx ID', 'Bx index',
            'Simulated bool', 'Simulated type'
        ),
        add_abs=True,                  # include precomputed |Δ|
        add_log1p=True,                # add log1p(|Δ|) columns
        return_long=True,              # also return tidy long df
        require_precomputed_abs=True,  # expect abs block; do not recompute
        fallback_recompute_abs=False   # set True only if you want on-the-fly |Δ|
    )
    # long:
    """
    print(combined_long_deltas_vs_gradient.columns) = 
    Index(['Patient ID', 'Bx index', 'Voxel index', 'Grad[nominal] (Gy/mm)',
        'Grad[median] (Gy/mm)', 'Grad[mean] (Gy/mm)', 'Grad[mode] (Gy/mm)',
        'Delta kind', 'Delta (signed)', '|Delta|', 'log1p|Delta|'],
        dtype='object')
    """
    # wide: 
    """
    print(combined_wide_deltas_vs_gradient.columns) =
    Index(['Voxel begin (Z)_x', 'Voxel end (Z)_x', 'Voxel index', 'Patient ID',
        'Bx ID_x', 'Bx index', 'Simulated bool_x',
        'Simulated type_x', 'Δ_mode (Gy)', 'Δ_median (Gy)', 'Δ_mean (Gy)',
        '|Δ_mode| (Gy)', '|Δ_median| (Gy)', '|Δ_mean| (Gy)',
        'Voxel begin (Z)_y', 'Voxel end (Z)_y', 'Bx ID_y', 
        'Simulated bool_y', 'Simulated type_y', 'Grad[nominal] (Gy/mm)',
        'Grad[median] (Gy/mm)', 'Grad[mean] (Gy/mm)', 'Grad[mode] (Gy/mm)',
        'log1p|Δ_mode| (Gy)', 'log1p|Δ_median| (Gy)', 'log1p|Δ_mean| (Gy)'],
        dtype='object')
    """


    ### compute delta bias correlations
    deltas_bias_corr_signed = deltas_bias_analysis.compute_interdelta_correlations(
        combined_wide_deltas_vs_gradient,
        delta_cols=["Δ_mode (Gy)", "Δ_median (Gy)", "Δ_mean (Gy)"],
    )

    deltas_bias_corr_abs = deltas_bias_analysis.compute_interdelta_correlations(
        combined_wide_deltas_vs_gradient,
        delta_cols=["|Δ_mode| (Gy)", "|Δ_median| (Gy)", "|Δ_mean| (Gy)"],
    )

    deltas_bias_corr_log_abs = deltas_bias_analysis.compute_interdelta_correlations(
        combined_wide_deltas_vs_gradient,
        delta_cols=[
            "log1p|Δ_mode| (Gy)",
            "log1p|Δ_median| (Gy)",
            "log1p|Δ_mean| (Gy)",
        ],
    )

    ## Save CSVs
    delta_bias_corr_signed_csv = delta_corr_dir.joinpath(
        "deltas_04a_deltas_biased_correlations_signed.csv"
    )
    deltas_bias_corr_signed.to_csv(delta_bias_corr_signed_csv, index=False)

    delta_bias_corr_abs_csv = delta_corr_dir.joinpath(
        "deltas_04b_deltas_biased_correlations_abs.csv"
    )
    deltas_bias_corr_abs.to_csv(delta_bias_corr_abs_csv, index=False)

    delta_bias_corr_log_abs_csv = delta_corr_dir.joinpath(
        "deltas_04c_deltas_biased_correlations_log_abs.csv"
    )
    deltas_bias_corr_log_abs.to_csv(delta_bias_corr_log_abs_csv, index=False)














    #### plot deltas vs grad :
    # 1) Signed & |Δ| together — separate trends by Measure (never mixed),
    #    color by Measure (clearer), plus optional LOESS overlay.
    _ = production_plots.plot_delta_vs_gradient(
        combined_long_deltas_vs_gradient,
        save_dir=cohort_output_figures_dir,
        fig_name="delta_vs_gradient_signed_and_abs",
        gradient_cols=None,                         # auto
        delta_kinds=("Δ_mode","Δ_median","Δ_mean"),
        y_variant="both",
        use_log1p_abs=True,                         # avoids tail squash
        show_scatter=True,
        scatter_sample=20000,
        bins=24, binning="quantile", min_per_bin=20,
        show_iqr_band=True, show_90_band=True,
        hue_by="Measure",                           # <<< key change
        trend_split="Measure",                      # <<< never mix Signed with Absolute
        regression="none",                          # or 'ols' / 'loess' if statsmodels is installed
        axes_label_fontsize=14, tick_label_fontsize=12, legend_fontsize=12,
        facet_cols=2, height=3.0, aspect=1.5,
        title="Δ vs Dose Gradient (Signed vs |Δ|)"
    )

    # 2) Absolute only — by Δ kind — ensure trends have enough points; add scatter or relax min_per_bin.
    grad_nom = [c for c in combined_long_deltas_vs_gradient.columns if c.startswith("Grad[nominal]")]
    _ = production_plots.plot_delta_vs_gradient(
        combined_long_deltas_vs_gradient,
        save_dir=cohort_output_figures_dir,
        fig_name="abs_delta_vs_grad_nominal",
        gradient_cols=grad_nom,
        delta_kinds=("Δ_mode","Δ_median","Δ_mean"),
        y_variant="abs",
        use_log1p_abs=False,
        hue_by="Delta kind",
        trend_split="Delta kind",
        show_scatter=True,               # turn on scatter to see points even if bins are thin
        bins=20, binning="quantile", min_per_bin=10,   # relax for per-kind splits
        regression="ols", poly_order=1,  # optional linear overlay
        label_style="math",
        title="|Δ| vs Grad[nominal]"
    )



    """
    # 1) Magnitude vs gradient (main figure, linear scale)
    _, _, stats_csv1, stats_df1 = production_plots.plot_abs_delta_vs_gradient_pkg(
        combined_long_deltas_vs_gradient,
        save_dir=cohort_output_figures_dir,
        file_prefix="abs_delta_vs_gradient",
        gradient_cols=None,                        # auto-detect all Grad[⋯]
        delta_kinds=("Δ_mode","Δ_median","Δ_mean"),
        use_log1p=False,                           # <- main text result
        scatter=True, scatter_sample=20000,
        ci=95, annotate_stats=True, write_stats_csv=True,
        title="|Δ| vs Dose Gradient",
        axes_label_fontsize=14, tick_label_fontsize=12, legend_fontsize=12,
        facet_cols=2, height=3.0, aspect=1.5
    )

    # 2) Same on log1p scale (supplement, optional)
    _, _, stats_csv2, _ =  production_plots.plot_abs_delta_vs_gradient_pkg(
        combined_long_deltas_vs_gradient,
        save_dir=cohort_output_figures_dir,
        file_prefix="abs_delta_vs_gradient_log1p",
        gradient_cols=["Grad[nominal] (Gy/mm)"],
        delta_kinds=("Δ_mode","Δ_median","Δ_mean"),
        use_log1p=True,                            # <- supplemental view
        scatter=True, scatter_sample=20000,
        ci=95, annotate_stats=True, write_stats_csv=True,
        title="log(1+|Δ|) vs Grad[nominal]",
        axes_label_fontsize=14, tick_label_fontsize=12, legend_fontsize=12,
        facet_cols=1, height=3.0, aspect=1.6
    )

    # 3) Bias vs gradient (signed), single key gradient
    _, _, stats_csv3, stats_df3 =  production_plots.plot_signed_delta_vs_gradient_pkg(
        combined_long_deltas_vs_gradient,
        save_dir=cohort_output_figures_dir,
        file_prefix="signed_delta_vs_grad_nominal",
        gradient_cols=["Grad[nominal] (Gy/mm)"],
        delta_kinds=("Δ_mode","Δ_median","Δ_mean"),
        scatter=True, scatter_sample=20000,
        ci=95, annotate_stats=True, write_stats_csv=True,
        title="Signed Δ vs Grad[nominal]",
        axes_label_fontsize=14, tick_label_fontsize=12, legend_fontsize=12,
        facet_cols=1, height=3.0, aspect=1.6
    )
    """

    """
    # 1) Magnitude: |Δ| vs gradient (facets over all Grad[⋯] columns)
    abs_svg, abs_png, abs_stats_csv, abs_stats_df = production_plots.plot_abs_delta_vs_gradient_pkg(
        long_df=combined_long_deltas_vs_gradient,
        save_dir=cohort_output_figures_dir,
        file_prefix="abs_delta_vs_gradient",
        # --- options ---
        gradient_cols=None,                              # auto-detect all columns starting with "Grad["
        delta_kinds=("Δ_mode", "Δ_median", "Δ_mean"),
        use_log1p=False,                                 # set True only for supplemental view
        scatter=True,
        scatter_sample=20000,
        scatter_alpha=0.15,
        scatter_size=10.0,
        ci=95,                                           # 95% CI for OLS bands
        annotate_stats=True,                             # add slope±CI, Spearman ρ, R² panel text
        write_stats_csv=True,                            # save <file_prefix>__stats.csv
        axes_label_fontsize=14,
        tick_label_fontsize=12,
        legend_fontsize=12,
        height=3.0,
        aspect=1.5,
        facet_cols=2,
        title="|Δ| vs Dose Gradient"
    )

    # 2) Bias: signed Δ vs gradient (single key gradient panel)
    grad_nom = ["Grad[nominal] (Gy/mm)"]
    signed_svg, signed_png, signed_stats_csv, signed_stats_df = production_plots.plot_signed_delta_vs_gradient_pkg(
        long_df=combined_long_deltas_vs_gradient,
        save_dir=cohort_output_figures_dir,
        file_prefix="signed_delta_vs_grad_nominal",
        # --- options ---
        gradient_cols=grad_nom,                          # or None to facet all gradient stats
        delta_kinds=("Δ_mode", "Δ_median", "Δ_mean"),
        scatter=True,
        scatter_sample=20000,
        scatter_alpha=0.15,
        scatter_size=10.0,
        ci=95,
        annotate_stats=True,
        write_stats_csv=True,
        axes_label_fontsize=14,
        tick_label_fontsize=12,
        legend_fontsize=12,
        height=3.0,
        aspect=1.6,
        facet_cols=1,
        title="Signed Δ vs Grad[nominal]"
    )
    """


    # 3) Batch mode: all gradients in one figure with subpanels
    gradients = [
        "Grad[nominal] (Gy/mm)",
        "Grad[median] (Gy/mm)",
        "Grad[mean] (Gy/mm)",
        "Grad[mode] (Gy/mm)",
    ]

    # Common label options:
    label_style = "latex"     # "latex" → mathtext (Δ^{mode}_{b,v}, Gy mm^{-1}); use "plain" for no math
    idx_sub     = ("b","v")   # the indices under Δ
    j_symbol    = "j"         # the superscript on Δ

    # ABSOLUTE batch (|Δ|)
    abs_svgs, abs_pngs, abs_combined_stats_csv = production_plots.plot_abs_delta_vs_gradient_pkg_batch(
        long_df=combined_long_deltas_vs_gradient,
        save_dir=cohort_output_figures_dir,
        base_prefix="abs_delta_vs_gradient",
        gradient_cols=gradients,                   # or None for all Grad[⋯]
        delta_kinds=("Δ_mode","Δ_median","Δ_mean"),
        use_log1p=False,                           # set True for a supplemental view
        # visuals & export
        scatter=True, scatter_sample=20000, scatter_alpha=0.15, scatter_size=10.0,
        ci=95, annotate_stats=False, write_stats_csv=True,
        axes_label_fontsize=14, tick_label_fontsize=12, legend_fontsize=12,
        height_single=3.0, aspect_single=1.6,
        height_combined=3.0, aspect_combined=1.5, facet_cols_combined=2,
        # <<— label knobs
        label_style=label_style, idx_sub=idx_sub, j_symbol=j_symbol,
        grad_stat_tex=None,  # or e.g. {"nominal": r"\mathrm{nom}"}
    )

    # SIGNED batch (Δ)
    signed_svgs, signed_pngs, signed_combined_stats_csv = production_plots.plot_signed_delta_vs_gradient_pkg_batch(
        long_df=combined_long_deltas_vs_gradient,
        save_dir=cohort_output_figures_dir,
        base_prefix="signed_delta_vs_gradient",
        gradient_cols=gradients,                   # pass ["Grad[nominal] (Gy/mm)"] for 1 panel
        delta_kinds=("Δ_mode","Δ_median","Δ_mean"),
        # visuals & export
        scatter=True, scatter_sample=20000, scatter_alpha=0.15, scatter_size=10.0,
        ci=95, annotate_stats=False, write_stats_csv=True,
        axes_label_fontsize=14, tick_label_fontsize=12, legend_fontsize=12,
        height_single=3.0, aspect_single=1.6,
        height_combined=3.0, aspect_combined=1.5, facet_cols_combined=2,
        # <<— label knobs
        label_style=label_style, idx_sub=idx_sub, j_symbol=j_symbol,
        grad_stat_tex=None,
    )










    print('test')



    # Plot deltas plots 



    









    ### Global dosimetry analysis (START)
    # Create output directory for global dosimetry
    global_dosimetry_dir = output_dir.joinpath("global_dosimetry")
    os.makedirs(global_dosimetry_dir, exist_ok=True)
    # Output filename
    output_filename = 'global_dosimetry_statistics_all_patients.csv'
    """
    col_pairs_to_summarize = [
    ('Dose (Gy)', 'argmax_density'),
    ('Dose (Gy)', 'min'),
    ('Dose (Gy)', 'mean'),
    ('Dose (Gy)', 'max'),
    ('Dose (Gy)', 'nominal (spatial average)')
    ('Dose grad (Gy/mm)', 'argmax_density')
    ]
    """
    exclude_for_all = [
    ('Bx ID',''),
    ('Patient ID',''),
    ('Bx index',''),
    ('Simulated bool',''),
    ('Simulated type',''),
    ('Bx refnum','')]


    ### simply save the global dosimetry by biopsy dataframe to csv (with included IQR and IPR90 columns now that we calced above)
    output_path = global_dosimetry_dir / 'cohort_global_dosimetry_by_biopsy.csv'
    cohort_global_dosimetry_df.to_csv(output_path, index=False)
    print(f'cohort global dosimetry by biopsy csv saved to file: {output_path}')




    # Get global dosimetry statistics
    biopsy_level_summary_statistics_df = summary_statistics.generate_summary_csv_with_argmax(global_dosimetry_dir, output_filename, cohort_global_dosimetry_df, col_pairs = None, exclude_columns = exclude_for_all)
    ## Global dosimetry analysis (END)

    ### Global dosimetry by voxel analysis (START)
    # Create output directory for global dosimetry by voxel
    global_dosimetry_by_voxel_dir = output_dir.joinpath("global_dosimetry_by_voxel")
    os.makedirs(global_dosimetry_by_voxel_dir, exist_ok=True)
    # Output filename
    output_filename = 'global_dosimetry_by_voxel_statistics_all_patients.csv'
    """
    col_pairs_to_summarize = [
    ('Dose (Gy)', 'argmax_density'),
    ('Dose (Gy)', 'min'),
    ('Dose (Gy)', 'mean'),
    ('Dose (Gy)', 'max'),
    ('Dose (Gy)', 'nominal (spatial average)')
    ('Dose grad (Gy/mm)', 'argmax_density')
    ]
    """
    exclude_for_all = [('Voxel begin (Z)',''),
    ('Voxel end (Z)',''),
    ('Voxel index',''),
    ('Bx ID',''),
    ('Patient ID',''),
    ('Bx index',''),
    ('Simulated bool',''),
    ('Simulated type',''),
    ('Bx refnum','')]

    # Get global dosimetry by voxel statistics
    voxel_wise_summary_statistics_df = summary_statistics.generate_summary_csv_with_argmax(global_dosimetry_by_voxel_dir, output_filename, cohort_global_dosimetry_by_voxel_df, col_pairs = None, exclude_columns = exclude_for_all)
    ## Global dosimetry by voxel analysis (END)





    # Get dvh summary statistics
    ### DVH metrics analysis (START)
    # Create output directory for DVH metrics
    dvh_metrics_dir = output_dir.joinpath("dvh_metrics")
    os.makedirs(dvh_metrics_dir, exist_ok=True)
    # Output filename
    output_filename = 'dvh_metrics_statistics_all_patients.csv'

    # 1. Define which columns to exclude
    """
    NOTE: The columns of the dataframe are:
    print(cohort_global_dosimetry_dvh_metrics_df.columns)
	Index(['Patient ID', 'Metric', 'Bx ID', 'Struct type', 
		'Simulated bool', 'Simulated type', 'Struct index', 'Mean', 'STD',
		'SEM', 'Max', 'Min', 'Skewness', 'Kurtosis', 'Q05', 'Q25', 'Q50', 'Q75',
		'Q95'],
		dtype='object')
    """

    # 2) which columns to carry along (but *not* summarize)
    exclude = [
        'Patient ID','Bx ID','Struct type',
        'Simulated bool','Simulated type','Struct index'
    ]

    # 3) which columns *are* the numeric stats to roll up
    value_cols = [c for c in cohort_global_dosimetry_dvh_metrics_df.columns if c not in exclude + ['Metric']]

    # 4) pivot Metric into your columns by unstacking —  
    #    since we first set_index on (all the exclude cols + 'Metric'), 
    #    each (exclude_combo,Metric) pair is unique, so no duplicate errors.
    wide = (
        cohort_global_dosimetry_dvh_metrics_df
        .set_index(exclude + ['Metric'])[value_cols]  # index=(all the things you want to keep + Metric)
        .unstack(level='Metric')                      # -> columns = (value_col, Metric)
        .swaplevel(0, 1, axis=1)                      # -> columns = (Metric, value_col)
        .sort_index(axis=1, level=0)                  # group by Metric
    )

    # Get DVH metrics statistics
    dvh_summary_statistics_df = summary_statistics.generate_summary_csv_with_argmax(dvh_metrics_dir, output_filename, wide, 
                                            col_pairs=None, 
                                            exclude_columns=exclude)

    # Print the statistics




    # Get all mapped dose values statistics across all trials, voxels and biopsies (START)
    # Create output directory for global dosimetry by voxel
    all_dosimetry_cohort_dir = output_dir.joinpath("all_dosimetry_values_cohort")
    os.makedirs(all_dosimetry_cohort_dir, exist_ok=True)
    # Output filename
    output_filename = 'all_dosimetry_values_cohort.csv'
    # Get all mapped dose values statistics across all trials, voxels and biopsies
    summary_statistics.compute_summary_non_multiindex(all_voxel_wise_dose_df, 
                                                   ["Dose (Gy)", "Dose grad (Gy/mm)"], 
                                                   output_dir=all_dosimetry_cohort_dir,
                                                    csv_name=output_filename)

    








    # Generate effect sizes dataframe

    ### Effect sizes analysis (START)
    print("--------------------------------------------------")
    print("Generating effect sizes analysis...")
    print("--------------------------------------------------")

    eff_sizes = ['cohen', 'hedges', 'mean_diff']
    all_effect_sizes_df_dict = {}
    all_effect_sizes_df_dose_grad_dict = {}
    for eff_size in eff_sizes:
        print(f"Calculating effect sizes for {eff_size}...")
        effect_size_dataframe = helper_funcs.create_eff_size_dataframe(all_voxel_wise_dose_df, "Patient ID", "Bx index", "Bx ID", "Voxel index", "Dose (Gy)", eff_size=eff_size, paired_bool=True)
        effect_size_dataframe_dose_grad = helper_funcs.create_eff_size_dataframe(all_voxel_wise_dose_df, "Patient ID", "Bx index", "Bx ID", "Voxel index", "Dose grad (Gy/mm)", eff_size=eff_size, paired_bool=True)
        # Append the effect size dataframe to the dictionary
        all_effect_sizes_df_dict[eff_size] = effect_size_dataframe
        all_effect_sizes_df_dose_grad_dict[eff_size] = effect_size_dataframe_dose_grad

    
    # Save the effect sizes dataframe to a CSV file
    # Create output directory for effect size analysis
    effect_sizes_analysis_dir = output_dir.joinpath("effect_sizes_analysis")
    os.makedirs(effect_sizes_analysis_dir, exist_ok=True)
    general_output_filename = 'effect_sizes_statistics_all_patients.csv'
    for eff_size in eff_sizes:
        effect_size_dataframe = all_effect_sizes_df_dict[eff_size] 
        effect_size_dataframe.to_csv(effect_sizes_analysis_dir.joinpath(f"{general_output_filename}_{eff_size}.csv"), index=False)
    
    general_output_filename_dose_grad = 'effect_sizes_dose_gradient_statistics_all_patients.csv'
    for eff_size in eff_sizes:
        effect_size_dataframe_dose_grad = all_effect_sizes_df_dose_grad_dict[eff_size]
        effect_size_dataframe_dose_grad.to_csv(effect_sizes_analysis_dir.joinpath(f"{general_output_filename_dose_grad}_{eff_size}.csv"), index=False)


    # be more involved with mean difference
    mean_diff_stats_output_filename = 'mean_diff_statistics_all_patients.csv'
    diffs_df_output_filename = 'mean_diff_values_all_patients.csv'
    mean_diff_stats_all_patients_df, diffs_and_abs_diffs_output_df, mean_diffs_and_abs_diffs_patient_pooled_stats_df, mean_diffs_and_abs_diffs_cohort_pooled_stats_df = helper_funcs.create_diff_stats_dataframe(
        all_voxel_wise_dose_df,
        "Patient ID", 
        "Bx index", 
        "Bx ID", 
        "Voxel index", 
        "Dose (Gy)",
        output_dir = effect_sizes_analysis_dir,
        csv_name_stats_out = mean_diff_stats_output_filename,
        csv_name_diffs_out = diffs_df_output_filename
    )

    mean_diff_stats_output_filename_dose_grad_filename = 'mean_diff_dose_gradient_statistics_all_patients.csv'
    diffs_df_output_filename_dose_grad_filename = 'mean_diff_dose_gradient_values_all_patients.csv'
    mean_diff_stats_all_patients_dose_grad_df, diffs_and_abs_diffs_gradient_output_df, mean_diffs_and_abs_diffs_gradient_patient_pooled_stats_df, mean_diffs_and_abs_diffs_gradient_cohort_pooled_stats_df = helper_funcs.create_diff_stats_dataframe(
        all_voxel_wise_dose_df,
        "Patient ID", 
        "Bx index", 
        "Bx ID", 
        "Voxel index", 
        "Dose grad (Gy/mm)",
        output_dir = effect_sizes_analysis_dir,
        csv_name_stats_out = mean_diff_stats_output_filename_dose_grad_filename,
        csv_name_diffs_out = diffs_df_output_filename_dose_grad_filename
    )


    ### Effect sizes analysis (END)




    # Generate dose differences voxel pairings of all length scales analysis

    ### Dose differences voxel pairings of all length scales analysis (START)
    print("--------------------------------------------------")
    print("Generating dose differences voxel pairings of all length scales for analysis...")
    print("--------------------------------------------------")

    dose_differences_cohort_df = helper_funcs.compute_dose_differences_vectorized(all_voxel_wise_dose_df,column_name = 'Dose (Gy)')
    dose_differences_grad_cohort_df = helper_funcs.compute_dose_differences_vectorized(all_voxel_wise_dose_df,column_name = 'Dose grad (Gy/mm)')



    ### Dose differences voxel pairings of all length scales analysis (END)



    print("--------------------------------------------------")
    print("Voxel pairings of all length scales analysis...")
    print("--------------------------------------------------")

    # Dose differences voxel pairings analysis (START)

    # Create output directory for length scales dosimetry
    length_scales_dir = output_dir.joinpath("length_scales_dosimetry")
    os.makedirs(length_scales_dir, exist_ok=True)
    # Output filename
    output_cohort_filename = 'length_scales_dosimetry_statistics_cohort.csv'
    output_per_biopsy_filename = 'length_scales_dosimetry_statistics_per_biopsy.csv'

    output_cohort_filename_grad = 'length_scales_dose_gradient_statistics_cohort.csv'
    output_per_biopsy_filename_grad = 'length_scales_dose_gradient_statistics_per_biopsy.csv'

    # A) per (Patient ID, Bx index, length_scale)
    _ = summary_statistics.compute_summary(
        dose_differences_cohort_df,
        ['Patient ID','Bx index','length_scale'],
        ['dose_diff','dose_diff_abs'],
        output_dir = length_scales_dir,
        csv_name = output_per_biopsy_filename
    )

    _ = summary_statistics.compute_summary(
        dose_differences_grad_cohort_df,
        ['Patient ID','Bx index','length_scale'],
        ['dose_diff','dose_diff_abs'],
        output_dir = length_scales_dir,
        csv_name = output_per_biopsy_filename_grad
    )

    # B) cohort-wide per length_scale
    _ = summary_statistics.compute_summary(
        dose_differences_cohort_df,
        ['length_scale'],
        ['dose_diff','dose_diff_abs'],
        output_dir = length_scales_dir,
        csv_name = output_cohort_filename
    )

    _ = summary_statistics.compute_summary(
        dose_differences_grad_cohort_df,
        ['length_scale'],
        ['dose_diff','dose_diff_abs'],
        output_dir = length_scales_dir,
        csv_name = output_cohort_filename_grad
    )









    ### Print break with horizontal lines
    print(" ")
    print("--------------------------------------------------")
    print("--------------------------------------------------")
    print("--------------------------------------------------")
    print(" ")



    ### PLOTS FROM RAW DATA
    print("--------------------------------------------------")
    print("Generating plots from raw data...")
    print("--------------------------------------------------")










    print("--------------------------------------------------")
    print("Figures: Cohort figures...")
    print("--------------------------------------------------")


    print("Generating deltas cohort figures...")

    if True:
        print("Skipping!")
    else:

        """        
        _, _ = production_plots.plot_cohort_deltas_boxplot(nominal_deltas_df, save_dir=cohort_output_figures_dir, fig_name="dose_deltas_cohort",
                            zero_level_index_str='Dose (Gy)', include_patient_ids=None,
                            show_points=True)
        _, _ = production_plots.plot_cohort_deltas_boxplot(nominal_gradient_deltas_df, save_dir=cohort_output_figures_dir, fig_name="dose_gradient_deltas_cohort",
                            zero_level_index_str='Dose grad (Gy/mm)', include_patient_ids=None,
                            show_points=True)
        """
        _ = plot_cohort_deltas_boxplot(
            nominal_deltas_df_with_abs,
            save_dir=cohort_output_figures_dir,
            fig_name="dose_deltas_cohort_math_hue",
            zero_level_index_str='Dose (Gy)',
            show_points=True,
            include_abs=True,
            abs_as_hue=True,
            label_style='math',         # -> Δ^{mean}/Δ^{mode}/Δ^{Q50}
            median_superscript='Q50',   # or 'median'
            require_precomputed_abs=True
        )
        _ = plot_cohort_deltas_boxplot(
            nominal_gradient_deltas_df_with_abs,
            save_dir=cohort_output_figures_dir,
            fig_name="dose_gradient_deltas_cohort_math_hue",
            zero_level_index_str='Dose grad (Gy/mm)',
            show_points=True,
            include_abs=True,
            abs_as_hue=True,
            label_style='math',         # -> Δ^{mean}/Δ^{mode}/Δ^{Q50}
            median_superscript='Q50',   # or 'median'
            require_precomputed_abs=True
        )


        

        """
        _, _ = production_plots.plot_cohort_deltas_boxplot_by_voxel(
            nominal_deltas_df,
            cohort_output_figures_dir,
            fig_name="dose_deltas_box_by_voxel_cohort",
            zero_level_index_str = 'Dose (Gy)',   # must match what you passed to compute_biopsy_nominal_deltas
            x_axis = 'Voxel index',               # or 'Voxel begin (Z)'
            axes_label_fontsize = 14,
            tick_label_fontsize= 12,
            title= None,
            show_points= False,   # new flag
            point_size = 3,        # adjust visibility
            alpha = 0.5, 
        )

        _, _ = production_plots.plot_cohort_deltas_boxplot_by_voxel(
            nominal_gradient_deltas_df,
            cohort_output_figures_dir,
            fig_name="dose_gradient_deltas_box_by_voxel_cohort",
            zero_level_index_str = 'Dose grad (Gy/mm)',   # must match what you passed to compute_biopsy_nominal_deltas
            x_axis = 'Voxel index',               # or 'Voxel begin (Z)'
            axes_label_fontsize = 14,
            tick_label_fontsize= 12,
            title= None,
            show_points= False,   # new flag
            point_size = 3,        # adjust visibility
            alpha = 0.5, 
        )
        """

        _ = plot_cohort_deltas_boxplot_by_voxel(
            nominal_deltas_df_with_abs,
            cohort_output_figures_dir,
            fig_name="dose_deltas_box_by_voxel_cohort",
            zero_level_index_str='Dose (Gy)',
            x_axis='Voxel index',
            show_points=False,
            include_abs=True,
            abs_as_hue=True,           # ← Signed vs Absolute as hue, facets = mean/mode/median
            label_style='math',
            median_superscript='Q50'
        )

        _ = plot_cohort_deltas_boxplot_by_voxel(
            nominal_gradient_deltas_df_with_abs,
            cohort_output_figures_dir,
            fig_name="dose_gradient_deltas_box_by_voxel_cohort",
            zero_level_index_str='Dose grad (Gy/mm)',
            x_axis='Voxel index',
            show_points=False,
            include_abs=True,
            abs_as_hue=False,          # ← Δ and |Δ| as separate legend entries
            label_style='math',
            median_superscript='Q50'
        )



        
    print('stop')

    if False:
        print("Skipping!")
    else:

        # 1. all voxels histograms dosimetry and gradient

        #dists_to_try = ['lognorm', 'gamma', 'gengamma', 'weibull_min', 'skewnorm'] # most likely correct
        #dists_to_try = None # try all
        dists_to_try = ['lognorm'] # lognorm is the best fit for most of the data, so we will use this for now
        xrange_dose = (0, 100)  # Adjust the range as needed
        xrange_dose_grad = (0, 50)  # Adjust the range as needed
        #production_plots.histogram_and_fit(all_voxel_wise_dose_df, dists_to_try = dists_to_try, bin_size = 1, dose_col="Dose (Gy)", save_path = cohort_output_figures_dir, custom_name = "histogram_fit_all_voxels_dose", xrange = xrange_dose, vertical_gridlines= True, horizontal_gridlines=True)

        #production_plots.histogram_and_fit(all_voxel_wise_dose_df, dists_to_try = dists_to_try, bin_size = 1, dose_col="Dose grad (Gy/mm)", save_path = cohort_output_figures_dir, custom_name = "histogram_fit_all_voxels_dose_gradient", xrange = xrange_dose_grad, vertical_gridlines= True, horizontal_gridlines=True)


        production_plots.histogram_and_fit_v2(
            all_voxel_wise_dose_df,
            dists_to_try = dists_to_try,
            bin_size=1,
            dose_col="Dose (Gy)",
            save_path=cohort_output_figures_dir,
            custom_name="histogram_fit_all_voxels_dose",
            xrange=xrange_dose,
            vertical_gridlines=True,
            horizontal_gridlines=True,
            show_minor_ticks=True,
            vertical_minor_gridlines=True,
            quantity_tex=r"D_{b,v}^{(t)}",
            quantity_unit_tex="Gy",
            title=None,
        )

        production_plots.histogram_and_fit_v2(
            all_voxel_wise_dose_df,
            dists_to_try = dists_to_try,
            bin_size=1,
            dose_col="Dose grad (Gy/mm)",
            save_path=cohort_output_figures_dir,
            custom_name="histogram_fit_all_voxels_dose_gradient",
            xrange=xrange_dose_grad,
            vertical_gridlines=True,
            horizontal_gridlines=True,
            show_minor_ticks=True,
            vertical_minor_gridlines=True,
            quantity_tex=r"G_{b,v}^{(t)}",
            quantity_unit_tex=r"Gy mm$^{-1}$",
            title=None,
        )






        # 1. DONE



        # 2. Effect size heatmaps
        print("Generating effect size heatmaps...")

        eff_size_heatmaps_dir = cohort_output_figures_dir.joinpath(f"effect_sizes_heatmaps")
        os.makedirs(eff_size_heatmaps_dir, exist_ok=True)
        
        for eff_size in eff_sizes:
        
            effect_size_dataframe = all_effect_sizes_df_dict[eff_size] 
            effect_size_dataframe_dose_grad = all_effect_sizes_df_dose_grad_dict[eff_size]
            for agg_abs in [False, True]:

                
                production_plots.plot_cohort_eff_size_heatmap_boxed_counts(effect_size_dataframe,
                                                "Effect Size",
                                                eff_size,
                                                save_path_base=eff_size_heatmaps_dir,
                                                save_name_base = 'dose',
                                                annotation_info=None,
                                                aggregate_abs=agg_abs,
                                                vmin=None,
                                                vmax=None)
                
                production_plots.plot_cohort_eff_size_heatmap_boxed_counts(effect_size_dataframe_dose_grad,
                                                "Effect Size",
                                                eff_size,
                                                save_path_base=eff_size_heatmaps_dir,
                                                save_name_base = 'dose_gradient',
                                                annotation_info=None,
                                                aggregate_abs=agg_abs,
                                                vmin=None,
                                                vmax=None)
                
        for agg_abs in [False, True]:
            production_plots.plot_cohort_eff_size_heatmap_boxed_counts_and_std(
                mean_diff_stats_all_patients_df,
                "mean_diff",
                "mean_diff_with_std",
                save_path_base=eff_size_heatmaps_dir,
                save_name_base = 'dose',
                annotation_info= None,
                vmin = None,
                vmax = None
            )

            production_plots.plot_cohort_eff_size_heatmap_boxed_counts_and_std(
                mean_diff_stats_all_patients_dose_grad_df,
                "mean_diff",
                "mean_diff_with_std",
                save_path_base=eff_size_heatmaps_dir,
                save_name_base = 'dose_gradient',
                annotation_info= None,
                vmin = None,
                vmax = None
            )

            # New: dual-triangle cohort plot (upper=dose, lower=dose gradient)
            production_plots.plot_cohort_eff_size_dualtri_mean_std(
                upper_df=mean_diff_stats_all_patients_df,
                lower_df=mean_diff_stats_all_patients_dose_grad_df,
                eff_size_col="mean_diff",
                eff_size_type_upper="Dose mean±std",
                eff_size_type_lower="Dose-Gradient mean±std",
                save_path_base=eff_size_heatmaps_dir,
                save_name_base=f"dose_upper__dosegrad_lower_{'abs' if agg_abs else 'signed'}",
                annotation_info=None,
                aggregate_abs=agg_abs,
                vmin=None, vmax=None,
                # counts overlay
                show_counts_boxes = True,
                counts_source = "lower",       # "lower" or "upper"
                # typography controls
                tick_label_fontsize = 12,
                axis_label_fontsize = 14,
                cbar_tick_fontsize = 12,
                cbar_label_fontsize = 14,
                cbar_label_upper = "Mean Difference (Gy, Upper triangle)",
                cbar_label_lower = "Mean Difference (Gy/mm, Lower triangle)",
                show_title = False,
                # n= annotation fontsize
                n_label_fontsize = 7,
                show_annotation_box=False
            )

            ## USING POOLED DATA STATISTICS!! THIS SHOULD BE BETTER
            # signed differences
            production_plots.plot_cohort_eff_size_dualtri_mean_std_with_pooled_dfs(
                mean_diffs_and_abs_diffs_cohort_pooled_stats_df,             # EXPECTS cohort-pooled stats per (voxel1, voxel2)
                mean_diffs_and_abs_diffs_gradient_cohort_pooled_stats_df,             # EXPECTS cohort-pooled stats per (voxel1, voxel2)

                # tell the function which columns to render in cells:
                upper_mean_col = "mean_diff",
                upper_std_col = "std_diff",
                lower_mean_col = "mean_diff",
                lower_std_col = "std_diff",

                # which column to use for the "n=" boxes (per voxel pair)
                n_col = "n_biopsies",

                eff_size_type_upper = "Dose mean±std",
                eff_size_type_lower = "Dose-Gradient mean±std",

                save_path_base=eff_size_heatmaps_dir,
                save_name_base=f"dose_upper__dosegrad_lower_signed_pooledstats",
                annotation_info = None,

                # color range controls
                vmin = None,
                vmax = None,
                vmin_upper = None,
                vmax_upper = None,
                vmin_lower = None,
                vmax_lower = None,

                # counts overlay
                show_counts_boxes = True,
                counts_source = "lower",       # "lower" or "upper"

                # typography
                tick_label_fontsize = 12,
                axis_label_fontsize = 14,
                cbar_tick_fontsize = 12,
                cbar_label_fontsize = 14,
                cbar_label_upper = "Mean of Signed Difference (Gy, Upper triangle)",
                cbar_label_lower = "Mean of Signed Difference (Gy/mm, Lower triangle)",

                # title
                show_title = False,

                # n= caption fontsize inside boxes
                n_label_fontsize = 7,

                # corner annotation box
                show_annotation_box = False,
            )

            # Absolute differences 
            production_plots.plot_cohort_eff_size_dualtri_mean_std_with_pooled_dfs(
                mean_diffs_and_abs_diffs_cohort_pooled_stats_df,             # EXPECTS cohort-pooled stats per (voxel1, voxel2)
                mean_diffs_and_abs_diffs_gradient_cohort_pooled_stats_df,             # EXPECTS cohort-pooled stats per (voxel1, voxel2)

                # tell the function which columns to render in cells:
                upper_mean_col = "mean_abs_diff",
                upper_std_col = "std_abs_diff",
                lower_mean_col = "mean_abs_diff",
                lower_std_col = "std_abs_diff",

                # which column to use for the "n=" boxes (per voxel pair)
                n_col = "n_biopsies",

                eff_size_type_upper = "Dose mean±std",
                eff_size_type_lower = "Dose-Gradient mean±std",

                save_path_base=eff_size_heatmaps_dir,
                save_name_base=f"dose_upper__dosegrad_lower_absolute_pooledstats",
                annotation_info = None,

                # color range controls
                vmin = None,
                vmax = None,
                vmin_upper = None,
                vmax_upper = None,
                vmin_lower = None,
                vmax_lower = None,

                # counts overlay
                show_counts_boxes = True,
                counts_source = "lower",       # "lower" or "upper"

                # typography
                tick_label_fontsize = 12,
                axis_label_fontsize = 14,
                cbar_tick_fontsize = 12,
                cbar_label_fontsize = 14,
                cbar_label_upper = "Mean of Absolute Differences (Gy, Upper triangle)",
                cbar_label_lower = "Mean of Absolute Differences (Gy/mm, Lower triangle)",

                # title
                show_title = False,

                # n= caption fontsize inside boxes
                n_label_fontsize = 7,

                # corner annotation box
                show_annotation_box = False,
            )

            # signed, with std
            production_plots.plot_cohort_eff_size_dualtri_mean_std_with_pooled_dfs_v2(
                mean_diffs_and_abs_diffs_cohort_pooled_stats_df,
                mean_diffs_and_abs_diffs_gradient_cohort_pooled_stats_df,
                upper_mean_col="mean_diff", upper_std_col="std_diff",
                lower_mean_col="mean_diff", lower_std_col="std_diff",
                save_path_base=eff_size_heatmaps_dir,
                save_name_base=f"dose_upper_dosegrad_lower_signed_pooledstats_with_std",
                n_col="n_biopsies",
                n_label_fontsize = 10,
                cell_annot_fontsize=4.5,
                tick_label_fontsize = 12,
                axis_label_fontsize = 14,
                cbar_tick_fontsize = 12,
                cbar_label_fontsize = 14,
                cbar_label_upper=r"$\overline{M_{ij}^{D}}$ ($\mathrm{Gy}$, Upper triangle)",
                cbar_label_lower=r"$\overline{M_{ij}^{G}}$ ($\mathrm{Gy}\,\mathrm{mm}^{-1}$, Lower triangle)",
                show_title=False,
                show_annotation_box=False,
                # NEW:
                cbar_pad = 0.6,        # distance between matrix and colorbar
                cbar_label_pad = 8.0,  # distance between bar and its label
            )

            # signed, NO std
            production_plots.plot_cohort_eff_size_dualtri_mean_std_with_pooled_dfs_v2(
                mean_diffs_and_abs_diffs_cohort_pooled_stats_df,
                mean_diffs_and_abs_diffs_gradient_cohort_pooled_stats_df,
                upper_mean_col="mean_diff", upper_std_col=None,
                lower_mean_col="mean_diff", lower_std_col=None,
                save_path_base=eff_size_heatmaps_dir,
                save_name_base=f"dose_upper_dosegrad_lower_signed_pooledstats_no_std",
                n_col="n_biopsies",
                n_label_fontsize = 10,
                cell_annot_fontsize=6,
                tick_label_fontsize = 12,
                axis_label_fontsize = 14,
                cbar_tick_fontsize = 12,
                cbar_label_fontsize = 14,
                cbar_label_upper=r"$\overline{M_{ij}^{D}}$ ($\mathrm{Gy}$, Upper triangle)",
                cbar_label_lower=r"$\overline{M_{ij}^{G}}$ ($\mathrm{Gy}\,\mathrm{mm}^{-1}$, Lower triangle)",
                show_title=False,
                show_annotation_box=False,
                # NEW:
                cbar_pad = 0.6,        # distance between matrix and colorbar
                cbar_label_pad = 8.0,  # distance between bar and its label
            )

            # Absolute differences with std
            production_plots.plot_cohort_eff_size_dualtri_mean_std_with_pooled_dfs_v2(
                mean_diffs_and_abs_diffs_cohort_pooled_stats_df,
                mean_diffs_and_abs_diffs_gradient_cohort_pooled_stats_df,
                upper_mean_col="mean_abs_diff", upper_std_col="std_abs_diff",
                lower_mean_col="mean_abs_diff", lower_std_col="std_abs_diff",
                save_path_base=eff_size_heatmaps_dir,
                save_name_base=f"dose_upper_dosegrad_lower_absolute_pooledstats_with_std",
                n_col="n_biopsies",
                n_label_fontsize = 10,
                cell_annot_fontsize=4.5,
                tick_label_fontsize = 12,
                axis_label_fontsize = 14,
                cbar_tick_fontsize = 12,
                cbar_label_fontsize = 14,
                cbar_label_upper=r"$\overline{|M_{ij}^{D}|}$ ($\mathrm{Gy}$, Upper triangle)",
                cbar_label_lower=r"$\overline{|M_{ij}^{G}|}$ ($\mathrm{Gy}\,\mathrm{mm}^{-1}$, Lower triangle)",
                show_title=False,
                show_annotation_box=False,
                # new: sequential colormap and zero anchored at the minimum
                vmin_upper=0.0,
                vmin_lower=0.0,
                cmap="Reds",
                # NEW:
                cbar_pad = 0.6,        # distance between matrix and colorbar
                cbar_label_pad = 8.0,  # distance between bar and its label
            )


            # Absolute differences NO std
            production_plots.plot_cohort_eff_size_dualtri_mean_std_with_pooled_dfs_v2(
                mean_diffs_and_abs_diffs_cohort_pooled_stats_df,
                mean_diffs_and_abs_diffs_gradient_cohort_pooled_stats_df,
                upper_mean_col="mean_abs_diff", upper_std_col=None,
                lower_mean_col="mean_abs_diff", lower_std_col=None,
                save_path_base=eff_size_heatmaps_dir,
                save_name_base=f"dose_upper_dosegrad_lower_absolute_pooledstats_no_std",
                n_col="n_biopsies",
                n_label_fontsize = 10,
                cell_annot_fontsize= 6.9,
                tick_label_fontsize = 12,
                axis_label_fontsize = 14,
                cbar_tick_fontsize = 12,
                cbar_label_fontsize = 14,
                cbar_label_upper=r"$\overline{|M_{ij}^{D}|}$ ($\mathrm{Gy}$, Upper triangle)",
                cbar_label_lower=r"$\overline{|M_{ij}^{G}|}$ ($\mathrm{Gy}\,\mathrm{mm}^{-1}$, Lower triangle)",
                show_title=False,
                show_annotation_box=False,
                # new: sequential colormap and zero anchored at the minimum
                vmin_upper=0.0,
                vmin_lower=0.0,
                cmap="Reds",
                # NEW:
                cbar_pad = 0.6,        # distance between matrix and colorbar
                cbar_label_pad = 8.0,  # distance between bar and its label
            )


            print('test')
            
            #production_plots.plot_eff_size_heatmaps(effect_size_dataframe, "Patient ID", "Bx index", "Bx ID", "Effect Size", eff_size, save_dir=eff_size_heatmaps_dir)

        # 2. DONE



        # 3. Cohort DVH metrics boxplot
        print("Generating cohort DVH metrics boxplot...")

        production_plots.dvh_boxplot(cohort_global_dosimetry_dvh_metrics_df, 
                                     save_path = cohort_output_figures_dir, 
                                     custom_name = "dvh_boxplot", 
                                     title = None, 
                                     axis_label_font_size = 14,
                                     tick_label_font_size = 12
                                    )




        # 4. Voxel pairings length scales strip plot

        print("Generating voxel pairings length scales strip plot...")
        """
        production_plots.plot_strip_scatter(
                                        dose_differences_cohort_df,
                                        'length_scale',
                                        'dose_diff_abs',
                                        save_dir = cohort_output_figures_dir,
                                        file_name = "dose_differences_voxel_pairings_length_scales_strip_plot",
                                        title = "Dose Differences Voxel Pairings Length Scales Strip Plot",
                                        figsize=(10, 6),
                                        dpi=300
                                        )
        """
        """
        production_plots.plot_dose_vs_length_with_summary(
                                    dose_differences_cohort_df,
                                    'length_scale',
                                    'dose_diff_abs',
                                    save_dir = cohort_output_figures_dir,
                                    file_name = "dose_differences_voxel_pairings_length_scales_strip_plot",
                                    title = "Dose Differences Voxel Pairings Length Scales Strip Plot",
                                    figsize=(10, 6),
                                    dpi=300,
                                    show_points=False,
                                    violin_or_box='box',
                                    trend_lines = ['mean'],
                                    annotate_counts=True,
                                    y_trim=True,
                                    y_min_quantile=0.05,
                                    y_max_quantile=0.95,
                                    y_min_fixed=0,
                                    y_max_fixed=None,
                                    xlabel = "Length Scale (mm)",
                                    ylabel = "Absolute Dose Difference (Gy)",
                                )

        production_plots.plot_dose_vs_length_with_summary(
                                    dose_differences_grad_cohort_df,
                                    'length_scale',
                                    'dose_diff_abs',
                                    save_dir = cohort_output_figures_dir,
                                    file_name = "dose_gradient_differences_voxel_pairings_length_scales_strip_plot",
                                    title = "Dose Gradient Differences Voxel Pairings Length Scales Strip Plot",
                                    figsize=(10, 6),
                                    dpi=300,
                                    show_points=False,
                                    violin_or_box='box',
                                    trend_lines = ['mean'],
                                    annotate_counts=True,
                                    y_trim=True,
                                    y_min_quantile=0.05,
                                    y_max_quantile=0.95,
                                    y_min_fixed=0,
                                    y_max_fixed=None,
                                    xlabel = "Length Scale (mm)",
                                    ylabel = "Absolute Dose Gradient Difference (Gy/mm)",
                                )
        """


        # Dose (absolute differences) — WITH per-biopsy family curves
        production_plots.plot_dose_vs_length_with_summary_cohort_v2(
            dose_differences_cohort_df,
            x_col="length_scale",
            y_col="dose_diff_abs",
            save_dir=cohort_output_figures_dir,
            file_name="cohort_dose_abs_box_with_all_biopsy_mean_curves_v2",
            title=None,
            violin_or_box="box",
            trend_lines=("mean",),
            annotate_counts=True,
            annotation_box=False,
            y_trim=True,
            y_min_fixed=0,
            metric_family="dose",
            show_pair_mean_curves=True,
            show_pair_legend=False,
            pair_line_alpha=0.5,
            pair_line_width=0.9,
            box_color="#D8D8D8",
        )

        # Dose (absolute differences) — NO family curves (global mean still shown)
        production_plots.plot_dose_vs_length_with_summary_cohort_v2(
            dose_differences_cohort_df,
            x_col="length_scale",
            y_col="dose_diff_abs",
            save_dir=cohort_output_figures_dir,
            file_name="cohort_dose_abs_box_v2",
            title=None,
            violin_or_box="box",
            trend_lines=("mean",),
            annotate_counts=True,
            annotation_box=False,
            y_trim=True,
            y_min_fixed=0,
            metric_family="dose",
            show_pair_mean_curves=False,
            box_color="#D8D8D8",
        )

        # Grad (absolute differences) — NO family curves (global mean still shown)
        production_plots.plot_dose_vs_length_with_summary_cohort_v2(
            dose_differences_grad_cohort_df,
            x_col="length_scale",
            y_col="dose_diff_abs",
            save_dir=cohort_output_figures_dir,
            file_name="cohort_grad_abs_box_v2",
            title=None,
            violin_or_box="box",
            trend_lines=("mean",),
            annotate_counts=True,
            annotation_box=False,
            y_trim=True,
            y_min_fixed=0,
            metric_family="grad",
            show_pair_mean_curves=False,
            box_color="#D8D8D8",
        )

        # Grad (absolute differences) — WITH per-biopsy family curves
        production_plots.plot_dose_vs_length_with_summary_cohort_v2(
            dose_differences_grad_cohort_df,
            x_col="length_scale",
            y_col="dose_diff_abs",
            save_dir=cohort_output_figures_dir,
            file_name="cohort_grad_abs_box_with_all_biopsy_mean_curves_v2",
            title=None,
            violin_or_box="box",
            trend_lines=("mean",),
            annotate_counts=True,
            annotation_box=False,
            y_trim=True,
            y_min_fixed=0,
            metric_family="grad",
            show_pair_mean_curves=True,
            show_pair_legend=False,
            pair_line_alpha=0.5,
            pair_line_width=0.9,
            box_color="#D8D8D8",
        )




        print('test')






        # 5. Cohort global scores boxplot
        print("Generating cohort global scores boxplot...")

        ### DOSE
        production_plots.plot_global_dosimetry_boxplot(
            cohort_global_dosimetry_df,
            'Dose (Gy)',  # e.g. 'Dose (Gy)'
            ['nominal (spatial average)', 'argmax_density','mean', 'std'],  # e.g. ['mean', 'min', 'max', 'quantile_05']
            cohort_output_figures_dir,
            file_name = "global_scores_boxplot",
            title = 'Cohort Global Scores Boxplot',
            figsize=(10, 6),
            dpi=300,
            xlabel = "Global Statistic",
            ylabel = "Dose (Gy)",
            showfliers = True,
            label_map={'argmax_density': 'Argmax Density',
            'min': 'Minimum',
            'mean': 'Mean',
            'max': 'Maximum',
            'std': 'STD',
            'nominal (spatial average)': 'Nominal'
        }
        )

        # horizontal 
        production_plots.plot_global_dosimetry_boxplot(
            cohort_global_dosimetry_df,
            'Dose (Gy)',  # e.g. 'Dose (Gy)'
            ['nominal (spatial average)', 'argmax_density','mean', 'std'],  # e.g. ['mean', 'min', 'max', 'quantile_05']
            cohort_output_figures_dir,
            file_name = "global_scores_boxplot_horizontal",
            title = 'Cohort Global Scores Boxplot',
            figsize=(10, 6),
            dpi=300,
            xlabel = "Dose (Gy)",
            ylabel = "Global Statistic",
            showfliers = True,
            label_map={'argmax_density': 'Argmax Density',
            'min': 'Minimum',
            'mean': 'Mean',
            'max': 'Maximum',
            'std': 'STD',
            'nominal (spatial average)': 'Nominal'
        },
            horizontal=True
        )

        production_plots.plot_global_dosimetry_boxplot(
            cohort_global_dosimetry_df,
            'Dose (Gy)',  # e.g. 'Dose (Gy)'
            ['quantile_05', 'quantile_25', 'quantile_50', 'quantile_75', 'quantile_95'], # e.g. ['mean', 'min', 'max', 'quantile_05']
            cohort_output_figures_dir,
            file_name = "global_scores_boxplot_quantiles",
            title = 'Cohort Global Scores Boxplot (Quantiles)',
            figsize=(10, 6),
            dpi=300,
            xlabel = "Global Statistic",
            ylabel = "Dose (Gy)",
            showfliers = True,
            label_map={'quantile_05': 'Quantile 5%',
            'quantile_25': 'Quantile 25%',
            'quantile_50': 'Quantile 50%',
            'quantile_75': 'Quantile 75%',
            'quantile_95': 'Quantile 95%'
            }
        )


        production_plots.plot_global_dosimetry_boxplot(
            cohort_global_dosimetry_df,
            'Dose (Gy)',  # e.g. 'Dose (Gy)'
            ['skewness','kurtosis'],  # e.g. ['mean', 'min', 'max', 'quantile_05']
            cohort_output_figures_dir,
            file_name = "global_scores_boxplot_skew_kurtosis",
            title = 'Cohort Global Scores Boxplot (Skewness and Kurtosis)',
            figsize=(10, 6),
            dpi=300,
            xlabel = "Global Statistic",
            ylabel = "Dimensionless Score",
            showfliers = True,
            label_map={
            'skewness': 'Skewness',
            'kurtosis': 'Kurtosis'
        }
        )

        ### DOSE GRADIENT


        production_plots.plot_global_dosimetry_boxplot(
            cohort_global_dosimetry_df,
            'Dose grad (Gy/mm)',  # e.g. 'Dose (Gy)'
            ['nominal (spatial average)', 'argmax_density','mean', 'std'],  # e.g. ['mean', 'min', 'max', 'quantile_05']
            cohort_output_figures_dir,
            file_name = "global_scores_boxplot_grad",
            title = 'Cohort Global Scores Boxplot (Dose Gradient)',
            figsize=(10, 6),
            dpi=300,
            xlabel = "Global Statistic",
            ylabel = "Dose grad (Gy/mm)",
            showfliers = True,
            label_map={'argmax_density': 'Argmax Density',
            'min': 'Minimum',
            'mean': 'Mean',
            'max': 'Maximum',
            'std': 'STD',
            'nominal (spatial average)': 'Nominal'
        }
        )

        # horizontal
        production_plots.plot_global_dosimetry_boxplot(
            cohort_global_dosimetry_df,
            'Dose grad (Gy/mm)',  # e.g. 'Dose (Gy)'
            ['nominal (spatial average)', 'argmax_density','mean', 'std'],  # e.g. ['mean', 'min', 'max', 'quantile_05']
            cohort_output_figures_dir,
            file_name = "global_scores_boxplot_grad_horizontal",
            title = 'Cohort Global Scores Boxplot (Dose Gradient)',
            figsize=(10, 6),
            dpi=300,
            xlabel = "Global Statistic",
            ylabel = "Dose grad (Gy/mm)",
            showfliers = True,
            label_map={'argmax_density': 'Argmax Density',
            'min': 'Minimum',
            'mean': 'Mean',
            'max': 'Maximum',
            'std': 'STD',
            'nominal (spatial average)': 'Nominal'
        },
            horizontal=True
        )

        production_plots.plot_global_dosimetry_boxplot(
            cohort_global_dosimetry_df,
            'Dose grad (Gy/mm)',  # e.g. 'Dose (Gy)'
            ['quantile_05', 'quantile_25', 'quantile_50', 'quantile_75', 'quantile_95'], # e.g. ['mean', 'min', 'max', 'quantile_05']
            cohort_output_figures_dir,
            file_name = "global_scores_boxplot_grad_quantiles",
            title = 'Cohort Global Scores Boxplot (Dose Gradient, Quantiles)',
            figsize=(10, 6),
            dpi=300,
            xlabel = "Global Statistic",
            ylabel = "Dose (Gy)",
            showfliers = True,
            label_map={'quantile_05': 'Quantile 5%',
            'quantile_25': 'Quantile 25%',
            'quantile_50': 'Quantile 50%',
            'quantile_75': 'Quantile 75%',
            'quantile_95': 'Quantile 95%'
            }
        )


        production_plots.plot_global_dosimetry_boxplot(
            cohort_global_dosimetry_df,
            'Dose grad (Gy/mm)',  # e.g. 'Dose (Gy)'
            ['skewness','kurtosis'],  # e.g. ['mean', 'min', 'max', 'quantile_05']
            cohort_output_figures_dir,
            file_name = "global_scores_boxplot_grad_skew_kurtosis",
            title = 'Cohort Global Scores Boxplot (Dose Gradient, Skewness and Kurtosis)',
            figsize=(10, 6),
            dpi=300,
            xlabel = "Global Statistic",
            ylabel = "Dimensionless Score",
            showfliers = True,
            label_map={
            'skewness': 'Skewness',
            'kurtosis': 'Kurtosis'
        }
        )


        # 6. cohort ridgeline dose
        # this one takes a while, the annotate and fill function is slow
        if True:
            print("Skipping cohort ridgeline dose plot!")
        else:
            production_plots.plot_dose_ridge_cohort_by_voxel(
                all_point_wise_dose_df,
                cohort_output_figures_dir,
                "Biopsy Voxel-Wise Dose Ridgeline Plot - Cohort",
                "dose",
                fig_scale=1.0,
                dpi=300,
                add_text_annotations=True,
                x_label="Dose (Gy)",
                y_label="Axial Dimension (mm)",
                space_between_ridgeline_padding_multiplier=1.2,
                ridgeline_vertical_padding_value=0.25
            )

    print("--------------------------------------------------")
    print("Figures: Cohort figures DONE!")
    print("--------------------------------------------------")








    ### Pick patient and biopsy pairs to plot

    patient_id_and_bx_index_pairs = [('181 (F2)',0), ('181 (F2)', 1), ('184 (F2)', 0), ('184 (F2)', 1), ('184 (F2)', 2), ('195 (F2)', 0), ('195 (F2)', 1), ('201 (F2)', 0),('201 (F2)', 1),('201 (F2)', 2)]

    ###



    print("--------------------------------------------------")
    print("Figures: Individual patient dosimetry and dose gradient deltas...")
    print("--------------------------------------------------")



    if False:
        print("Skipping!")
    else:
        """
        for patient_id, bx_index in patient_id_and_bx_index_pairs:

            # Create a directory for the patient
            patient_dir = pt_sp_figures_dir.joinpath(patient_id)
            os.makedirs(patient_dir, exist_ok=True)
            
            bx_id = nominal_deltas_df[(nominal_deltas_df['Patient ID'] == patient_id) & (nominal_deltas_df['Bx index'] == bx_index)]['Bx ID'].values[0]


            general_plot_name_string = f"{patient_id} - {bx_id} - dosimetry-deltas-plot"

            zero_level_index_str = 'Dose (Gy)'   # must match what you passed to compute_biopsy_nominal_deltas

            production_plots.plot_biopsy_deltas_line(
                nominal_deltas_df,
                patient_id,
                bx_index,
                patient_dir,
                general_plot_name_string,
                zero_level_index_str = zero_level_index_str,   # must match what you passed to compute_biopsy_nominal_deltas
                x_axis = 'Voxel index',               # or 'Voxel begin (Z)'
                axes_label_fontsize = 14,
                tick_label_fontsize = 12,
                title = zero_level_index_str+ ' Deltas line plot - Patient '+patient_id+', Bx index '+str(bx_index)+', Bx ID '+str(bx_id),
            )

            # Dose gradient deltas
            general_plot_name_string = f"{patient_id} - {bx_id} - dosimetry-gradient-deltas-plot"
            zero_level_index_str = 'Dose grad (Gy/mm)'   # must match what you passed to compute_biopsy_nominal_deltas

            bx_id = nominal_gradient_deltas_df[(nominal_gradient_deltas_df['Patient ID'] == patient_id) & (nominal_gradient_deltas_df['Bx index'] == bx_index)]['Bx ID'].values[0]
            production_plots.plot_biopsy_deltas_line(
                nominal_gradient_deltas_df,
                patient_id,
                bx_index,
                patient_dir,
                general_plot_name_string,
                zero_level_index_str = zero_level_index_str,   # must match what you passed to compute_biopsy_nominal_deltas
                x_axis = 'Voxel index',               # or 'Voxel begin (Z)'
                axes_label_fontsize = 14,
                tick_label_fontsize = 12,
                title = zero_level_index_str+ ' Deltas line plot - Patient '+patient_id+', Bx index '+str(bx_index)+', Bx ID '+str(bx_id),
            )
        """

        for patient_id, bx_index in patient_id_and_bx_index_pairs:
            # directories
            patient_dir = pt_sp_figures_dir.joinpath(patient_id)
            os.makedirs(patient_dir, exist_ok=True)

            # ----- Dose (Gy) -----
            bx_id = nominal_deltas_df_with_abs[
                (nominal_deltas_df_with_abs[('Patient ID','')] == patient_id) &
                (nominal_deltas_df_with_abs[('Bx index','')] == bx_index)
            ][('Bx ID','')].values[0]

            general_plot_name_string = f"{patient_id} - {bx_id} - dosimetry-deltas-plot"
            zero_level_index_str = 'Dose (Gy)'

            production_plots.plot_biopsy_deltas_line_both_signed_and_abs(
                nominal_deltas_df_with_abs,              # <-- use the _with_abs df
                patient_id,
                bx_index,
                patient_dir,
                general_plot_name_string,
                zero_level_index_str = zero_level_index_str,
                x_axis = 'Voxel index',                  # or 'Voxel begin (Z)'
                axes_label_fontsize = 14,
                tick_label_fontsize = 12,
                title = f"{zero_level_index_str} Deltas — Patient {patient_id}, Bx {bx_index}, Bx ID {bx_id}",
                include_abs = True,
                require_precomputed_abs = True,          # strict: read abs block only
                fallback_recompute_abs = False,          # set True only if you want |Δ| on-the-fly
                label_style = 'math',                    # Δ^{mean}, Δ^{mode}, Δ^{Q50}
                median_superscript = 'Q50',
                order_kinds = ('mean','mode','median'),
                show_points = False,
                point_size = 3,
                alpha = 0.5,
                linewidth_signed = 2.0,
                linewidth_abs = 2.0,
                show_title=False
            )

            # ----- Dose gradient (Gy/mm) -----
            bx_id = nominal_gradient_deltas_df_with_abs[
                (nominal_gradient_deltas_df_with_abs[('Patient ID','')] == patient_id) &
                (nominal_gradient_deltas_df_with_abs[('Bx index','')] == bx_index)
            ][('Bx ID','')].values[0]

            general_plot_name_string = f"{patient_id} - {bx_id} - dosimetry-gradient-deltas-plot"
            zero_level_index_str = 'Dose grad (Gy/mm)'

            production_plots.plot_biopsy_deltas_line_both_signed_and_abs(
                nominal_gradient_deltas_df_with_abs,     # <-- use the _with_abs df
                patient_id,
                bx_index,
                patient_dir,
                general_plot_name_string,
                zero_level_index_str = zero_level_index_str,
                x_axis = 'Voxel index',
                axes_label_fontsize = 14,
                tick_label_fontsize = 12,
                title = f"{zero_level_index_str} Deltas — Patient {patient_id}, Bx {bx_index}, Bx ID {bx_id}",
                include_abs = True,
                require_precomputed_abs = True,
                fallback_recompute_abs = False,
                label_style = 'math',
                median_superscript = 'Q50',
                order_kinds = ('mean','mode','median'),
                show_points = False,
                point_size = 3,
                alpha = 0.5,
                linewidth_signed = 2.0,
                linewidth_abs = 2.0,
                show_title=False
            )



            biopsies = [(patient_id, 1), (patient_id, 2)]

            production_plots.plot_biopsy_deltas_line_multi(
                deltas_df=nominal_deltas_df_with_abs,
                biopsies=biopsies,
                save_dir=patient_dir,
                fig_name=f"{patient_id} - Bx1&2 - dosimetry-deltas-line-overlay-with-abs",
                zero_level_index_str='Dose (Gy)',
                x_axis='Voxel index',
                linewidth_signed=2.0,
                linewidth_abs=3.5,            # thicker dotted
                show_markers=True,           # flip True to encode biopsy by marker
                legend_fontsize = 12,
                include_abs = True,
            )

            production_plots.plot_biopsy_deltas_line_multi(
                deltas_df=nominal_deltas_df_with_abs,
                biopsies=biopsies,
                save_dir=patient_dir,
                fig_name=f"{patient_id} - Bx1&2 - dosimetry-deltas-line-overlay-no-abs",
                zero_level_index_str='Dose (Gy)',
                x_axis='Voxel index',
                linewidth_signed=2.0,
                linewidth_abs=3.5,            # thicker dotted
                show_markers=True,           # flip True to encode biopsy by marker
                legend_fontsize = 12,
                include_abs = False,
            )

            production_plots.plot_biopsy_deltas_line_multi(
                deltas_df=nominal_gradient_deltas_df_with_abs,
                biopsies=biopsies,
                save_dir=patient_dir,
                fig_name=f"{patient_id} - Bx1&2 - gradient-deltas-line-overlay-with-abs",
                zero_level_index_str='Dose grad (Gy/mm)',
                x_axis='Voxel index',
                linewidth_signed=2.0,
                linewidth_abs=3.5,
                show_markers=True,
                legend_fontsize = 12,
                include_abs = True,
            )

            production_plots.plot_biopsy_deltas_line_multi(
                deltas_df=nominal_gradient_deltas_df_with_abs,
                biopsies=biopsies,
                save_dir=patient_dir,
                fig_name=f"{patient_id} - Bx1&2 - gradient-deltas-line-overlay-no-abs",
                zero_level_index_str='Dose grad (Gy/mm)',
                x_axis='Voxel index',
                linewidth_signed=2.0,
                linewidth_abs=3.5,
                show_markers=True,
                legend_fontsize = 12,
                include_abs = False,
            )


            # --- Dose (Gy): Δ and |Δ| on same axes ---
            base = f"{patient_id} - {bx_id} - voxel-boxplot-dose-dual"
            production_plots.plot_biopsy_voxel_trial_boxplots_dual(
                deltas_df=mc_deltas,             # trial-level df with Δ columns
                patient_id=patient_id,
                bx_index=bx_index,
                output_dir=patient_dir,
                plot_name_base=base,
                metric='Dose (Gy)',
                x_axis='Voxel index',
                axes_label_fontsize=14,
                tick_label_fontsize=12,
                show_title=False,
                whis=(5, 95),                    # whiskers match IPR90 narrative
                showfliers=False,
                sort_voxels_by='median',
                show_points_signed=True,
                show_points_abs=True,
                point_size_signed=6,
                point_size_abs=6,
                point_alpha_signed=0.30,
                point_alpha_abs=0.30,
                require_precomputed_abs=True,
                fallback_recompute_abs=False,
                save_formats=('png', 'svg'),
            )

            # --- Dose grad (Gy/mm): Δ and |Δ| on same axes ---
            base = f"{patient_id} - {bx_id} - voxel-boxplot-grad-dual"
            production_plots.plot_biopsy_voxel_trial_boxplots_dual(
                deltas_df=mc_deltas,             # or gradient-specific df if stored separately
                patient_id=patient_id,
                bx_index=bx_index,
                output_dir=patient_dir,
                plot_name_base=base,
                metric='Dose grad (Gy/mm)',
                x_axis='Voxel index',
                axes_label_fontsize=14,
                tick_label_fontsize=12,
                show_title=False,
                whis=(5, 95),
                showfliers=False,
                sort_voxels_by='median',
                show_points_signed=True,
                show_points_abs=True,
                point_size_signed=6,
                point_size_abs=6,
                point_alpha_signed=0.30,
                point_alpha_abs=0.30,
                require_precomputed_abs=True,
                fallback_recompute_abs=False,
                save_formats=('png', 'svg'),
            )




            biopsies = [(patient_id, 1), (patient_id, 2)]
            patient_id = biopsies[0][0]
            multi = "Bx" + "&".join(str(bx) for _, bx in biopsies)

            # Dose
            production_plots.plot_voxel_dualboxes_by_biopsy_lanes(
                deltas_df=mc_deltas,
                biopsies=biopsies,
                output_dir=patient_dir,
                plot_name_base=f"{patient_id} - {multi} - voxel-boxplot-dose-dual",
                metric="Dose (Gy)",
                x_axis="Voxel index",
                lane_gap=2.0, box_width=0.32, pair_gap=0.10, biopsy_gap=0.22,
                show_points=False,
                whisker_mode='q05q95',
                showfliers=False,
                save_formats=("png","svg")
            )

            # Dose grad
            production_plots.plot_voxel_dualboxes_by_biopsy_lanes(
                deltas_df=mc_deltas,
                biopsies=biopsies,
                output_dir=patient_dir,
                plot_name_base=f"{patient_id} - {multi} - voxel-boxplot-grad-dual",
                metric="Dose grad (Gy/mm)",
                x_axis="Voxel index",
                lane_gap=2.0, box_width=0.32, pair_gap=0.10, biopsy_gap=0.22,
                show_points=False,
                whisker_mode='q05q95',
                showfliers=False,
                save_formats=("png","svg"),
            )





    print("--------------------------------------------------")
    print("Figures: Individual patient dosimetry and dose gradient kernel regressions...")
    print("--------------------------------------------------")




    if False:
        print("Skipping!")
    else:
        # 1. individual patient dosimetry and dose gradient kernel regressions
        """
        for patient_id, bx_index in patient_id_and_bx_index_pairs:
            # Create a directory for the patient
            patient_dir = pt_sp_figures_dir.joinpath(patient_id)
            os.makedirs(patient_dir, exist_ok=True)
            general_plot_name_string = " - dosimetry-kernel-regression"

            # this option determines how the trials are annotated in the plot
            random_trial_annotation_style = 'number' # can be 'number' or 'arrow'
            
            num_rand_trials_to_show = 3
            value_col_key = 'Dose (Gy)'
            y_axis_label = 'Dose (Gy)'
            custom_fig_title = 'Dosimetry Regression'
            
            sp_patient_all_structure_shifts_pandas_data_frame = all_mc_structure_transformation_df[all_mc_structure_transformation_df['Patient ID'] == patient_id]
            dose_output_nominal_and_all_MC_trials_pandas_data_frame = all_point_wise_dose_df[(all_point_wise_dose_df['Patient ID'] == patient_id) & (all_point_wise_dose_df['Bx index'] == bx_index)]
            dose_output_nominal_and_all_MC_trials_fully_voxelized_pandas_data_frame = all_voxel_wise_dose_df[(all_voxel_wise_dose_df['Patient ID'] == patient_id) & (all_voxel_wise_dose_df['Bx index'] == bx_index)]

            bx_struct_roi = cohort_biopsy_basic_spatial_features_df[(cohort_biopsy_basic_spatial_features_df['Patient ID'] == patient_id) & (cohort_biopsy_basic_spatial_features_df['Bx index'] == bx_index)]['Bx ID'].values[0]

            production_plots.production_plot_axial_dose_distribution_quantile_regression_by_patient_matplotlib(sp_patient_all_structure_shifts_pandas_data_frame,
                                                                                        dose_output_nominal_and_all_MC_trials_pandas_data_frame,
                                                                                        dose_output_nominal_and_all_MC_trials_fully_voxelized_pandas_data_frame,
                                                                                        patient_id,
                                                                                        bx_struct_roi,
                                                                                        bx_index,
                                                                                        bx_ref,
                                                                                        value_col_key,
                                                                                        patient_dir,
                                                                                        general_plot_name_string,
                                                                                        num_rand_trials_to_show,
                                                                                        y_axis_label,
                                                                                        custom_fig_title,
                                                                                        trial_annotation_style = random_trial_annotation_style)
            
            general_plot_name_string = " - dosimetry-gradient-kernel-regression"
            value_col_key = 'Dose grad (Gy/mm)'
            y_axis_label = 'Dose Gradient Norm (Gy/mm)'
            custom_fig_title = 'Dosimetry Gradient Regression'

            production_plots.production_plot_axial_dose_distribution_quantile_regression_by_patient_matplotlib(sp_patient_all_structure_shifts_pandas_data_frame,
                                                                                        dose_output_nominal_and_all_MC_trials_pandas_data_frame,
                                                                                        dose_output_nominal_and_all_MC_trials_fully_voxelized_pandas_data_frame,
                                                                                        patient_id,
                                                                                        bx_struct_roi,
                                                                                        bx_index,
                                                                                        bx_ref,
                                                                                        value_col_key,
                                                                                        patient_dir,
                                                                                        general_plot_name_string,
                                                                                        num_rand_trials_to_show,
                                                                                        y_axis_label,
                                                                                        custom_fig_title,
                                                                                        trial_annotation_style = random_trial_annotation_style)


        """

        for patient_id, bx_index in patient_id_and_bx_index_pairs:
            patient_dir = pt_sp_figures_dir.joinpath(patient_id)
            os.makedirs(patient_dir, exist_ok=True)

            random_trial_annotation_style = 'number'  # or 'arrow'
            num_rand_trials_to_show = 3

            sp_patient_all_structure_shifts_pandas_data_frame = \
                all_mc_structure_transformation_df[all_mc_structure_transformation_df['Patient ID'] == patient_id]

            dose_output_nominal_and_all_MC_trials_pandas_data_frame = \
                all_point_wise_dose_df[(all_point_wise_dose_df['Patient ID'] == patient_id) &
                                    (all_point_wise_dose_df['Bx index'] == bx_index)]

            dose_output_nominal_and_all_MC_trials_fully_voxelized_pandas_data_frame = \
                all_voxel_wise_dose_df[(all_voxel_wise_dose_df['Patient ID'] == patient_id) &
                                    (all_voxel_wise_dose_df['Bx index'] == bx_index)]

            bx_struct_roi = cohort_biopsy_basic_spatial_features_df[
                (cohort_biopsy_basic_spatial_features_df['Patient ID'] == patient_id) &
                (cohort_biopsy_basic_spatial_features_df['Bx index'] == bx_index)
            ]['Bx ID'].values[0]

            # --- Dose (Gy) ---
            production_plots.production_plot_axial_dose_distribution_quantile_regression_by_patient_matplotlib_v2(
                sp_patient_all_structure_shifts_pandas_data_frame,
                dose_output_nominal_and_all_MC_trials_pandas_data_frame,
                dose_output_nominal_and_all_MC_trials_fully_voxelized_pandas_data_frame,
                patient_id,
                bx_struct_roi,
                bx_index,
                bx_ref,
                value_col_key='Dose (Gy)',
                patient_sp_output_figures_dir=patient_dir,
                general_plot_name_string=" - dosimetry-kernel-regression",
                num_rand_trials_to_show=3,
                y_axis_label=r'Dose $(\mathrm{Gy})$',
                custom_fig_title="",                 # ignored when show_title=False
                trial_annotation_style=random_trial_annotation_style,
                axis_label_fontsize=16,
                tick_labelsize=14,
                show_x_direction_arrow=True,
                x_direction_label="To biopsy needle tip / patient superior",
                use_latex_labels=True,
                show_title=False                     # <-- turn off title
            )

            # --- Dose-gradient Magnitude (Gy mm^-1) ---
            production_plots.production_plot_axial_dose_distribution_quantile_regression_by_patient_matplotlib_v2(
                sp_patient_all_structure_shifts_pandas_data_frame,
                dose_output_nominal_and_all_MC_trials_pandas_data_frame,
                dose_output_nominal_and_all_MC_trials_fully_voxelized_pandas_data_frame,
                patient_id,
                bx_struct_roi,
                bx_index,
                bx_ref,
                value_col_key='Dose grad (Gy/mm)',
                patient_sp_output_figures_dir=patient_dir,
                general_plot_name_string=" - dosimetry-gradient-kernel-regression",
                num_rand_trials_to_show=3,
                y_axis_label=r'Dose-gradient Magnitude $(\mathrm{Gy}\ \mathrm{mm}^{-1})$',
                custom_fig_title="",
                trial_annotation_style=random_trial_annotation_style,
                axis_label_fontsize=16,
                tick_labelsize=14,
                show_x_direction_arrow=True,
                x_direction_label="To biopsy needle tip / patient superior",
                use_latex_labels=True,
                show_title=False                     # <-- turn off title
            )


    print("DONE!")
    print("--------------------------------------------------")
    print("Figures: Individual patient cumulative and differential DVH...")    
    print("--------------------------------------------------")


    # 2. individual patient cumulative and differential DVH
    if False:
        print("Skipping!")
    else:
        for patient_id, bx_index in patient_id_and_bx_index_pairs:
            # Create a directory for the patient
            patient_dir = pt_sp_figures_dir.joinpath(patient_id)
            os.makedirs(patient_dir, exist_ok=True)

            # global
            num_rand_trials_to_show = 3
            bx_struct_roi = cohort_biopsy_basic_spatial_features_df[(cohort_biopsy_basic_spatial_features_df['Patient ID'] == patient_id) & (cohort_biopsy_basic_spatial_features_df['Bx index'] == bx_index)]['Bx ID'].values[0]
            sp_patient_all_structure_shifts_pandas_data_frame = all_mc_structure_transformation_df[all_mc_structure_transformation_df['Patient ID'] == patient_id]


            ### cumulative DVH


            # options
            random_trial_annotation_style = 'number' # can be 'number' or 'arrow'
            general_plot_name_string = " - cumulative-DVH" # file name
            custom_fig_title = 'Cumulative DVH' # title of the plot


            sp_bx_cumulative_dvh_pandas_dataframe = all_cumulative_dvh_by_mc_trial_number_df[(all_cumulative_dvh_by_mc_trial_number_df['Patient ID'] == patient_id) & (all_cumulative_dvh_by_mc_trial_number_df['Bx index'] == bx_index)]

            dvh_option = {'dvh':'cumulative', 'x-col': 'Dose (Gy)', 'x-axis-label': 'Dose (Gy)', 'y-col': 'Percent volume', 'y-axis-label': 'Percent Volume (%)'}
            
            print(f"Creating cDVH for {patient_id}, bx index {bx_index}, bx ID {bx_struct_roi}...")
            """
            production_plots.production_plot_cumulative_or_differential_DVH_kernel_quantile_regression_NEW_v4_1(sp_patient_all_structure_shifts_pandas_data_frame,
                                                                                            sp_bx_cumulative_dvh_pandas_dataframe,
                                                                                                patient_dir,
                                                                                                patient_id,
                                                                                                bx_struct_roi,
                                                                                                bx_index,
                                                                                                bx_ref,
                                                                                                general_plot_name_string,
                                                                                                num_rand_trials_to_show,
                                                                                                custom_fig_title,
                                                                                                trial_annotation_style= random_trial_annotation_style,
                                                                                                dvh_option = dvh_option,
                                                                                                # NEW options
                                                                                                bands_mode="horizontal",           # 'vertical' (as in v3), 'horizontal' (Dx-consistent), or 'both'
                                                                                                show_median_line=True,
                                                                                                show_mean_line=False,
                                                                                                show_dx_vy_markers=True,
                                                                                                dx_list=(2, 50, 98),
                                                                                                vy_list=(100, 125, 150, 175, 200, 300),
                                                                                                ref_dose_gy=13.5
                                                                                                )
            """
            """
            production_plots.production_plot_cumulative_or_differential_DVH_kernel_quantile_regression_NEW_v4_3(sp_patient_all_structure_shifts_pandas_data_frame,
                                                                                            sp_bx_cumulative_dvh_pandas_dataframe,
                                                                                                patient_dir,
                                                                                                patient_id,
                                                                                                bx_struct_roi,
                                                                                                bx_index,
                                                                                                bx_ref,
                                                                                                general_plot_name_string,
                                                                                                num_rand_trials_to_show,
                                                                                                custom_fig_title,
                                                                                                trial_annotation_style= random_trial_annotation_style,
                                                                                                dvh_option = dvh_option,
                                                                                                # NEW options
                                                                                                bands_mode="both",           # 'vertical' (as in v3), 'horizontal' (Dx-consistent), or 'both'
                                                                                                show_median_line=True,
                                                                                                show_mean_line=False,
                                                                                                # tick/marker options
                                                                                                show_computed_ticks=True,          # draw ticks from trials (what you compare against)
                                                                                                show_table_markers=True,           # overlay markers from dvh_metrics_df
                                                                                                dx_list=(2, 50, 98),
                                                                                                vy_list=(100, 125, 150, 175, 200, 300),
                                                                                                ref_dose_gy=13.5,
                                                                                                # NEW: overlay table metrics
                                                                                                dvh_metrics_df=cohort_global_dosimetry_dvh_metrics_df,               # dataframe with columns shown in your message
                                                                                                overlay_metrics_stats=('Nominal','Q05','Q25','Q50','Q75','Q95'),  # which stats to plot
                                                                                                overlay_metrics_alpha=0.95,
                                                                                                # styles
                                                                                                dx_tick_color='black',
                                                                                                vy_tick_color='tab:blue',
                                                                                                )
            """
            # QA version
            production_plots.production_plot_cumulative_or_differential_DVH_kernel_quantile_regression_NEW_v4_5(
                sp_patient_all_structure_shifts_pandas_data_frame=sp_patient_all_structure_shifts_pandas_data_frame,
                cumulative_dvh_pandas_dataframe=sp_bx_cumulative_dvh_pandas_dataframe,
                patient_sp_output_figures_dir=patient_dir,
                patientUID=patient_id,
                bx_struct_roi=bx_struct_roi,
                bx_struct_ind=bx_index,
                bx_ref=bx_ref,
                general_plot_name_string="cumulative-DVH-QA",
                num_rand_trials_to_show=3,            # dashed sample trials (1..10)
                custom_fig_title="Cumulative DVH (QA)",
                trial_annotation_style='number',  # 'arrow' or 'number'
                dvh_option={'dvh':'cumulative', 'x-col':'Dose (Gy)', 'x-axis-label':'Dose (Gy)',
                            'y-col':'Percent volume','y-axis-label':'Percent Volume (%)'},
                bands_mode="both",                     # vertical + horizontal
                show_median_line=True,
                show_mean_line=False,
                show_ticks=True,                       # show computed Dx/Vy ticks
                show_markers=True,                     # overlay table markers
                show_dx_ticks=True,
                show_vy_ticks=True,
                show_dx_markers=True,
                show_vy_markers=True,
                dx_list=(2,50,98),
                vy_list=(100,125,150,175,200,300),
                ref_dose_gy=13.5,
                dvh_metrics_df=cohort_global_dosimetry_dvh_metrics_df,   # your table df
                overlay_metrics_stats=('Nominal','Q05','Q25','Q50','Q75','Q95'),
                # optional: constrain horizontal envelopes to common y-range across trials
                limit_horizontal_to_common=False,
                show_title=True,            
                axis_label_fontsize=16,      # x/y label size
                tick_label_fontsize=14,      # tick label size
            )
            # Final version
            production_plots.production_plot_cumulative_or_differential_DVH_kernel_quantile_regression_NEW_v4_5(
                sp_patient_all_structure_shifts_pandas_data_frame=sp_patient_all_structure_shifts_pandas_data_frame,
                cumulative_dvh_pandas_dataframe=sp_bx_cumulative_dvh_pandas_dataframe,
                patient_sp_output_figures_dir=patient_dir,
                patientUID=patient_id,
                bx_struct_roi=bx_struct_roi,
                bx_struct_ind=bx_index,
                bx_ref=bx_ref,
                general_plot_name_string="cumulative-DVH-paper",
                num_rand_trials_to_show=3,            # dashed sample trials (1..10)
                custom_fig_title="Cumulative DVH (Manuscript)",
                trial_annotation_style='number',  # 'arrow' or 'number'
                dvh_option={'dvh':'cumulative', 'x-col':'Dose (Gy)', 'x-axis-label':'Dose (Gy)',
                            'y-col':'Percent volume','y-axis-label':'Percent Volume (%)'},
                bands_mode="vertical",                     # vertical + horizontal
                quantile_line_style='smooth',  # 'smooth' or 'step'
                show_median_line=True,
                show_mean_line=False,
                show_ticks=False,                       # show computed Dx/Vy ticks
                show_markers=False,                     # overlay table markers
                show_dx_ticks=True,
                show_vy_ticks=True,
                show_dx_markers=True,
                show_vy_markers=True,
                dx_list=(2,50,98),
                vy_list=(100,125,150,175,200,300),
                ref_dose_gy=13.5,
                dvh_metrics_df=cohort_global_dosimetry_dvh_metrics_df,   # your table df
                overlay_metrics_stats=('Nominal','Q05','Q25','Q50','Q75','Q95'),
                # optional: constrain horizontal envelopes to common y-range across trials
                limit_horizontal_to_common=False,
                show_title=False,            # hide title entirely
                axis_label_fontsize=16,      # x/y label size
                tick_label_fontsize=14,      # tick label size
            )

            print('...done!')

            ### differential DVH
            if False:
                print(f'Creating dDVH for {patient_id}, bx index {bx_index}, bx ID {bx_struct_roi}...')
                random_trial_annotation_style = 'number' # can be 'number' or 'arrow'
                general_plot_name_string = " - differential-DVH" # file name
                custom_fig_title = 'Differential DVH' # title of the plot

                sp_bx_differential_dvh_pandas_dataframe = all_differential_dvh_by_mc_trial_number_df[(all_differential_dvh_by_mc_trial_number_df['Patient ID'] == patient_id) & (all_differential_dvh_by_mc_trial_number_df['Bx index'] == bx_index)]

                dvh_option = {'dvh':'differential', 'x-col': 'Dose bin center (Gy)', 'x-axis-label': 'Dose (Gy)', 'y-col': 'Percent volume', 'y-axis-label': 'Percent Volume (%)'}
                
                production_plots.production_plot_cumulative_or_differential_DVH_kernel_quantile_regression_NEW_v2(sp_patient_all_structure_shifts_pandas_data_frame,
                                                                                                sp_bx_differential_dvh_pandas_dataframe,
                                                                                                    patient_dir,
                                                                                                    patient_id,
                                                                                                    bx_struct_roi,
                                                                                                    bx_index,
                                                                                                    bx_ref,
                                                                                                    general_plot_name_string,
                                                                                                    num_rand_trials_to_show,
                                                                                                    custom_fig_title,
                                                                                                    trial_annotation_style=random_trial_annotation_style,
                                                                                                    dvh_option = dvh_option
                                                                                                    )
                
                print('...done!')




    print("DONE!")
    print("--------------------------------------------------")
    print("Figures: Individual patient effect sizes heatmaps...")    
    print("--------------------------------------------------")


    # 2. individual patient heatmaps
    if False:
        print("Skipping!")
    else:
        for patient_id, bx_index in patient_id_and_bx_index_pairs:
            # Create a directory for the patient
            patient_dir = pt_sp_figures_dir.joinpath(patient_id)
            os.makedirs(patient_dir, exist_ok=True)

            # global
            bx_struct_roi = cohort_biopsy_basic_spatial_features_df[(cohort_biopsy_basic_spatial_features_df['Patient ID'] == patient_id) & (cohort_biopsy_basic_spatial_features_df['Bx index'] == bx_index)]['Bx ID'].values[0]
            sp_patient_all_structure_shifts_pandas_data_frame = all_mc_structure_transformation_df[all_mc_structure_transformation_df['Patient ID'] == patient_id]



            eff_size_heatmaps_dir = patient_dir.joinpath(f"effect_sizes_heatmaps")
            os.makedirs(eff_size_heatmaps_dir, exist_ok=True)

            for eff_size in eff_sizes:
                eff_size_df = all_effect_sizes_df_dict[eff_size]
                # Filter the effect size dataframe for the specific patient and biopsy index
                eff_size_df = eff_size_df[(eff_size_df['Patient ID'] == patient_id) & (eff_size_df['Bx index'] == bx_index)]

                # Create a directory for the patient
                

                production_plots.plot_eff_size_heatmaps(eff_size_df, "Patient ID", "Bx index", "Bx ID", "Effect Size", eff_size, save_dir=eff_size_heatmaps_dir, save_name_base="dose")
                eff_size_df_dose_grad = all_effect_sizes_df_dose_grad_dict[eff_size]
                eff_size_df_dose_grad = eff_size_df_dose_grad[(eff_size_df_dose_grad['Patient ID'] == patient_id) & (eff_size_df_dose_grad['Bx index'] == bx_index)]
                production_plots.plot_eff_size_heatmaps(eff_size_df_dose_grad, "Patient ID", "Bx index", "Bx ID", "Effect Size", eff_size, save_dir=eff_size_heatmaps_dir, save_name_base="dose_gradient")

            # filter the mean difference stats dataframe for the specific patient and biopsy index
            mean_diff_stats_all_patients_sp_bx_df = mean_diff_stats_all_patients_df[(mean_diff_stats_all_patients_df['Patient ID'] == patient_id) & (mean_diff_stats_all_patients_df['Bx index'] == bx_index)]
            
            production_plots.plot_diff_stats_heatmaps_with_std(
                mean_diff_stats_all_patients_sp_bx_df,
                "Patient ID", 
                "Bx index", 
                "Bx ID",
                mean_col= "mean_diff",
                std_col = "std_diff",
                save_dir = eff_size_heatmaps_dir,
                save_name_base="dose",
                annotation_info = None,
                vmin= None,
                vmax= None
            )
            
            mean_diff_stats_all_patients_sp_bx_df_dose_grad = mean_diff_stats_all_patients_dose_grad_df[(mean_diff_stats_all_patients_dose_grad_df['Patient ID'] == patient_id) & (mean_diff_stats_all_patients_dose_grad_df['Bx index'] == bx_index)]
            production_plots.plot_diff_stats_heatmaps_with_std(
                mean_diff_stats_all_patients_sp_bx_df_dose_grad,
                "Patient ID", 
                "Bx index", 
                "Bx ID",
                mean_col= "mean_diff",
                std_col = "std_diff",
                save_dir = eff_size_heatmaps_dir,
                save_name_base="dose_gradient",
                annotation_info = None,
                vmin= None,
                vmax= None
            )

            patient_id_and_bx_index_pairs_for_paper = [('184 (F2)', 1), ('184 (F2)', 2)]
            if patient_id == patient_id_and_bx_index_pairs_for_paper[0][0] and bx_index == patient_id_and_bx_index_pairs_for_paper[0][1]:
                cell_box_fontsize = 8
                cell_box_fontsize_no_std = 10
            elif patient_id == patient_id_and_bx_index_pairs_for_paper[1][0] and bx_index == patient_id_and_bx_index_pairs_for_paper[1][1]:
                cell_box_fontsize = 7.5
                cell_box_fontsize_no_std = 9.5
            else:
                cell_box_fontsize = 7.5
                cell_box_fontsize_no_std = 9.5


            ### compute per-triangle vmin/vmax for symmetric colour scale centred at zero
            # Upper (dose)
            upper_vals = mean_diff_stats_all_patients_sp_bx_df["mean_diff"].to_numpy()
            # Lower (grad)
            lower_vals = mean_diff_stats_all_patients_sp_bx_df_dose_grad["mean_diff"].to_numpy()

            maxabs_upper = np.nanmax(np.abs(upper_vals))
            maxabs_lower = np.nanmax(np.abs(lower_vals))

            vmin_upper = -maxabs_upper
            vmax_upper =  maxabs_upper
            vmin_lower = -maxabs_lower
            vmax_lower =  maxabs_lower

            # If you want the SAME scale for both triangles:
            """
            maxabs = max(maxabs_upper, maxabs_lower)
            vmin_upper = vmin_lower = -maxabs
            vmax_upper = vmax_lower =  maxabs
            """




            # Upper = dose, Lower = dose gradient
            production_plots.plot_diff_stats_heatmap_upper_lower(
                upper_df=mean_diff_stats_all_patients_sp_bx_df,
                lower_df=mean_diff_stats_all_patients_sp_bx_df_dose_grad,   # your filtered grad DF
                patient_id_col="Patient ID",
                bx_index_col="Bx index",
                bx_id_col="Bx ID",
                upper_mean_col="mean_diff",
                upper_std_col="std_diff",
                lower_mean_col="mean_diff",
                lower_std_col="std_diff",
                save_dir=eff_size_heatmaps_dir,
                save_name_base="dose_upper__dosegrad_lower_signed",
                annotation_info=None,
                # global fallback limits (used only if per-triangle limits not provided)
                vmin = None,
                vmax = None,
                # OPTIONAL: per-triangle limits (take precedence if provided)
                # 🔹 explicit symmetric, zero-centred limits
                vmin_upper=vmin_upper,
                vmax_upper=vmax_upper,
                vmin_lower=vmin_lower,
                vmax_lower=vmax_lower,
                cmap="coolwarm",
                # typography
                tick_label_fontsize = 12,
                axis_label_fontsize = 14,
                cbar_tick_fontsize = 12,
                cbar_label_fontsize = 14,
                cbar_label_upper=r"$\overline{M_{b,ij}^{D}}$ ($\mathrm{Gy}$, Upper triangle)",
                cbar_label_lower=r"$\overline{M_{b,ij}^{G}}$ ($\mathrm{Gy}\,\mathrm{mm}^{-1}$, Lower triangle)",
                # title & corner annotation
                show_title = False,
                show_annotation_box = False,
                cell_annot_fontsize = cell_box_fontsize,
                cell_value_decimals = 1, 
            )
            # no std
            production_plots.plot_diff_stats_heatmap_upper_lower(
                upper_df=mean_diff_stats_all_patients_sp_bx_df,
                lower_df=mean_diff_stats_all_patients_sp_bx_df_dose_grad,   # your filtered grad DF
                patient_id_col="Patient ID",
                bx_index_col="Bx index",
                bx_id_col="Bx ID",
                upper_mean_col="mean_diff",
                upper_std_col=None,
                lower_mean_col="mean_diff",
                lower_std_col=None,
                save_dir=eff_size_heatmaps_dir,
                save_name_base="dose_upper__dosegrad_lower_signed_no_std",
                annotation_info=None,
                # global fallback limits (used only if per-triangle limits not provided)
                vmin = None,
                vmax = None,
                # OPTIONAL: per-triangle limits (take precedence if provided)
                # 🔹 explicit symmetric, zero-centred limits
                vmin_upper=vmin_upper,
                vmax_upper=vmax_upper,
                vmin_lower=vmin_lower,
                vmax_lower=vmax_lower,
                cmap="coolwarm",
                # typography
                tick_label_fontsize = 12,
                axis_label_fontsize = 14,
                cbar_tick_fontsize = 12,
                cbar_label_fontsize = 14,
                cbar_label_upper=r"$\overline{M_{b,ij}^{D}}$ ($\mathrm{Gy}$, Upper triangle)",
                cbar_label_lower=r"$\overline{M_{b,ij}^{G}}$ ($\mathrm{Gy}\,\mathrm{mm}^{-1}$, Lower triangle)",
                # title & corner annotation
                show_title = False,
                show_annotation_box = False,
                cell_annot_fontsize = cell_box_fontsize_no_std,
                cell_value_decimals = 1, 
            )

            # Upper = dose, Lower = dose gradient
            production_plots.plot_diff_stats_heatmap_upper_lower(
                upper_df=mean_diff_stats_all_patients_sp_bx_df,
                lower_df=mean_diff_stats_all_patients_sp_bx_df_dose_grad,   # your filtered grad DF
                patient_id_col="Patient ID",
                bx_index_col="Bx index",
                bx_id_col="Bx ID",
                upper_mean_col="mean_abs_diff",
                upper_std_col="std_abs_diff",
                lower_mean_col="mean_abs_diff",
                lower_std_col="std_abs_diff",
                save_dir=eff_size_heatmaps_dir,
                save_name_base="dose_upper__dosegrad_lower_absolute",
                annotation_info=None,
                # global fallback limits (used only if per-triangle limits not provided)
                vmin = None,
                vmax = None,
                # OPTIONAL: per-triangle limits (take precedence if provided)
                # 🔹 explicit symmetric, zero-centred limits
                vmin_upper=0.0,
                vmax_upper=None,
                vmin_lower=0.0,
                vmax_lower=None,
                cmap="Reds",
                # typography
                tick_label_fontsize = 14,
                axis_label_fontsize = 16,
                cbar_tick_fontsize = 14,
                cbar_label_fontsize = 16,
                cbar_label_upper=r"$\overline{|M_{b,ij}^{D}|}$ ($\mathrm{Gy}$, Upper triangle)",
                cbar_label_lower=r"$\overline{|M_{b,ij}^{G}|}$ ($\mathrm{Gy}\,\mathrm{mm}^{-1}$, Lower triangle)",
                # title & corner annotation
                show_title = False,
                show_annotation_box = False,
                cell_annot_fontsize = cell_box_fontsize,
                cell_value_decimals = 1, 
            )
            # without std
            production_plots.plot_diff_stats_heatmap_upper_lower(
                upper_df=mean_diff_stats_all_patients_sp_bx_df,
                lower_df=mean_diff_stats_all_patients_sp_bx_df_dose_grad,   # your filtered grad DF
                patient_id_col="Patient ID",
                bx_index_col="Bx index",
                bx_id_col="Bx ID",
                upper_mean_col="mean_abs_diff",
                upper_std_col=None,
                lower_mean_col="mean_abs_diff",
                lower_std_col=None,
                save_dir=eff_size_heatmaps_dir,
                save_name_base="dose_upper__dosegrad_lower_absolute_no_std",
                annotation_info=None,
                # global fallback limits (used only if per-triangle limits not provided)
                vmin = None,
                vmax = None,
                # OPTIONAL: per-triangle limits (take precedence if provided)
                # 🔹 explicit symmetric, zero-centred limits
                vmin_upper=0.0,
                vmax_upper=None,
                vmin_lower=0.0,
                vmax_lower=None,
                cmap="Reds",
                # typography
                tick_label_fontsize = 14,
                axis_label_fontsize = 16,
                cbar_tick_fontsize = 14,
                cbar_label_fontsize = 16,
                cbar_label_upper=r"$\overline{|M_{b,ij}^{D}|}$ ($\mathrm{Gy}$, Upper triangle)",
                cbar_label_lower=r"$\overline{|M_{b,ij}^{G}|}$ ($\mathrm{Gy}\,\mathrm{mm}^{-1}$, Lower triangle)",
                # title & corner annotation
                show_title = False,
                show_annotation_box = False,
                cell_annot_fontsize = cell_box_fontsize_no_std,
                cell_value_decimals = 1, 
            )
    


    # 3. individual patient dose differences voxel pairings of all length scales analysis
    print("--------------------------------------------------")
    print("Figures: Individual patient dose differences voxel pairings of all length scales analysis...")
    print("--------------------------------------------------")

    if False:
        print("Skipping!")
    else:
        for patient_id, bx_index in patient_id_and_bx_index_pairs:
            # Create a directory for the patient
            patient_dir = pt_sp_figures_dir.joinpath(patient_id)
            os.makedirs(patient_dir, exist_ok=True)

            # global
            bx_struct_roi = cohort_biopsy_basic_spatial_features_df[(cohort_biopsy_basic_spatial_features_df['Patient ID'] == patient_id) & (cohort_biopsy_basic_spatial_features_df['Bx index'] == bx_index)]['Bx ID'].values[0]


            dose_differences_cohort_df_patient = dose_differences_cohort_df[(dose_differences_cohort_df['Patient ID'] == patient_id) & (dose_differences_cohort_df['Bx index'] == bx_index)]

            production_plots.plot_dose_vs_length_with_summary(
                                    dose_differences_cohort_df_patient,
                                    'length_scale',
                                    'dose_diff_abs',
                                    save_dir = patient_dir,
                                    file_name = f"{patient_id}-{bx_struct_roi}-dose_differences_voxel_pairings_length_scales_strip_plot",
                                    title = f"{patient_id}-{bx_struct_roi} - Dose Differences Voxel Pairings Length Scales Strip Plot",
                                    figsize=(10, 6),
                                    dpi=300,
                                    show_points=False,
                                    violin_or_box='box',
                                    trend_lines = ['mean'],
                                    annotate_counts=True,
                                    y_trim=True,
                                    y_min_quantile=0.05,
                                    y_max_quantile=0.95,
                                    y_min_fixed=0,
                                    y_max_fixed=None,
                                    xlabel = "Length Scale (mm)",
                                    ylabel = "Absolute Dose Difference (Gy)",
                                )
            
            dose_differences_grad_cohort_df_patient = dose_differences_grad_cohort_df[(dose_differences_grad_cohort_df['Patient ID'] == patient_id) & (dose_differences_grad_cohort_df['Bx index'] == bx_index)]    
            production_plots.plot_dose_vs_length_with_summary(
                                    dose_differences_grad_cohort_df_patient,
                                    'length_scale',
                                    'dose_diff_abs',
                                    save_dir = patient_dir,
                                    file_name = f"{patient_id}-{bx_struct_roi}-dose_gradient_differences_voxel_pairings_length_scales_strip_plot",
                                    title = f"{patient_id}-{bx_struct_roi} - Dose Gradient Differences Voxel Pairings Length Scales Strip Plot",
                                    figsize=(10, 6),
                                    dpi=300,
                                    show_points=False,
                                    violin_or_box='box',
                                    trend_lines = ['mean'],
                                    annotate_counts=True,
                                    y_trim=True,
                                    y_min_quantile=0.05,
                                    y_max_quantile=0.95,
                                    y_min_fixed=0,
                                    y_max_fixed=None,
                                    xlabel = "Length Scale (mm)",
                                    ylabel = "Absolute Dose Gradient Difference (Gy/mm)",
                                )


        # multi boxplot example

        print('dose multi boxplot')
        patient_id_and_bx_index_pairs_for_multi_boxplot = [('184 (F2)', 1), ('184 (F2)', 2)]


        production_plots.plot_dose_vs_length_with_summary_mutlibox(
            dose_differences_cohort_df,
            'length_scale',
            'dose_diff_abs',
            save_dir=cohort_output_figures_dir,
            file_name="multi-patient-box",
            title=None,  # ✅ no title at all
            figsize=(10, 6),
            dpi=300,
            show_points=False,
            violin_or_box='box',
            trend_lines=['mean'],
            annotate_counts=True,
            annotation_box=False,
            y_trim=True,
            y_min_fixed=0,
            xlabel=None,
            ylabel=None,
            title_font_size=20,
            axis_label_font_size=16,
            tick_label_font_size=14,
            multi_pairs=patient_id_and_bx_index_pairs_for_multi_boxplot,
            metric_family='dose',           # <-- add this

        )
        """
        production_plots.plot_dose_vs_length_with_summary_mutlibox(
            dose_differences_cohort_df,
            'length_scale',
            'dose_diff_abs',
            save_dir=cohort_output_figures_dir,
            file_name="multi-patient-box-annotated",
            title=None,  # ✅ no title at all
            figsize=(10, 6),
            dpi=300,
            show_points=True,
            violin_or_box='box',
            trend_lines=['mean'],
            annotate_counts=True,
            annotation_box=True,
            y_trim=True,
            y_min_fixed=0,
            xlabel="Length Scale (mm)",
            ylabel="Absolute Dose Difference (Gy)",
            title_font_size=20,
            axis_label_font_size=14,
            tick_label_font_size=12,
            multi_pairs=patient_id_and_bx_index_pairs_for_multi_boxplot
        )

        production_plots.plot_dose_vs_length_with_summary_mutlibox(
            dose_differences_cohort_df,
            'length_scale',
            'dose_diff_abs',
            save_dir=cohort_output_figures_dir,
            file_name="multi-patient-violin",
            title=None,  # ✅ no title at all
            figsize=(10, 6),
            dpi=300,
            show_points=False,
            violin_or_box='violin',
            trend_lines=['mean'],
            annotate_counts=True,
            annotation_box=False,
            y_trim=True,
            y_min_fixed=0,
            xlabel="Length Scale (mm)",
            ylabel="Absolute Dose Difference (Gy)",
            title_font_size=20,
            axis_label_font_size=14,
            tick_label_font_size=12,
            multi_pairs=patient_id_and_bx_index_pairs_for_multi_boxplot
        )

        print('gradient multi boxplot')
        production_plots.plot_dose_vs_length_with_summary_mutlibox(
            dose_differences_grad_cohort_df,
            'length_scale',
            'dose_diff_abs',
            save_dir=cohort_output_figures_dir,
            file_name="multi-patient-gradient-violin",
            title=None,  # ✅ no title at all
            figsize=(10, 6),
            dpi=300,
            show_points=False,
            violin_or_box='violin',
            trend_lines=['mean'],
            annotate_counts=True,
            annotation_box=False,
            y_trim=True,
            y_min_fixed=0,
            xlabel="Length Scale (mm)",
            ylabel="Absolute Dose Difference (Gy)",
            title_font_size=20,
            axis_label_font_size=14,
            tick_label_font_size=12,
            multi_pairs=patient_id_and_bx_index_pairs_for_multi_boxplot
        )
        """
        production_plots.plot_dose_vs_length_with_summary_mutlibox(
            dose_differences_grad_cohort_df,
            'length_scale',
            'dose_diff_abs',
            save_dir=cohort_output_figures_dir,
            file_name="multi-patient-gradient-box",
            title=None,  # ✅ no title at all
            figsize=(10, 6),
            dpi=300,
            show_points=False,
            violin_or_box='box',
            trend_lines=['mean'],
            annotate_counts=True,
            annotation_box=False,
            y_trim=True,
            y_min_fixed=0,
            xlabel=None,
            ylabel=None,
            title_font_size=20,
            axis_label_font_size=16,
            tick_label_font_size=14,
            multi_pairs=patient_id_and_bx_index_pairs_for_multi_boxplot,
            metric_family='grad',           # <-- add this

        )

        












    # 4. individual patient dose ridgeline plots
    print("--------------------------------------------------")
    print("Figures: Individual patient dose ridgeline plots...")
    print("--------------------------------------------------")

    if False:
        print("Skipping!")
    else:
        for patient_id, bx_index in patient_id_and_bx_index_pairs:
            # Create a directory for the patient
            patient_dir = pt_sp_figures_dir.joinpath(patient_id)
            os.makedirs(patient_dir, exist_ok=True)

            # global
            bx_struct_roi = cohort_biopsy_basic_spatial_features_df[(cohort_biopsy_basic_spatial_features_df['Patient ID'] == patient_id) & (cohort_biopsy_basic_spatial_features_df['Bx index'] == bx_index)]['Bx ID'].values[0]

            #dose_output_nominal_and_all_MC_trials_fully_voxelized_pandas_data_frame = all_voxel_wise_dose_df[(all_voxel_wise_dose_df['Patient ID'] == patient_id) & (all_voxel_wise_dose_df['Bx index'] == bx_index)]
        

            sp_bx_all_point_wise_dose_df = all_point_wise_dose_df[(all_point_wise_dose_df['Patient ID'] == patient_id) & (all_point_wise_dose_df['Bx index'] == bx_index)]
            sp_bx_cohort_global_dosimetry_by_voxel_df = cohort_global_dosimetry_by_voxel_df[(cohort_global_dosimetry_by_voxel_df['Patient ID'] == patient_id) & (cohort_global_dosimetry_by_voxel_df['Bx index'] == bx_index)]


            svg_image_height = 1080
            svg_image_width = 1920
            cancer_tissue_label = 'DIL'
            dpi = 300
            production_plots.plot_dose_ridge_for_single_biopsy(
                                sp_bx_all_point_wise_dose_df,
                                sp_bx_cohort_global_dosimetry_by_voxel_df,
                                None,
                                patient_dir,
                                "Biopsy Voxel-Wise Dose Ridgeline Plot",
                                "dose",
                                cancer_tissue_label,
                                fig_scale=1.0,
                                dpi=300,
                                add_text_annotations=False,
                                x_label="Dose (Gy)",
                                y_label="Biopsy Axial Dimension (mm)",
                                space_between_ridgeline_padding_multiplier = 1.2,
                                ridgeline_vertical_padding_value = 0.5
                            )


if __name__ == "__main__":
    main()