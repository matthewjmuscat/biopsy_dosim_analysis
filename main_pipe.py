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

def main():
    # This one is 10k containment and 10 (very low) dosim for speed, all vitesse patients, also 2.5^3 for dil, 2.5 only for OARs and bxs
    #main_output_path = Path("/home/matthew-muscat/Documents/UBC/Research/Data/Output data/MC_sim_out- Date-Apr-03-2025 Time-15,59,46")
    
    # This one is 10 (very low for speed) containment and 10k dosim, all vitesse patients, also 2.5^3 for dil, 2.5 only for OARs and bxs (not including variation in contouring - although this is negligible)
    main_output_path = Path("/home/matthew-muscat/Documents/UBC/Research/Data/Output data/MC_sim_out- Date-May-15-2025 Time-18,11,24")



    ### Load master dicts results
    
    master_structure_info_dict_results = load_files.load_master_dict(main_output_path,
                                                        "master_structure_info_dict_results")
    

    ### Define mimic structs ref dict

    bx_ref = "Bx ref"


    ### Load Dataframes 

    # Set csv directory
    csv_directory = main_output_path.joinpath("Output CSVs")
    cohort_csvs_directory = csv_directory.joinpath("Cohort")



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

    # Cohort bx dvh metrics
    cohort_global_dosimetry_dvh_metrics_path = cohort_csvs_directory.joinpath("Cohort: Bx DVH metrics (generalized).csv")  # Ensure the directory is a Path object
    cohort_global_dosimetry_dvh_metrics_df = load_files.load_csv_as_dataframe(cohort_global_dosimetry_dvh_metrics_path)
    """NOTE: The columns of the dataframe are:
    print(cohort_global_dosimetry_dvh_metrics_df.columns)
	Index(['Patient ID', 'Metric', 'Bx ID', 'Struct type', 'Dicom ref num',
		'Simulated bool', 'Simulated type', 'Struct index', 'Mean', 'STD',
		'SEM', 'Max', 'Min', 'Skewness', 'Kurtosis', 'Q05', 'Q25', 'Q50', 'Q75',
		'Q95'],
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


    ########### LOADING COMPLETE









    ## Create output directory
    # Output directory 
    output_dir = Path(__file__).parents[0].joinpath("output_data")
    os.makedirs(output_dir, exist_ok=True)




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






	### cumulatiove dil volume stats (START)
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
    ('Bx refnum',''),
    ('Simulated bool',''),
    ('Simulated type','')]

    # Get global dosimetry statistics
    summary_statistics.generate_summary_csv(global_dosimetry_dir, output_filename, cohort_global_dosimetry_df, col_pairs = None, exclude_columns = exclude_for_all)
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
    ('Bx refnum',''),
    ('Simulated bool',''),
    ('Simulated type','')]

    # Get global dosimetry by voxel statistics
    summary_statistics.generate_summary_csv(global_dosimetry_by_voxel_dir, output_filename, cohort_global_dosimetry_by_voxel_df, col_pairs = None, exclude_columns = exclude_for_all)
    ## Global dosimetry by voxel analysis (END)


    # Generate effect sizes dataframe

    ### Effect sizes analysis (START)
    print("--------------------------------------------------")
    print("Generating effect sizes analysis...")
    print("--------------------------------------------------")

    eff_sizes = ['cohen', 'hedges', 'mean_diff']
    all_effect_sizes_df_dict = {}
    for eff_size in eff_sizes:
        print(f"Calculating effect sizes for {eff_size}...")
        effect_size_dataframe = helper_funcs.create_eff_size_dataframe(all_voxel_wise_dose_df, "Patient ID", "Bx index", "Bx ID", "Voxel index", "Dose (Gy)", eff_size=eff_size, paired_bool=True)
        # Append the effect size dataframe to the dictionary
        all_effect_sizes_df_dict[eff_size] = effect_size_dataframe

    
    # Save the effect sizes dataframe to a CSV file
    # Create output directory for effect size analysis
    effect_sizes_analysis_dir = output_dir.joinpath("effect_sizes_analysis")
    os.makedirs(effect_sizes_analysis_dir, exist_ok=True)
    general_output_filename = 'effect_sizes_statistics_all_patients.csv'
    for eff_size in eff_sizes:
        effect_size_dataframe = all_effect_sizes_df_dict[eff_size] 
        effect_size_dataframe.to_csv(effect_sizes_analysis_dir.joinpath(f"{general_output_filename}_{eff_size}.csv"), index=False)

    
    ### Effect sizes analysis (END)




    # Generate dose differences voxel pairings of all length scales analysis

    ### Dose differences voxel pairings of all length scales analysis (START)
    print("--------------------------------------------------")
    print("Generating dose differences voxel pairings of all length scales analysis...")
    print("--------------------------------------------------")

    dose_differences_cohort_df = helper_funcs.compute_dose_differences_vectorized(all_voxel_wise_dose_df)


    ### Dose differences voxel pairings of all length scales analysis (END)




















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


    # make dirs
    output_fig_directory = output_dir.joinpath("figures")
    os.makedirs(output_fig_directory, exist_ok=True)
    cohort_output_figures_dir = output_fig_directory.joinpath("cohort_output_figures")
    os.makedirs(cohort_output_figures_dir, exist_ok=True)
    pt_sp_figures_dir = output_fig_directory.joinpath("patient_specific_output_figures")
    os.makedirs(pt_sp_figures_dir, exist_ok=True)







    print("--------------------------------------------------")
    print("Figures: Cohort figures...")
    print("--------------------------------------------------")

    if True:
        print("Skipping!")
    else:

        # 1. all voxels histograms dosimetry and gradient

        #dists_to_try = ['lognorm', 'gamma', 'gengamma', 'weibull_min', 'skewnorm'] # most likely correct
        #dists_to_try = None # try all
        dists_to_try = ['lognorm'] # lognorm is the best fit for most of the data, so we will use this for now
        xrange_dose = (0, 100)  # Adjust the range as needed
        xrange_dose_grad = (0, 50)  # Adjust the range as needed
        production_plots.histogram_and_fit(all_voxel_wise_dose_df, dists_to_try = dists_to_try, bin_size = 1, dose_col="Dose (Gy)", save_path = cohort_output_figures_dir, custom_name = "histogram_fit_all_voxels_dose", xrange = xrange_dose, vertical_gridlines= True, horizontal_gridlines=True)

        production_plots.histogram_and_fit(all_voxel_wise_dose_df, dists_to_try = dists_to_try, bin_size = 1, dose_col="Dose grad (Gy/mm)", save_path = cohort_output_figures_dir, custom_name = "histogram_fit_all_voxels_dose_gradient", xrange = xrange_dose_grad, vertical_gridlines= True, horizontal_gridlines=True)

        # 1. DONE



        # 2. Effect size heatmaps
        print("Generating effect size heatmaps...")

        eff_size_heatmaps_dir = cohort_output_figures_dir.joinpath(f"effect_sizes_heatmaps")
        os.makedirs(eff_size_heatmaps_dir, exist_ok=True)
        
        for eff_size in eff_sizes:
        
            effect_size_dataframe = all_effect_sizes_df_dict[eff_size] 
            for agg_abs in [False, True]:

                
                production_plots.plot_cohort_eff_size_heatmap_boxed_counts(effect_size_dataframe,
                                                "Effect Size",
                                                eff_size,
                                                save_path_base=eff_size_heatmaps_dir,
                                                annotation_info=None,
                                                aggregate_abs=agg_abs,
                                                vmin=None,
                                                vmax=None)

            
            #production_plots.plot_eff_size_heatmaps(effect_size_dataframe, "Patient ID", "Bx index", "Bx ID", "Effect Size", eff_size, save_dir=eff_size_heatmaps_dir)

        # 2. DONE



        # 3. Cohort DVH metrics boxplot
        print("Generating cohort DVH metrics boxplot...")

        production_plots.dvh_boxplot(cohort_global_dosimetry_dvh_metrics_df, save_path = cohort_output_figures_dir, custom_name = "dvh_boxplot")




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
    print("Figures: Individual patient dosimetry and dose gradient kernel regressions...")
    print("--------------------------------------------------")


    if True:
        print("Skipping!")
    else:
        # 1. individual patient dosimetry and dose gradient kernel regressions
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

    print("DONE!")
    print("--------------------------------------------------")
    print("Figures: Individual patient cumulative and differential DVH...")    
    print("--------------------------------------------------")


    # 2. individual patient cumulative and differential DVH
    if True:
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
            
            production_plots.production_plot_cumulative_or_differential_DVH_kernel_quantile_regression_NEW_v2(sp_patient_all_structure_shifts_pandas_data_frame,
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
                                                                                                dvh_option = dvh_option
                                                                                                )


            ### differential DVH
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
            





    print("DONE!")
    print("--------------------------------------------------")
    print("Figures: Individual patient effect sizes heatmaps...")    
    print("--------------------------------------------------")


    # 2. individual patient cumulative and differential DVH
    if True:
        print("Skipping!")
    else:
        for patient_id, bx_index in patient_id_and_bx_index_pairs:
            # Create a directory for the patient
            patient_dir = pt_sp_figures_dir.joinpath(patient_id)
            os.makedirs(patient_dir, exist_ok=True)

            # global
            bx_struct_roi = cohort_biopsy_basic_spatial_features_df[(cohort_biopsy_basic_spatial_features_df['Patient ID'] == patient_id) & (cohort_biopsy_basic_spatial_features_df['Bx index'] == bx_index)]['Bx ID'].values[0]
            sp_patient_all_structure_shifts_pandas_data_frame = all_mc_structure_transformation_df[all_mc_structure_transformation_df['Patient ID'] == patient_id]


            ### cumulative DVH


            # options
            random_trial_annotation_style = 'number' # can be 'number' or 'arrow'
            general_plot_name_string = " - cumulative-DVH" # file name
            custom_fig_title = 'Cumulative DVH' # title of the plot

            eff_size_heatmaps_dir = patient_dir.joinpath(f"effect_sizes_heatmaps")
            os.makedirs(eff_size_heatmaps_dir, exist_ok=True)

            for eff_size in eff_sizes:
                eff_size_df = all_effect_sizes_df_dict[eff_size]
                # Filter the effect size dataframe for the specific patient and biopsy index
                eff_size_df = eff_size_df[(eff_size_df['Patient ID'] == patient_id) & (eff_size_df['Bx index'] == bx_index)]

                # Create a directory for the patient
                

                production_plots.plot_eff_size_heatmaps(eff_size_df, "Patient ID", "Bx index", "Bx ID", "Effect Size", eff_size, save_dir=eff_size_heatmaps_dir)

            

    # 3. individual patient dose differences voxel pairings of all length scales analysis
    print("--------------------------------------------------")
    print("Figures: Individual patient dose differences voxel pairings of all length scales analysis...")
    print("--------------------------------------------------")

    if True:
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