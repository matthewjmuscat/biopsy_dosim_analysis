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


def main():
    # This one is 10k containment and 10 (very low) dosim for speed, all vitesse patients, also 2.5^3 for dil, 2.5 only for OARs and bxs
    #main_output_path = Path("/home/matthew-muscat/Documents/UBC/Research/Data/Output data/MC_sim_out- Date-Apr-03-2025 Time-15,59,46")
    
    # This one is 10 (very low for speed) containment and 10k dosim, all vitesse patients, also 2.5^3 for dil, 2.5 only for OARs and bxs (not including variation in contouring - although this is negligible)
    main_output_path = Path("/home/matthew-muscat/Documents/UBC/Research/Data/Output data/MC_sim_out- Date-May-15-2025 Time-01,37,51")



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
    Patient ID	Structure ID	Simulated bool	Simulated type	Structure type	Structure ref num	Structure index	Dilation (XY)	Dilation (Z)	Rotation (X)	Rotation (Y)	Rotation (Z)	Shift (X)	Shift (Y)	Shift (Z)	Shift (z_needle)	Trial
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










    ### PLOTS FROM RAW DATA
    # make dirs
    output_fig_directory = output_dir.joinpath("figures")
    os.makedirs(output_fig_directory, exist_ok=True)
    cohort_output_figures_dir = output_fig_directory.joinpath("cohort_output_figures")
    os.makedirs(cohort_output_figures_dir, exist_ok=True)




    # 1. individual patient dosimetry and dose gradient kernel regressions
    patient_id_and_bx_index_pairs = [('181 (F2)',0), ('181 (F2)', 1), ('184 (F2)', 0), ('184 (F2)', 1), ('184 (F2)', 2), ('195 (F2)', 0), ('195 (F2)', 1), ('201 (F2)', 0),('201 (F2)', 1),('201 (F2)', 2)]
    for patient_id, bx_index in patient_id_and_bx_index_pairs:
        # Create a directory for the patient
        patient_dir = cohort_output_figures_dir.joinpath(patient_id)
        os.makedirs(patient_dir, exist_ok=True)
        general_plot_name_string = " - dosimetry-kernel-regression"
        
        num_rand_trials_to_show = 10
        value_col_key = 'Dose (Gy)'
        y_axis_label = 'Dose (Gy)'
        
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
                                                                                    y_axis_label)
        
        general_plot_name_string = " - dosimetry-gradient-kernel-regression"
        value_col_key = 'Dose grad (Gy/mm)'
        y_axis_label = 'Dose Gradient Norm (Gy/mm)'

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
                                                                                    y_axis_label)
        













if __name__ == "__main__":
    main()