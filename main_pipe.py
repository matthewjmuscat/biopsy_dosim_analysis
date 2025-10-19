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
	Index(['Patient ID', 'Metric', 'Bx ID', 'Struct type', 'Dicom ref num',
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
    Index(['Patient ID', 'Metric', 'Bx ID', 'Struct type', 'Dicom ref num',
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
       'Simulated type', 'Bx refnum', 'Bx ID', 'D_2% (Gy)', 'D_50% (Gy)',
       'D_98% (Gy)', 'V_100% (%)', 'V_125% (%)', 'V_150% (%)', 'V_175% (%)',
       'V_200% (%)', 'V_300% (%)'],
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
        Index(['Patient ID', 'Metric', 'Bx ID', 'Struct type', 'Dicom ref num',
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
                                                       'Bx refnum',
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
                (           'Bx refnum',                       ''),
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
                (                   'Bx refnum',                       ''),
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
            'Patient ID', 'Bx ID', 'Bx index', 'Bx refnum',
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
        'Bx ID_x', 'Bx index', 'Bx refnum_x', 'Simulated bool_x',
        'Simulated type_x', 'Δ_mode (Gy)', 'Δ_median (Gy)', 'Δ_mean (Gy)',
        '|Δ_mode| (Gy)', '|Δ_median| (Gy)', '|Δ_mean| (Gy)',
        'Voxel begin (Z)_y', 'Voxel end (Z)_y', 'Bx ID_y', 'Bx refnum_y',
        'Simulated bool_y', 'Simulated type_y', 'Grad[nominal] (Gy/mm)',
        'Grad[median] (Gy/mm)', 'Grad[mean] (Gy/mm)', 'Grad[mode] (Gy/mm)',
        'log1p|Δ_mode| (Gy)', 'log1p|Δ_median| (Gy)', 'log1p|Δ_mean| (Gy)'],
        dtype='object')
    """
    
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
    ('Bx refnum',''),
    ('Simulated bool',''),
    ('Simulated type','')]


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
    ('Bx refnum',''),
    ('Simulated bool',''),
    ('Simulated type','')]

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
	Index(['Patient ID', 'Metric', 'Bx ID', 'Struct type', 'Dicom ref num',
		'Simulated bool', 'Simulated type', 'Struct index', 'Mean', 'STD',
		'SEM', 'Max', 'Min', 'Skewness', 'Kurtosis', 'Q05', 'Q25', 'Q50', 'Q75',
		'Q95'],
		dtype='object')
    """

    # 2) which columns to carry along (but *not* summarize)
    exclude = [
        'Patient ID','Bx ID','Struct type','Dicom ref num',
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
                vmin_upper = None,
                vmax_upper = None,
                vmin_lower = None,
                vmax_lower = None,
                # typography
                tick_label_fontsize = 12,
                axis_label_fontsize = 14,
                cbar_tick_fontsize = 12,
                cbar_label_fontsize = 14,
                cbar_label_upper=r"Mean of $M_{b,ij}^{D,(t)}$ ($\mathrm{Gy}$, Upper triangle)",
                cbar_label_lower=r"Mean of $M_{b,ij}^{G,(t)}$ ($\mathrm{Gy}\,\mathrm{mm}^{-1}$, Lower triangle)",
                # title & corner annotation
                show_title = False,
                show_annotation_box = False,
                cell_annot_fontsize = 8
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
                vmin_upper = None,
                vmax_upper = None,
                vmin_lower = None,
                vmax_lower = None,
                # typography
                tick_label_fontsize = 14,
                axis_label_fontsize = 16,
                cbar_tick_fontsize = 14,
                cbar_label_fontsize = 16,
                cbar_label_upper=r"Mean of $|M_{b,ij}^{D,(t)}|$ ($\mathrm{Gy}$, Upper triangle)",
                cbar_label_lower=r"Mean of $|M_{b,ij}^{G,(t)}|$ ($\mathrm{Gy}\,\mathrm{mm}^{-1}$, Lower triangle)",
                # title & corner annotation
                show_title = False,
                show_annotation_box = False,
                cell_annot_fontsize = 8
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