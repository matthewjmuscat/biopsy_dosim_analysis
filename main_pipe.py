import pandas as pd 
import load_files
from pathlib import Path
import os 
import statistical_tests_1_quick_and_dirty
import shape_and_radiomic_features
import misc_funcs
import biopsy_information
import uncertainties_analysis
import production_plots
import pickle
import pathlib # imported for navigating file system

def main():
    # This one is 10k containment and 10 (very low) dosim for speed, all vitesse patients, also 2.5^3 for dil, 2.5 only for OARs
    main_output_path = Path("/home/matthew-muscat/Documents/UBC/Research/Data/Output data/MC_sim_out- Date-Apr-03-2025 Time-15,59,46")




    ### Load master dicts results
    
    master_structure_info_dict_results = load_files.load_master_dict(main_output_path,
                                                        "master_structure_info_dict_results")
    




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











if __name__ == "__main__":
    main()