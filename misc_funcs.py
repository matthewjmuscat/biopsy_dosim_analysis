def get_unique_patient_ids_fraction_prioritize(df, patient_id_col='Patient ID', priority_fraction='F1'):
    """
    Get unique patient IDs, prioritizing a specific fraction (e.g., F1).
    If the priority fraction is not present for a particular ID, take the next available fraction (e.g., F2).
    """
    # Get unique patient IDs
    unique_patient_ids = df[patient_id_col].unique()
    # Create a dictionary to store the base patient ID and its corresponding prioritized fraction
    patient_id_dict = {}
    # Loop through the unique patient IDs
    for patient_id in unique_patient_ids:
        # Extract the base patient ID (everything before 'F')
        base_id = patient_id.split('F')[0]
        # Check if the priority fraction is present or if another fraction should be added
        if priority_fraction in patient_id:
            patient_id_dict[base_id] = patient_id
        elif base_id not in patient_id_dict:
            # Add the first encountered fraction if the priority fraction is not found
            patient_id_dict[base_id] = patient_id
    # Return the list of unique patient IDs
    return list(patient_id_dict.values())

def get_unique_patient_ids_fraction_specific(df, patient_id_col='Patient ID',fraction='F1'):
    """
    Get unique patient IDs, prioritizing a specific fraction (e.g., F1).
    If the priority fraction is not present for a particular ID, take the next available fraction (e.g., F2).
    """
    # Get unique patient IDs
    unique_patient_ids = df[patient_id_col].unique()
    # Create a dictionary to store the base patient ID and its corresponding prioritized fraction
    patient_id_dict = {}
    # Loop through the unique patient IDs
    for patient_id in unique_patient_ids:
        # Extract the base patient ID (everything before 'F')
        base_id = patient_id.split('F')[0]
        # Check if the priority fraction is present or if another fraction should be added
        if fraction in patient_id:
            patient_id_dict[base_id] = patient_id
    # Return the list of unique patient IDs
    return list(patient_id_dict.values())