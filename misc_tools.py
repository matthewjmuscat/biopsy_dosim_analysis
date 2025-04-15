import pandas

def tissue_heirarchy_list_tissue_names_creator_func(structs_referenced_dict,
                                       append_default_exterior_tissue = False,
                                       default_exterior_tissue = 'Periprostatic'):
    
    heirarchy_list = [value["Tissue class name"] for key, value in sorted(structs_referenced_dict.items(), 
                                                       key=lambda x: (x[1]['Tissue heirarchy'] is None, x[1]['Tissue heirarchy'])) 
                           if value.get('Tissue heirarchy') is not None]
    
    if append_default_exterior_tissue == True:
        heirarchy_list.append(default_exterior_tissue)
    
    return heirarchy_list


def convert_categorical_columns(df, columns, types):
    """
    Convert specified categorical columns in a DataFrame to given types if they are categorical.
    Non-categorical columns are silently skipped without conversion.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the columns.
        columns (list of str): List of column names to check for categorical type.
        types (list of type): List of types to convert the corresponding columns to if they are categorical. Can pass numpy types like np.int64, np.float64, etc. as well as str float or int etc.

    Returns:
        pd.DataFrame: The DataFrame with the specified columns converted if they were categorical.
    """
    if len(columns) != len(types):
        raise ValueError("The length of 'columns' and 'types' must be equal.")

    for column, dtype in zip(columns, types):
        # Check if the column dtype is an instance of pd.CategoricalDtype
        if isinstance(df[column].dtype, pandas.CategoricalDtype):
            df[column] = df[column].astype(dtype)

    return df