from pandas.api.types import is_object_dtype

__all__ = [
    "_checkColumnTypes",
    "_checkColumnTypesEqual"
]


def _checkColumnTypes(df, cols, columnMapping):
    """
    Checks that each dataframe column listed in cols has Pandas dtype "Object".
    
    Parameters
    ----------
    df : `~pandas.DataFrame`
        Pandas dataframe
    cols : list
        Columns to check for appropriate data type. 
    columnMapping : dict
        Column name mapping to internally used column names (truth, linkage_id, obs_id).
    
    Raises
    ------
    TypeError : If any column is not of type "Object" or String .
    
    Returns
    -------
    None
    """
    error_text = ""
    for col in cols:
        value = columnMapping[col]
        if not is_object_dtype(df[value].dtype):
            error = "\n{1} column ('{0}') should have type string. " \
                    "Please convert column using: \n" \
                    "dataframe['{0}'] = dataframe['{0}'].astype(str)`\n"
            error = error.format(value, col)
            error_text += error
    
    if len(error_text) > 0:
        raise TypeError(error_text)
    return

def _checkColumnTypesEqual(df1, df2, cols, columnMapping):
    """
    Checks that each column listed in cols have the same Pandas dtype in df1 and df2.
    
    Parameters
    ----------
    df1 : `~pandas.DataFrame`
        Pandas dataframe
    df2 : `~pandas.DataFrame`
        Pandas dataframe
    cols : list
        Columns to check for data type equality. 
    columnMapping : dict
        Column name mapping to internally used column names (truth, linkage_id, obs_id).
    
    Raises
    ------
    TypeError : If any column is not of type "Object" or String .
    
    Returns
    -------
    None
    """
    error_text = ""
    for col in cols:
        value = columnMapping[col]
        if not (df1[value].dtype == df2[value].dtype):
            error = "\n{1} column ('{0}') in the first data frame has type: {2}\n" \
                    "{1} column ('{0}') in the second data frame has type: {3}\n" \
                    "Please insure both columns have the same type!"
            error = error.format(value, col, df1[value].dtype, df2[value].dtype)
            error_text += error
    
    if len(error_text) > 0:
        raise TypeError(error_text)
    return

                   