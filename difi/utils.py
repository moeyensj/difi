from pandas.api.types import is_object_dtype

__all__ = ["_checkColumnTypes"]


def _checkColumnTypes(df, cols, columnMapping):
    """
    Checks that each dataframe column listed in cols has Pandas dtype "Object".
    
    Parameters
    ----------
    df : `~pandas.DataFrame`
        Pandas dataframe with at least 
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

                   