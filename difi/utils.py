import numpy as np
from pandas.api.types import is_object_dtype

__all__ = [
    "_checkColumnTypes",
    "_checkColumnTypesEqual",
    "_classHandler",
    "_percentHandler"
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

def _classHandler(classes, dataframe, columnMapping):
    """
    Tests that the `classes` keyword argument is defined correctly.
    `classes` should one of the following:
        str : Name of the column in the dataframe which identifies 
            the class of each truth.
        dict : A dictionary with class names as keys and a list of unique 
            truths belonging to each class as values.
        None : If there are no classes of truths.

    Parameters
    ----------
    classes : {str, dict, or None}
        Declares if the truths in a data frame have classes. 
    dataframe : `~pandas.DataFrame`
        A pandas data frame containing a column of truths and optionally
        a column that specifies the class (if classes is a str).
    columnMapping : dict
        Column name mapping to internally used column names (truth, class).

    Returns
    -------
    class_list : list
        A list of class names. 
    truths_list : list
        A list of the truths belonging to each class. 
    """
    class_list = ["all"]
    truths_list = [[]]
    
    if type(classes) == str:
        if classes not in dataframe.columns:
            err = (
                "Could not find class column ({}) in observations."
            )
            raise ValueError(err.format(classes))
        else:
            
            for c in dataframe[classes].unique():
                class_list.append(c)
                truths_list.append(dataframe[dataframe[c].isin([c])][columnMapping["truth"]].unique())
            
    elif type(classes) == dict:
        for c, t in classes.items():
            if len(np.unique(t)) != len(t):
                err = (
                    "Truths for class {} are not unique."
                )
                raise ValueError(err.format(c))
            else:
                class_list.append(c)
                truths_list[0].append(t)
                if type(t) is list:
                    truths_list.append(np.array(t))
                else:
                    truths_list.append(t)

        truths_list[0] = np.hstack(truths_list[0])

    elif classes == None:
        truths_list = [dataframe[columnMapping["truth"]].unique()]

    else:
        err = (
            "Classes should be one of:\n" \
            "  str : Name of the column in the dataframe which\n" \
            "    identifies the class of each truth.\n" \
            "  dict : A dictionary with class names as keys\n" \
            "    and a list of unique truths belonging to each class\n" \
            "    as values.\n" \
            "  None : If there are no classes of truths."
        )
        raise ValueError(err)
        
    return class_list, truths_list

def _percentHandler(number, number_total):
    """
    Returns a percentage value of number / number_total. Returns
    np.NaN is number total is zero. 

    Parameters
    ----------
    number : int or float
        Numerator (number out of number_total).
    number_total : int or float
        Denominator (total number).

    Returns
    -------
    percent : float
    """
    if number_total == 0:
        percent_total = np.NaN
    else:
        percent_total = 100. * number / number_total
    
    return percent_total
        