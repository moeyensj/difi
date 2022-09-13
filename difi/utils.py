import warnings
import numpy as np
from pandas.api.types import is_object_dtype

__all__ = [
    "_checkColumnTypes",
    "_checkColumnTypesEqual",
    "_classHandler",
    "_percentHandler",
    "calcFirstFindableNight"
]


def _checkColumnTypes(df, cols, column_mapping):
    """
    Checks that each dataframe column listed in cols has Pandas dtype "Object".
    
    Parameters
    ----------
    df : `~pandas.DataFrame`
        Pandas dataframe
    cols : list
        Columns to check for appropriate data type. 
    column_mapping : dict
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
        value = column_mapping[col]
        if not is_object_dtype(df[value].dtype):
            error = "\n{1} column ('{0}') should have type string. " \
                    "Please convert column using: \n" \
                    "dataframe['{0}'] = dataframe['{0}'].astype(str)`\n"
            error = error.format(value, col)
            error_text += error
    
    if len(error_text) > 0:
        raise TypeError(error_text)
    return

def _checkColumnTypesEqual(df1, df2, cols, column_mapping):
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
    column_mapping : dict
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
        value = column_mapping[col]
        if not (df1[value].dtype == df2[value].dtype):
            error = "\n{1} column ('{0}') in the first data frame has type: {2}\n" \
                    "{1} column ('{0}') in the second data frame has type: {3}\n" \
                    "Please insure both columns have the same type!"
            error = error.format(value, col, df1[value].dtype, df2[value].dtype)
            error_text += error
    
    if len(error_text) > 0:
        raise TypeError(error_text)
    return

def _classHandler(classes, dataframe, column_mapping):
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
    column_mapping : dict
        Column name mapping to internally used column names (truth, class).

    Returns
    -------
    class_list : list
        A list of class names. 
    truths_list : list
        A list of the truths belonging to each class. 

    Raises
    ------
    UserWarning : If not all truths in the dataframe are assigned a class
    """
    class_list = ["All"]
    truths_list = [[]]
    unique_truths = []

    if classes == None:
        truths_list = [dataframe[column_mapping["truth"]].unique()]
        unique_truths = [truths_list[0]]
    
    elif type(classes) == str:
        if classes not in dataframe.columns:
            err = (
                "Could not find class column ({}) in observations."
            )
            raise ValueError(err.format(classes))
        else:
            
            for c in dataframe[~dataframe[classes].isna()][classes].unique():
                class_list.append(c)
                class_truths = dataframe[dataframe[classes].isin([c])][column_mapping["truth"]].unique()
                unique_truths.append(class_truths)
                truths_list.append(class_truths)

        truths_list[0] = dataframe[column_mapping["truth"]].unique()
            
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
                unique_truths.append(t)
                if type(t) is list:
                    truths_list.append(np.array(t))
                else:
                    truths_list.append(t)

        truths_list[0] = np.hstack(truths_list[0])

    else:
        err = (
            "Classes should be one of:\n" \
            "  str : Name of the column in the dataframe which\n" \
            "        identifies the class of each truth.\n" \
            "  dict : A dictionary with class names as keys\n" \
            "        and a list of unique truths belonging to each class\n" \
            "        as values.\n" \
            "  None : If there are no classes of truths."
        )
        raise ValueError(err)

    # Test that the unique truths are in fact unique
    unique_truths = np.concatenate(unique_truths)
    if not len(np.unique(unique_truths)) == len(unique_truths):
        err = (
            "Some truths are duplicated in multiple classes."
        )
        raise ValueError(err)

    if not dataframe[column_mapping["truth"]].isin(unique_truths).all():
        warning = (
            "Some truths do not have an assigned class.\n" \
            "Unclassified truths have been added as 'Unclassified'."
        )

        unclassified = dataframe[~dataframe[column_mapping["truth"]].isin(unique_truths)][column_mapping["truth"]].unique()
        class_list.append("Unclassified")
        truths_list.append(unclassified)
        truths_list[0] = np.concatenate([truths_list[0], unclassified])
        warnings.warn(warning, UserWarning)
        
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


def firstFindableNightMinObs(obs_ids, observations,
                             column_mapping={
                                "obs_id": "obs_id",
                                "night_id": "night_id"
                             },
                             min_obs=6):
    """For a particular findable object, find the first night on which it
    becomes findable when requiring a minimum of `min_obs` observations.

    Parameters
    ----------
    obs_ids : `list`
        List of observation IDs for this findable object
    observations : `DataFrame`
        Observations table
    column_mapping : dict, optional
        The mapping of columns in observations to internally used names.
        Needs at least the following: "truth": ... and "obs_id" : ... . Other
        columns may be needed for different findability metrics.
    min_obs : int, optional
        The minimum number of observations required for a truth to be
        considered findable.

    Returns
    -------
    first_findable_night : `int`
        First night on which this object becomes findable
    """
    return observations.loc[obs_ids[min_obs - 1]][column_mapping["night_id"]]


def firstFindableNightNightlyLinkages(obs_ids, observations,
                                      column_mapping={
                                          "obs_id": "obs_id",
                                          "night_id": "night_id"
                                      },
                                      min_linkage_nights=3,
                                      detection_window=15):
    """For a particular findable object, find the first night on which it
    becomes findable when requiring a minimum of `min_linkage_nights`
    nights of linkages within a `detection_window` night range.

    Parameters
    ----------
    obs_ids : `list`
        List of observation IDs for this findable object
    observations : `DataFrame`
        Observations table
    column_mapping : dict, optional
        The mapping of columns in observations to internally used names.
        Needs at least the following: "truth": ... and "obs_id" : ... . Other
        columns may be needed for different findability metrics.
    min_linkage_nights : int, optional
        Minimum number of nights on which a linkage should appear.
    detection_window : int, optional
        Number of nights in the detection window in which a
        minimum of `min_linkage_nights` must occur

    Returns
    -------
    first_findable_night : `int`
        First night on which this object becomes findable
    """
    # get the linkage nights from the observations table
    linkage_nights = np.unique(observations[np.isin(observations[column_mapping["obs_id"]], obs_ids)][column_mapping["night_id"]].values)
    diff_nights = np.diff(linkage_nights)

    # compute the potential window sizes
    window_sizes = np.array([sum(diff_nights[i:i + min_linkage_nights - 1])
                            for i in range(len(diff_nights) - min_linkage_nights + 2)])

    # check if there are no matching ones just in case it *isn't* findable
    if not any(window_sizes <= detection_window):
        return -1
    else:
        # return the night corresponding to last night in the first valid window
        return linkage_nights[np.arange(len(window_sizes))[window_sizes <= detection_window][0] + min_linkage_nights - 1]


def calcFirstFindableNight(findable_obs, observations,
                           metric="min_obs",
                           column_mapping={
                               "obs_id": "obs_id",
                               "night_id": "night_id"
                           },
                           **metric_kwargs):
    """
    Calculate the first night on which a truth becomes findable based
    on the `findable_obs` table returned by `analyzeObservations`.

    Parameters
    ----------
    observations : `~pandas.DataFrame`
        Pandas DataFrame with at least two columns: observation IDs and the truth values
        (the object to which the observation belongs to).
    metric : {'min_obs', 'nightly_linkages'}
        The desired findability metric that calculates which truths are actually findable.
        If 'min_obs' [default]:
            Finds all truths with a minimum of min_obs observations and the observations
            that makes them findable.
            See `~difi.calcFindableMinObs` for more details.
        If 'nightly_linkages':
            Finds the truths that have at least min_linkage_nights linkages of length
            linkage_min_obs or more. Observations are considered to be in a possible intra-night
            linkage if their observation time does not exceed max_obs_separation.
            See `~difi.calcFindableNightlyLinkages` for more details.
    column_mapping : dict, optional
        The mapping of columns in observations to internally used names.
        Needs at least the following: "truth": ... and "obs_id" : ... . Other
        columns may be needed for different findability metrics.
    **metric_kwargs
        Any additional keyword arguments are passed to the desired findability metric. 
        Note that column_mapping is also passed to the findability metric.

    Returns
    -------
    first_findable_night : `pandas Series`
        New column for `findable_obs` representing the first night
        on which a truth becomes findable
        
    Raises
    ------
    TypeError : If the truth column in observations does not have type "Object"
    """
    if metric == "min_obs":
        first_findable_night = findable_obs["obs_ids"].apply(firstFindableNightMinObs, observations=observations, column_mapping=column_mapping, **metric_kwargs)
    elif metric == "nightly_linkages":
        first_findable_night = findable_obs["obs_ids"].apply(firstFindableNightNightlyLinkages, observations=observations, column_mapping=column_mapping, **metric_kwargs)
    return first_findable_night
