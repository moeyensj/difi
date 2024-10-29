import warnings
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype

__all__ = [
    "_checkColumnTypes",
    "_checkColumnTypesEqual",
    "_classHandler",
    "_percentHandler",
    "calcFirstFindableNight",
    "_firstFindableNightMinObs",
    "_firstFindableNightNightlyLinkages",
]


def _checkColumnTypes(df: pd.DataFrame, cols: List[str]):
    """
    Checks that each dataframe column listed in cols has Pandas dtype "Object".

    Parameters
    ----------
    df : `~pandas.DataFrame`
        Pandas dataframe
    cols : list
        Columns to check for appropriate data type.

    Raises
    ------
    TypeError : If any column is not of type "Object" or String .

    Returns
    -------
    None
    """
    error_text = ""
    for col in cols:
        if not is_object_dtype(df[col].dtype):
            error = (
                "\n{0} column should have type string. "
                "Please convert column using: \n"
                "dataframe['{0}'] = dataframe['{0}'].astype(str)`\n"
            )
            error = error.format(col)
            error_text += error

    if len(error_text) > 0:
        raise TypeError(error_text)
    return


def _checkColumnTypesEqual(df1: pd.DataFrame, df2: pd.DataFrame, cols: List[str]):
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

    Raises
    ------
    TypeError : If any column is not of type "Object" or String .

    Returns
    -------
    None
    """
    error_text = ""
    for col in cols:
        if not (df1[col].dtype == df2[col].dtype):
            error = (
                "\nColumn ('{0}') in the first data frame has type: {1}\n"
                "Column ('{0}') in the second data frame has type: {2}\n"
                "Please insure both columns have the same type!"
            )
            error = error.format(col, df1[col].dtype, df2[col].dtype)
            error_text += error

    if len(error_text) > 0:
        raise TypeError(error_text)
    return


def _classHandler(
    classes: Union[str, dict, None], dataframe: pd.DataFrame
) -> Tuple[List[str], List[List[str]]]:
    """
    Tests that the `classes` keyword argument is defined correctly.
    `classes` should one of the following:
        str : Name of the column in the dataframe which identifies
            the class of each object.
        dict : A dictionary with class names as keys and a list of unique
            objects belonging to each class as values.
        None : If there are no classes of objects.

    Parameters
    ----------
    classes : {str, dict, or None}
        Declares if the objects in a data frame have classes.
    dataframe : `~pandas.DataFrame`
        A pandas data frame containing a column of objects and optionally
        a column that specifies the class (if classes is a str).

    Returns
    -------
    class_list : list
        A list of class names.
    object_ids_list : list
        A list of the objects belonging to each class.

    Raises
    ------
    UserWarning : If not all objects in the dataframe are assigned a class
    """
    class_list = ["All"]
    object_ids_list = [[]]  # type: ignore
    unique_objects = []

    if classes is None:
        object_ids_list = [dataframe["object_id"].unique()]
        unique_objects = [object_ids_list[0]]

    elif isinstance(classes, str):
        if classes not in dataframe.columns:
            err = "Could not find class column ({}) in observations."
            raise ValueError(err.format(classes))
        else:
            for c in dataframe[~dataframe[classes].isna()][classes].unique():
                class_list.append(c)
                class_objects = dataframe[dataframe[classes].isin([c])]["object_id"].unique()
                unique_objects.append(class_objects)
                object_ids_list.append(class_objects)

        object_ids_list[0] = dataframe["object_id"].unique()

    elif isinstance(classes, dict):
        for c, t in classes.items():
            if len(np.unique(t)) != len(t):
                err = "Truths for class {} are not unique."
                raise ValueError(err.format(c))
            else:
                class_list.append(c)
                object_ids_list[0].append(t)
                unique_objects.append(t)
                if type(t) is list:
                    object_ids_list.append(np.array(t))
                else:
                    object_ids_list.append(t)

        object_ids_list[0] = np.hstack(object_ids_list[0])

    else:
        err = (
            "Classes should be one of:\n"
            "  str : Name of the column in the dataframe which\n"
            "        identifies the class of each object.\n"
            "  dict : A dictionary with class names as keys\n"
            "        and a list of unique objects belonging to each class\n"
            "        as values.\n"
            "  None : If there are no classes of objects."
        )
        raise ValueError(err)

    # Test that the unique objects are in fact unique
    unique_objects = np.concatenate(unique_objects)
    if not len(np.unique(unique_objects)) == len(unique_objects):
        err = "Some objects are duplicated in multiple classes."
        raise ValueError(err)

    if not dataframe["object_id"].isin(unique_objects).all():
        warning = (
            "Some objects do not have an assigned class.\n"
            "Unclassified objects have been added as 'Unclassified'."
        )

        unclassified = dataframe[~dataframe["object_id"].isin(unique_objects)]["object_id"].unique()
        class_list.append("Unclassified")
        object_ids_list.append(unclassified)
        object_ids_list[0] = np.concatenate([object_ids_list[0], unclassified])
        warnings.warn(warning, UserWarning)

    return class_list, object_ids_list


def _percentHandler(number: float, number_total: float) -> float:
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
        percent_total = 100.0 * number / number_total

    return percent_total


def _firstFindableNightMinObs(
    obs_ids: List[str],
    observations: pd.DataFrame,
    min_obs: int = 6,
) -> int:
    """
    For a particular findable object, find the first night on which it
    becomes findable when requiring a minimum of `min_obs` observations.

    Parameters
    ----------
    obs_ids : `list`
        List of observation IDs for this findable object
    observations : `DataFrame`
        Observations table
    min_obs : int, optional
        The minimum number of observations required for an object to be
        considered findable.

    Returns
    -------
    first_findable_night : `int`
        First night on which this object becomes findable
    """
    return observations.loc[obs_ids[min_obs - 1]]["night_id"]


def _firstFindableNightNightlyLinkages(
    obs_ids: List[str],
    observations: pd.DataFrame,
    min_linkage_nights: int = 3,
    detection_window: int = 15,
) -> int:
    """
    For a particular findable object, find the first night on which it
    becomes findable when requiring a minimum of `min_linkage_nights`
    nights of linkages within a `detection_window` night range.

    Parameters
    ----------
    obs_ids : `list`
        List of observation IDs for this findable object
    observations : `DataFrame`
        Observations table
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
    linkage_nights = np.unique(observations[np.isin(observations["obs_id"], obs_ids)]["night_id"].values)
    diff_nights = np.diff(linkage_nights)

    # compute the potential window sizes
    window_sizes = np.array(
        [
            sum(diff_nights[i : i + min_linkage_nights - 1])
            for i in range(len(diff_nights) - min_linkage_nights + 2)
        ]
    )

    # check if there are no matching ones just in case it *isn't* findable
    if not any(window_sizes <= detection_window):
        return -1
    else:
        # return the night corresponding to last night in the first valid window
        return linkage_nights[
            np.arange(len(window_sizes))[window_sizes <= detection_window][0] + min_linkage_nights - 1
        ]


def calcFirstFindableNight(
    findable_obs: pd.DataFrame, observations: pd.DataFrame, metric: str = "min_obs", **metric_kwargs
) -> pd.Series:
    """
    Calculate the first night on which an object becomes findable based
    on the `findable_obs` table returned by `analyzeObservations`.

    Parameters
    ----------
    observations : `~pandas.DataFrame`
        Pandas DataFrame with at least two columns: observation IDs and the object IDs
        (the object to which the observation belongs to).
    metric : {'min_obs', 'nightly_linkages'}
        The desired findability metric that calculates which objects are actually findable.
        If 'min_obs' [default]:
            Finds all objects with a minimum of min_obs observations and the observations
            that makes them findable.
            See `~difi.calcFindableMinObs` for more details.
        If 'nightly_linkages':
            Finds the objects that have at least min_linkage_nights linkages of length
            linkage_min_obs or more. Observations are considered to be in a possible intra-night
            linkage if their observation time does not exceed max_obs_separation.
            See `~difi.calcFindableNightlyLinkages` for more details.
    **metric_kwargs
        Any additional keyword arguments are passed to the desired findability metric.

    Returns
    -------
    first_findable_night : `pandas Series`
        New column for `findable_obs` representing the first night
        on which an object becomes findable
    """
    if metric == "min_obs":
        first_findable_night = findable_obs["obs_ids"].apply(
            _firstFindableNightMinObs, observations=observations, **metric_kwargs
        )
    elif metric == "nightly_linkages":
        first_findable_night = findable_obs["obs_ids"].apply(
            _firstFindableNightNightlyLinkages, observations=observations, **metric_kwargs
        )
    return first_findable_night
