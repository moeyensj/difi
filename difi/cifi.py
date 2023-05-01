from typing import List, Optional, Tuple, TypeVar, Union

import numpy as np
import pandas as pd

from .metrics import FindabilityMetric, MinObsMetric, NightlyLinkagesMetric
from .utils import _classHandler

__all__ = ["analyzeObservations"]

Metrics = TypeVar("Metrics", bound=FindabilityMetric)


def _create_summary(observations, classes, all_truths):
    """
    Create a summary dataframe that contains a summary of the number of members
    per class, the number of observations per class, and the number of findable.

    Parameters
    ----------
    observations : `~pandas.DataFrame`
        Pandas DataFrame with at least two columns: observation IDs and the truth values
        (the object to which the observation belongs to).
    classes : dict
        Dictionary containing the classes and their members.
    all_truths : `~pandas.DataFrame`
        Pandas DataFrame containing the truth values and the number of observations
        that make them findable.

    Returns
    -------
    summary : `~pandas.DataFrame`
        Pandas DataFrame containing the summary of the number of members per class,
        the number of observations per class, and the number of findable.
    """
    num_observations_list: List[int] = []
    num_truths_list: List[int] = []
    num_findable_list: List[int] = []
    class_list, truths_list = _classHandler(classes, observations)

    for c, v in zip(class_list, truths_list):
        num_obs = len(observations[observations["truth"].isin(v)])
        unique_truths = observations[observations["truth"].isin(v)]["truth"].unique()
        num_unique_truths = len(unique_truths)
        findable = int(all_truths[all_truths["truth"].isin(v)]["findable"].sum())

        num_observations_list.append(num_obs)
        num_truths_list.append(num_unique_truths)
        num_findable_list.append(findable)

    # Prepare summary DataFrame
    summary = pd.DataFrame(
        {
            "class": class_list,
            "num_members": num_truths_list,
            "num_obs": num_observations_list,
            "findable": num_findable_list,
        }
    )
    summary.sort_values(by=["num_obs", "class"], ascending=False, inplace=True, ignore_index=True)

    return summary


def analyzeObservations(
    observations: pd.DataFrame,
    classes: Optional[dict] = None,
    metric: Union[str, Metrics] = "min_obs",
    detection_window: Optional[int] = None,
    ignore_after_detected: bool = True,
    **metric_kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Can I Find It?

    Analyzes a DataFrame containing observations. These observations need at least two columns:
    i) the observation ID column
    ii) the truth column

    Parameters
    ----------
    observations : `~pandas.DataFrame`
        Pandas DataFrame with at least two columns: observation IDs and the truth values
        (the object to which the observation belongs to).
    metric : {'min_obs', 'nightly_linkages', callable}
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
        If callable:
            A user-defined function call also be passed, this function must return a `~pandas.DataFrame`
            with the truth IDs that are findable as an index, and a column named
            'obs_ids' containing `~numpy.ndarray`s of the observations that made each truth findable.
    classes : {dict, str, None}
        Analyze observations for truths grouped in different classes.
        str : Name of the column in the dataframe which identifies
            the class of each truth.
        dict : A dictionary with class names as keys and a list of unique
            truths belonging to each class as values.
        None : If there are no classes of truths.
    detection_window : int, optional
        Number of nights within which the metric for detection must be met. Set `detection_window=None` to
        ignore this requirement.
    ignore_after_detected : bool, optional
        For use with `detection_window` - Whether to ignore an object in subsequent windows after it has been
        detected in an earlier window.
    **metric_kwargs
        Any additional keyword arguments are passed to the desired findability metric.

    Returns
    -------
    all_truths: `~pandas.DataFrame`
        A per-truth summary.

        Columns:
            "truth" : str
                Truth
            "num_obs" : int
                Number of observations in the observations dataframe
                for each truth
            "findable" : int
                1 if the object is findable, 0 if the object is not findable.
                (NaN if no findable column is found in the all_truths dataframe)

    findable_observations : `~pandas.DataFrame`
        A breakdown of the which observations made each object findable.
        Columns :
            "obs_ids" : `~numpy.ndarray`
                Observation IDs that made each truth findable.
            "window_start_night" : int
                The start night of the window in which this object was detected. Note: column only exists if
                `detection_window` is not `None` and there are at least 2 potential detection windows.

    summary : `~pandas.DataFrame`
        A per-class summary.

        Columns:
            "class" : str
                Name of class (if none are defined, will only contain) "All".
            "num_members" : int
                Number of unique truths that belong to the class.
            "num_obs" : int
                Number of observations of truths belonging to the class in
                the observations dataframe.
            "findable" : int
                Number of truths deemed findable (all_truths must be passed to this
                function with a findable column)

    Raises
    ------
    TypeError : If the truth column in observations does not have type "Object"
    """
    # Raise error if there are no observations
    if len(observations) == 0:
        raise ValueError("There are no observations in the observations DataFrame!")

    # Populate all_truths DataFrame
    dtypes = np.dtype([("truth", str), ("num_obs", int), ("findable", int)])
    data = np.empty(0, dtype=dtypes)
    all_truths = pd.DataFrame(data)

    num_obs_per_object = observations["truth"].value_counts().values
    num_obs_descending = observations["truth"].value_counts().index.values
    all_truths["truth"] = num_obs_descending
    all_truths["num_obs"] = num_obs_per_object

    metric_func_mapper = {
        "min_obs": MinObsMetric,
        "nightly_linkages": NightlyLinkagesMetric,
    }
    if isinstance(metric, str):
        if metric not in metric_func_mapper:
            raise ValueError(f"Unknown metric {metric}")
        metric_func = metric_func_mapper[metric]
        metric_ = metric_func(**metric_kwargs)
    elif isinstance(metric, FindabilityMetric):
        metric_ = metric
    else:
        raise ValueError("metric must be a string or a FindabilityMetric")

    findable_observations, window_summary = metric_.run(observations, detection_window=detection_window)

    all_truths.loc[:, "findable"] = 0
    all_truths.loc[all_truths["truth"].isin(findable_observations["truth"].values), "findable"] = 1

    all_truths["findable"] = all_truths["findable"].astype(int)
    all_truths.sort_values(by=["num_obs", "truth"], ascending=[False, True], inplace=True, ignore_index=True)

    # Populate summary DataFrame
    summary = _create_summary(observations, classes, all_truths)

    return all_truths, findable_observations, summary
