import numpy as np
import pandas as pd

__all__ = ["_findNightlyLinkages", "calcFindableNightlyLinkages", "calcFindableMinObs"]


def _findNightlyLinkages(
    object_observations: pd.DataFrame,
    linkage_min_obs: int = 2,
    max_obs_separation: float = 1.5 / 24,
    min_linkage_nights: int = 3,
    column_mapping: dict[str, str] = {
        "obs_id": "obs_id",
        "time": "time",
        "night": "night",
    },
) -> np.ndarray:
    """
    Given observations belonging to one object, finds all observations that are within
    max_obs_separation of each other.

    Parameters
    ----------
    object_observations : `~pandas.DataFrame` or `~pandas.core.groupby.generic.DataFrameGroupBy`
        Pandas DataFrame with at least two columns for a single unique truth: observation IDs and the observation times
        in units of decimal days.
    linkage_min_obs : int, optional
        Minimum number of observations needed to make a intra-night
        linkage.
    max_obs_separation : float, optional
        Maximum temporal separation between two observations for them
        to be considered to be in a linkage (in the same units of decimal days).
        Maximum timespan between two observations.
    min_linkage_nights : int, optional
        Minimum number of nights on which a linkage should appear.
    column_mapping : dict, optional
        The mapping of columns in observations to internally used names.
        Needs the following: obs_id" : ..., "time" : ... , "night" : ....

    Returns
    -------
    linkage_obs : `~numpy.ndarray`
        Array of observation IDs that made the object findable.
    """
    # Grab times and observation IDs from grouped observations
    times = object_observations[column_mapping["time"]].values
    obs_ids = object_observations[column_mapping["obs_id"]].values
    nights = object_observations[column_mapping["night"]].values

    if linkage_min_obs > 1:
        # Calculate the time difference between observations
        # (assumes observations are sorted by ascending time)
        delta_t = times[1:] - times[:-1]

        # Create mask that selects all observations within max_obs_separation of
        # each other
        mask = delta_t <= max_obs_separation
        start_times = times[np.where(mask)[0]]
        end_times = times[np.where(mask)[0] + 1]

        # Combine times and select all observations match the linkage times
        linkage_times = np.unique(np.concatenate([start_times, end_times]))
        linkage_obs = obs_ids[np.isin(times, linkage_times)]
        linkage_nights, night_counts = np.unique(
            nights[np.isin(obs_ids, linkage_obs)], return_counts=True
        )

        # Make sure that the number of observations is still linkage_min_obs * min_linkage_nights
        enough_obs = len(linkage_obs) >= (linkage_min_obs * min_linkage_nights)

        # Make sure that the number of unique nights on which a linkage is made
        # is still equal to or greater than the minimum number of nights.
        enough_nights = (
            len(night_counts[night_counts >= linkage_min_obs]) >= min_linkage_nights
        )

        if not enough_obs or not enough_nights:
            return np.array([])

    else:
        linkage_obs = obs_ids

    return linkage_obs


def calcFindableNightlyLinkages(
    observations: pd.DataFrame,
    linkage_min_obs: int = 2,
    max_obs_separation: float = 1.5 / 24,
    min_linkage_nights: int = 3,
    column_mapping: dict[str, str] = {
        "obs_id": "obs_id",
        "truth": "truth",
        "time": "time",
        "night": "night",
    },
) -> pd.DataFrame:
    """
    Finds the truths that have at least min_linkage_nights linkages of length
    linkage_min_obs or more. Observations are considered to be in a possible intra-night
    linkage if their observation time does not exceed max_obs_separation.

    Parameters
    ----------
    observations : `~pandas.DataFrame`
        Pandas DataFrame with at least four columns: observation IDs, the truth values
        (the object to which the observation belongs to), the time of the observation
        in units of decimal days and the night of the observation.
    linkage_min_obs : int, optional
        Minimum number of observations needed to make a intra-night
        linkage.
    max_obs_separation : float, optional
        Maximum temporal separation between two observations for them
        to be considered to be in a linkage (in the same units of decimal days).
    min_linkage_nights : int, optional
        Minimum number of nights on which a linkage should appear.
    column_mapping : dict, optional
        The mapping of columns in observations to internally used names.
        Needs the following: "truth": ..., "obs_id" : ..., "time" : ..., "night" : ... .

    Returns
    -------
    findable : `~pandas.DataFrame`
        A `~pandas.DataFrame` with one column of the truth IDs that are findable, and a column named
        'obs_ids' containing `~numpy.ndarray`s of the observations that made each truth findable.
    """
    # Group by indivual object, then count number of observations on each night
    object_obs_per_night = (
        observations.groupby(column_mapping["truth"])[column_mapping["night"]]
        .value_counts()
        .to_frame("num_obs")
    )
    object_obs_per_night.reset_index(inplace=True)

    # Only keep objects that have equal to or more than the number of observations to make a linkage each night
    object_obs_per_night = object_obs_per_night[
        object_obs_per_night["num_obs"] >= linkage_min_obs
    ]

    # Now, group by individual object and count the number of nights during which a linkage can be made
    object_linkage_nights = (
        object_obs_per_night.groupby(column_mapping["truth"])[column_mapping["night"]]
        .nunique()
        .to_frame("num_linkage_nights")
    )
    object_linkage_nights.reset_index(inplace=True)

    # Only keep the objects that have enough linkages found on enough nights
    object_linkage_nights = object_linkage_nights[
        object_linkage_nights["num_linkage_nights"] >= min_linkage_nights
    ]
    possible_objects = object_linkage_nights[column_mapping["truth"]].unique()

    # Now, find which of the possible objects actually have observations in a linkage that meet the maximum time criterion
    track_observations = observations[
        observations[column_mapping["truth"]].isin(possible_objects)
    ]

    # If nothing is findable, return an empty dataframe
    if len(track_observations) > 0:
        findable_observations = (
            track_observations.groupby(by=column_mapping["truth"])
            .apply(
                _findNightlyLinkages,
                linkage_min_obs=linkage_min_obs,
                max_obs_separation=max_obs_separation,
                min_linkage_nights=min_linkage_nights,
                column_mapping=column_mapping,
            )
            .to_frame(name="obs_ids")
        )

        # Some observations may have been to far apart and been removed by the previous function, remove
        # the objects that are no longer findable
        findable_objects = (
            findable_observations["obs_ids"].apply(len).to_frame("num_obs")
        )
        findable_objects = findable_objects[
            findable_objects["num_obs"] != 0
        ].index.values

        findable = findable_observations[
            findable_observations.index.isin(findable_objects)
        ]
        findable.reset_index(inplace=True, drop=False)

    else:
        findable = pd.DataFrame(columns=[column_mapping["truth"], "obs_ids"])

    return findable


def calcFindableMinObs(
    observations: pd.DataFrame,
    min_obs: int = 5,
    column_mapping: dict[str, str] = {"truth": "truth", "obs_id": "obs_id"},
) -> pd.DataFrame:
    """
    Finds all truths with a minimum of min_obs observations and the observations
    that makes them findable.

    Parameters
    ----------
    observations : `~pandas.DataFrame`
        Pandas DataFrame with at least two columns: observation IDs and the truth values
        (the object to which the observation belongs to).
    min_obs : int, optional
        The minimum number of observations required for a truth to be considered
        findable.
    column_mapping : dict, optional
        The mapping of columns in observations to internally used names.
        Needs the following: "truth": ... and "obs_id" : ... .

    Returns
    -------
    findable : `~pandas.DataFrame`
        A `~pandas.DataFrame` with one column of the truth IDs that are findable, and a column named
        'obs_ids' containing `~numpy.ndarray`s of the observations that made each truth findable.
    """
    object_num_obs = (
        observations[column_mapping["truth"]].value_counts().to_frame("num_obs")
    )
    object_num_obs = object_num_obs[object_num_obs["num_obs"] >= min_obs]
    findable_objects = object_num_obs.index.values
    findable_observations = observations[
        observations[column_mapping["truth"]].isin(findable_objects)
    ]
    findable = (
        findable_observations.groupby(by=[column_mapping["truth"]])[
            column_mapping["obs_id"]
        ]
        .apply(np.array)
        .to_frame("obs_ids")
    )
    findable.reset_index(inplace=True, drop=False)
    return findable
