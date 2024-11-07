from typing import Dict, List, Optional, Tuple, TypeVar, Union

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv

from .metrics import (
    FindabilityMetric,
    FindableObservations,
    MinObsMetric,
    NightlyLinkagesMetric,
)
from .observations import Observations
from .partitions import PartitionSummary
from .utils import _classHandler

__all__ = ["analyzeObservations"]


Metrics = TypeVar("Metrics", bound=FindabilityMetric)


class AllObjects(qv.Table):
    object_id = qv.LargeStringColumn()
    partition_id = qv.LargeStringColumn()
    mjd_min = qv.Float64Column()
    mjd_max = qv.Float64Column()
    arc_length = qv.Float64Column()
    num_obs = qv.Int64Column()
    num_observatories = qv.Int64Column()
    findable = qv.Int64Column(nullable=True)
    found = qv.Int64Column(nullable=True)
    found_pure = qv.Int64Column(nullable=True)
    found_partial = qv.Int64Column(nullable=True)
    pure = qv.Int64Column(nullable=True)
    pure_complete = qv.Int64Column(nullable=True)
    partial = qv.Int64Column(nullable=True)
    partial_contaminant = qv.Int64Column(nullable=True)
    mixed = qv.Int64Column(nullable=True)
    obs_in_pure = qv.Int64Column(nullable=True)
    obs_in_pure_complete = qv.Int64Column(nullable=True)
    obs_in_partial = qv.Int64Column(nullable=True)
    obs_in_partial_contaminant = qv.Int64Column(nullable=True)
    obs_in_mixed = qv.Int64Column(nullable=True)

    @classmethod
    def create(
        cls,
        observations: Observations,
        findable: FindableObservations,
        partition_summary: PartitionSummary,
    ) -> "AllObjects":
        """
        Create a summary of all objects in the observations subdivided by partitions.
        For each unique object in a partition, the number of observations, number of observatories,
        and the arc length of the observations are calculated.


        Parameters
        ----------
        observations : Observations
            Table of observations.
        findable : FindableObservations
            Table of findable observations per partition.
        partition_summary : PartitionSummary
            Table of partition summaries that defines the start and end nights of the partitions.

        Returns
        -------
        all_objects : AllObjects
            Summary of all objects in the observations.
        """
        all_objects = cls.empty()

        for partition in partition_summary:
            partition_id = partition.id[0].as_py()

            # Get the findable objects for this window
            findable_i = findable.select("partition_id", partition_id)

            observations_in_window = observations.filter_partition(partition)
            observations_in_window_table = observations_in_window.flattened_table().append_column(
                "mjd_utc", observations_in_window.time.rescale("utc").mjd()
            )

            num_obs_per_object = observations_in_window_table.group_by(
                ["object_id"], use_threads=False
            ).aggregate(
                [
                    ("object_id", "count", pc.CountOptions(mode="all")),
                    ("mjd_utc", "max", pc.CountOptions(mode="all")),
                    ("mjd_utc", "min", pc.CountOptions(mode="all")),
                    ("observatory_code", "count_distinct", pc.CountOptions(mode="all")),
                ]
            )

            all_objects_i = cls.from_kwargs(
                partition_id=pa.repeat(partition_id, len(num_obs_per_object)),
                object_id=num_obs_per_object["object_id"],
                num_obs=num_obs_per_object["object_id_count"],
                num_observatories=num_obs_per_object["observatory_code_count_distinct"],
                mjd_min=num_obs_per_object["mjd_utc_min"],
                mjd_max=num_obs_per_object["mjd_utc_max"],
                arc_length=pc.subtract(num_obs_per_object["mjd_utc_max"], num_obs_per_object["mjd_utc_min"]),
                findable=pc.is_in(num_obs_per_object["object_id"], findable_i.object_id.unique()),
            )

            all_objects = qv.concatenate([all_objects, all_objects_i])

        return all_objects.sort_by([("partition_id", "ascending"), ("num_obs", "descending")])


def _create_summary(
    observations: pd.DataFrame,
    classes: Union[None, str, Dict],
    all_objects: pd.DataFrame,
    window_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a summary dataframe that contains a summary of the number of members
    per class, the number of observations per class, and the number of findable.

    Parameters
    ----------
    observations : `~pandas.DataFrame`
        Pandas DataFrame with at least two columns: observation IDs and the object IDs
        (the object to which the observation belongs to).
    classes : dict
        Dictionary containing the classes and their members.
    all_objects : `~pandas.DataFrame`
        Pandas DataFrame containing the object ID values and the number of observations
        that make them findable.
    window_summary : `~pandas.DataFrame`
        Pandas DataFrame containing the window IDs and the start and end nights of the

    Returns
    -------
    summary : `~pandas.DataFrame`
        Pandas DataFrame containing the summary of the number of members per class,
        the number of observations per class, and the number of findable.
    """

    window_ids = window_summary["window_id"].unique()

    summary_dfs = []
    for window_id in window_ids:
        # Create masks for the dataframes
        window_i = window_summary[window_summary["window_id"] == window_id]
        all_objects_i = all_objects[all_objects["window_id"] == window_id]
        night_min = window_i["start_night"].values[0]
        night_max = window_i["end_night"].values[0]
        observations_in_window = observations[
            observations["night"].between(night_min, night_max, inclusive="both")
        ]

        num_observations_list: List[int] = []
        num_objects_list: List[int] = []
        num_findable_list: List[int] = []
        class_list, objects_list = _classHandler(classes, observations_in_window)

        for c, v in zip(class_list, objects_list):
            num_obs = len(observations_in_window[observations_in_window["object_id"].isin(v)])
            unique_objects = observations_in_window[observations_in_window["object_id"].isin(v)][
                "object_id"
            ].unique()
            num_unique_objects = len(unique_objects)
            findable = int(all_objects_i[all_objects_i["object_id"].isin(v)]["findable"].sum())

            num_observations_list.append(num_obs)
            num_objects_list.append(num_unique_objects)
            num_findable_list.append(findable)

        # Prepare summary DataFrame
        summary = pd.DataFrame(
            {
                "window_id": [window_id for _ in range(len(class_list))],
                "class": class_list,
                "num_members": num_objects_list,
                "num_obs": num_observations_list,
                "findable": num_findable_list,
            }
        )
        summary_dfs.append(summary)

    summary = pd.concat(summary_dfs, ignore_index=True)
    summary.sort_values(by=["window_id", "num_obs"], ascending=[True, False], inplace=True, ignore_index=True)
    return summary


def analyzeObservations(
    observations: pd.DataFrame,
    classes: Optional[dict] = None,
    metric: Union[str, Metrics] = "min_obs",
    detection_window: Optional[int] = None,
    discovery_opportunities: bool = False,
    ignore_after_discovery: bool = True,
    num_jobs: Optional[int] = 1,
    **metric_kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Can I Find It?

    Analyzes a DataFrame containing observations. These observations need at least two columns:
    i) the observation ID column
    ii) the object ID column

    Parameters
    ----------
    observations : `~pandas.DataFrame`
        Pandas DataFrame with at least two columns: observation IDs and the object ID values
        (the object to which the observation belongs to).
    metric : {'min_obs', 'nightly_linkages', callable}
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
        If callable:
            A user-defined function call also be passed, this function must return a `~pandas.DataFrame`
            with the object IDs that are findable as an index, and a column named
            'obs_ids' containing `~numpy.ndarray`s of the observations that made each object findable.
    classes : {dict, str, None}
        Analyze observations for objects grouped in different classes.
        str : Name of the column in the dataframe which identifies
            the class of each object.
        dict : A dictionary with class names as keys and a list of unique
            objects belonging to each class as values.
        None : If there are no classes of objects.
    detection_window : int, optional
        The number of days of observations to consider when
        determining if a object is findable. If the number of consecutive days
        of observations exceeds the detection_window, then a rolling window
        of size detection_window is used to determine if the object is findable.
        If None, then the detection_window is the entire range observations.
    ignore_after_discovery : bool, optional
        For use with `detection_window` - Whether to ignore an object in subsequent windows after it has been
        detected in an earlier window.
    num_jobs : int, optional
        The number of jobs to run in parallel. If 1, then run in serial. If None, then use the number of
        CPUs on the machine.
    **metric_kwargs
        Any additional keyword arguments are passed to the desired findability metric.

    Returns
    -------
    all_objects: `~pandas.DataFrame`
        A per-object summary.

        Columns:
            "object_id" : str
                Object ID
            "num_obs" : int
                Number of observations in the observations dataframe
                for each object
            "findable" : int
                1 if the object is findable, 0 if the object is not findable.
                (NaN if no findable column is found in the all_objects dataframe)

    findable_observations : `~pandas.DataFrame`
        A breakdown of the which observations made each object findable.
        Columns :
            "obs_ids" : `~numpy.ndarray`
                Observation IDs that made each object findable.
            "window_start_night" : int
                The start night of the window in which this object was detected. Note: column only exists if
                `detection_window` is not `None` and there are at least 2 potential detection windows.

    summary : `~pandas.DataFrame`
        A per-class summary.

        Columns:
            "class" : str
                Name of class (if none are defined, will only contain) "All".
            "num_members" : int
                Number of unique objects that belong to the class.
            "num_obs" : int
                Number of observations of objects belonging to the class in
                the observations dataframe.
            "findable" : int
                Number of objects deemed findable (all_objects must be passed to this
                function with a findable column)
    """
    # Raise error if there are no observations
    if len(observations) == 0:
        raise ValueError("There are no observations in the observations DataFrame!")

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

    findable_observations, window_summary = metric_.run(
        observations,
        detection_window=detection_window,
        discovery_opportunities=discovery_opportunities,
        ignore_after_discovery=ignore_after_discovery,
        num_jobs=num_jobs,
    )
    # Create the all objects dataframe
    all_objects = AllObjects.create(observations, findable_observations, window_summary)

    # Populate summary DataFrame
    summary = _create_summary(observations, classes, all_objects, window_summary)

    return all_objects, findable_observations, summary
