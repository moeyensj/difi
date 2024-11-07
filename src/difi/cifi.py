from typing import Optional, Tuple, TypeVar, Union

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
from .partitions import Partitions, PartitionSummary

__all__ = ["analyze_observations", "AllObjects"]


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


def analyze_observations(
    observations: Observations,
    partitions: Optional[Partitions] = None,
    metric: Union[str, Metrics] = "min_obs",
    discovery_opportunities: bool = False,
    discovery_probability: float = 1.0,
    by_object: bool = False,
    ignore_after_discovery: bool = False,
    max_processes: Optional[int] = 1,
    **metric_kwargs,
) -> Tuple[AllObjects, FindableObservations, PartitionSummary]:
    """
    Can I Find It?

    Parameters
    ----------
    observations : Observations
        Table of observations.
    partitions : Partitions, optional
        Table of partitions defining the start and end nights (both inclusive) of the partitions.
        If None, a single partition is created from the unique nights in the observations.
    metric : Union[str, Metrics], optional
        Metric to use to determine findability. If a string, the metric is looked up in the metric mapper.
        If a FindabilityMetric, the metric is used directly.
    discovery_opportunities : bool, optional
        Whether to include each discovery opportunity in the output.
    discovery_probability : float, optional
        The probability of a discovery opportunity actually resulting in a discovery.
    by_object : bool, optional
        Whether to calculate findability by object.
    ignore_after_discovery : bool, optional
        Whether to ignore observation that follow a discovery. Only used if by_object is True.
    max_processes : int, optional
        The maximum number of processes to use for parallelization.

    Returns
    -------
    all_objects : AllObjects
        Summary of all objects in the observations.
    findable_observations : FindableObservations
        Table of findable observations per partition.
    partition_summary : PartitionSummary
        Summary of the observations within each partition and details
        about the numbers of objects that are findable.
    """
    if len(observations) == 0:
        return AllObjects.empty(), FindableObservations.empty(), PartitionSummary.empty()

    if partitions is None:
        partitions = Partitions.create_single(observations.night)

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

    findable_observations = metric_.run(
        observations,
        partitions,
        discovery_opportunities=discovery_opportunities,
        discovery_probability=discovery_probability,
        by_object=by_object,
        ignore_after_discovery=ignore_after_discovery,
        max_processes=max_processes,
    )

    # Create the partition summary table
    partition_summary = PartitionSummary.create(observations, partitions)
    partition_summary = partition_summary.update_findable(findable_observations)

    # Create the AllObjects table
    all_objects = AllObjects.create(observations, findable_observations, partition_summary)

    return all_objects, findable_observations, partition_summary
