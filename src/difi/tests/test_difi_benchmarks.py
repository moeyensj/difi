import pyarrow as pa
import pytest

from ..cifi import analyze_observations
from ..difi import PartitionMapping, analyze_linkages
from ..partitions import Partitions


@pytest.mark.parametrize(
    ["min_obs", "contamination_percentage"],
    [
        (5, 0.0),
        (10, 0.0),
        (20, 0.0),
        (5, 30.0),
        (10, 30.0),
        (20, 30.0),
    ],
)
@pytest.mark.benchmark(group="analyze_linkages")
def test_benchmark_analyze_linkages_no_classes_no_all_objects(
    benchmark, test_observations, test_linkage_members, min_obs, contamination_percentage
):
    # Prepare all_objects and partition_summary via cifi
    partitions = Partitions.create_single(test_observations.night)
    all_objects, _, partition_summary = analyze_observations(
        test_observations,
        partitions=partitions,
        metric="singletons",
        by_object=True,
        ignore_after_discovery=False,
        max_processes=1,
    )

    # Map all linkage ids to this partition
    linkage_ids_unique = test_linkage_members.linkage_id.unique()
    partition_mapping = PartitionMapping.from_kwargs(
        linkage_id=linkage_ids_unique,
        partition_id=pa.repeat(partition_summary.id[0], len(linkage_ids_unique)),
    )

    # all_objects already prepared above

    # Benchmark analyze_linkages
    all_objects_updated, all_linkages, partition_summaries_updated = benchmark(
        analyze_linkages,
        test_observations,
        test_linkage_members,
        all_objects,
        partition_summary,
        partition_mapping,
        min_obs=min_obs,
        contamination_percentage=contamination_percentage,
    )

    return


@pytest.mark.parametrize(
    ["min_obs", "contamination_percentage"],
    [
        (5, 0.0),
        (10, 0.0),
        (20, 0.0),
        (5, 30.0),
        (10, 30.0),
        (20, 30.0),
    ],
)
@pytest.mark.benchmark(group="analyze_linkages")
def test_benchmark_analyze_linkages_no_classes_all_objects(
    benchmark, test_observations, test_linkage_members, min_obs, contamination_percentage
):
    # Prepare all_objects and partition_summary via cifi
    partitions = Partitions.create_single(test_observations.night)
    all_objects, _, partition_summary = analyze_observations(
        test_observations,
        partitions=partitions,
        metric="singletons",
        by_object=True,
        ignore_after_discovery=False,
        max_processes=1,
    )

    # Map all linkage ids to this partition
    linkage_ids_unique = test_linkage_members.linkage_id.unique()
    partition_mapping = PartitionMapping.from_kwargs(
        linkage_id=linkage_ids_unique,
        partition_id=pa.repeat(partition_summary.id[0], len(linkage_ids_unique)),
    )

    # all_objects already prepared above

    all_objects_updated, all_linkages, partition_summaries_updated = benchmark(
        analyze_linkages,
        test_observations,
        test_linkage_members,
        all_objects,
        partition_summary,
        partition_mapping,
        min_obs=min_obs,
        contamination_percentage=contamination_percentage,
    )

    return
