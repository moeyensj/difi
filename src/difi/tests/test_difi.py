import pyarrow as pa
import pyarrow.compute as pc
import pytest

from ..cifi import analyze_observations
from ..difi import PartitionMapping, analyze_linkages
from ..partitions import Partitions, PartitionSummary


def test_analyze_linkages_basic(test_observations, test_linkage_members):

    # Prepare all_objects and partition_summary via cifi (single partition with findable set)
    all_objects, _, partition_summary = analyze_observations(
        test_observations,
        partitions=None,
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

    # Run difi with a permissive contamination threshold
    all_objects_updated, all_linkages, partition_summaries_updated = analyze_linkages(
        test_observations,
        partition_summary,
        test_linkage_members,
        partition_mapping,
        all_objects,
        min_obs=6,
        contamination_percentage=50.0,
    )

    num_objects = len(test_observations.object_id.unique())

    # Validate linkage classifications
    assert len(all_linkages.select("pure", True)) == 2 * num_objects
    assert len(all_linkages.select("pure_complete", True)) == num_objects
    # With balanced mixed linkages, contaminated should be exactly num_objects
    assert len(all_linkages.select("contaminated", True)) == num_objects
    assert len(all_linkages.select("mixed", True)) == 5

    # Validate all_objects updates aggregate counts
    assert pc.sum(all_objects_updated.found_pure).as_py() == 2 * num_objects
    assert pc.sum(all_objects_updated.found_contaminated).as_py() == num_objects
    assert pc.sum(all_objects_updated.pure).as_py() == 2 * num_objects
    assert pc.sum(all_objects_updated.pure_complete).as_py() == num_objects
    assert pc.sum(all_objects_updated.contaminated).as_py() == num_objects
    assert pc.sum(all_objects_updated.mixed).as_py() >= 1

    # Partition summary updated
    assert len(partition_summaries_updated) == 1
    assert partition_summaries_updated.observations[0].as_py() == len(test_observations)

    return


def test_analyze_linkages_errors(test_observations, test_linkage_members):
    # Single partition
    partitions = Partitions.create_single(test_observations.night)
    partition_summary = PartitionSummary.create(test_observations, partitions)

    # Map linkage ids
    linkage_ids_unique = test_linkage_members.linkage_id.unique()
    partition_mapping = PartitionMapping.from_kwargs(
        linkage_id=linkage_ids_unique,
        partition_id=pa.repeat(partition_summary.id[0], len(linkage_ids_unique)),
    )

    # Non-overlapping observations and linkage_members should raise an error
    empty_obs = test_observations.select("id", "non-existent-id")
    with pytest.raises(ValueError):
        _ = analyze_linkages(
            empty_obs,
            partition_summary,
            test_linkage_members,
            partition_mapping,
            all_objects=analyze_observations(empty_obs)[0],
            min_obs=6,
            contamination_percentage=20.0,
        )

    return
