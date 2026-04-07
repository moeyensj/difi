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
        test_linkage_members,
        all_objects,
        partition_summary=partition_summary,
        partition_mapping=partition_mapping,
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
            test_linkage_members,
            analyze_observations(empty_obs)[0],
            partition_summary=partition_summary,
            partition_mapping=partition_mapping,
            min_obs=6,
            contamination_percentage=20.0,
        )

    # More than one partition should raise
    partitions_multi = Partitions.create_linking_windows(test_observations.night, detection_window=5)
    partition_summary_multi = PartitionSummary.create(test_observations, partitions_multi)
    with pytest.raises(ValueError):
        _ = analyze_linkages(
            test_observations,
            test_linkage_members,
            analyze_observations(test_observations, partitions=partitions)[0],
            partition_summary=partition_summary_multi,
            partition_mapping=partition_mapping,
            min_obs=6,
            contamination_percentage=20.0,
        )

    return


def test_analyze_linkages_optional_kwargs(test_observations, test_linkage_members):
    # Explicit single-partition call
    partitions = Partitions.create_single(test_observations.night)
    partition_summary = PartitionSummary.create(test_observations, partitions)

    linkage_ids_unique = test_linkage_members.linkage_id.unique()
    partition_mapping = PartitionMapping.from_kwargs(
        linkage_id=linkage_ids_unique,
        partition_id=pa.repeat(partition_summary.id[0], len(linkage_ids_unique)),
    )

    all_objects, _, _ = analyze_observations(
        test_observations,
        partitions=partitions,
        metric="singletons",
        by_object=True,
        ignore_after_discovery=False,
        max_processes=1,
    )

    explicit_all_objects, explicit_all_linkages, explicit_summary = analyze_linkages(
        test_observations,
        test_linkage_members,
        all_objects,
        partition_summary=partition_summary,
        partition_mapping=partition_mapping,
        min_obs=6,
        contamination_percentage=50.0,
    )

    # Now omit both partition_summary and partition_mapping (they are optional)
    implicit_all_objects, implicit_all_linkages, implicit_summary = analyze_linkages(
        test_observations,
        test_linkage_members,
        all_objects,
        min_obs=6,
        contamination_percentage=50.0,
    )

    # Compare key properties
    assert len(explicit_all_linkages) == len(implicit_all_linkages)
    assert len(explicit_summary) == len(implicit_summary)
    # Object IDs and counts should match
    assert set(explicit_all_objects.object_id.to_pylist()) == set(implicit_all_objects.object_id.to_pylist())
