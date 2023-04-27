import numpy as np
import pytest

from ..difi import analyzeLinkages


def assert_valid_pure_linkage(linkage_id, all_linkages, linkage_members, observations, pure_complete=False):
    """
    Checks that the given linkage is correctly identified as a pure linkage.
    """
    linkage = all_linkages[all_linkages["linkage_id"] == linkage_id]
    assert len(linkage) == 1
    members = linkage_members[linkage_members["linkage_id"] == linkage_id]
    observations_linkage = observations[observations["obs_id"].isin(members["obs_id"])]

    assert observations_linkage["truth"].nunique() == 1
    linked_truth = observations_linkage["truth"].unique()[0]

    assert linkage["linked_truth"].values[0] == linked_truth
    assert linkage["num_obs"].values[0] == members["obs_id"].nunique()
    assert linkage["num_members"].values[0] == 1
    assert linkage["pure"].values[0] == 1
    assert linkage["partial"].values[0] == 0
    assert linkage["mixed"].values[0] == 0
    assert linkage["contamination_percentage"].values[0] == 0.0

    if pure_complete:
        observations_truth = observations[observations["truth"] == linked_truth]
        all_obs_ids = observations_truth["obs_id"].values
        assert linkage["num_obs"].values[0] == members["obs_id"].nunique()
        assert linkage["num_obs"].values[0] == len(members["obs_id"].values)
        assert np.all(np.isin(all_obs_ids, members["obs_id"].values))
        assert linkage["pure_complete"].values[0] == 1
    else:
        assert linkage["pure_complete"].values[0] == 0


def assert_valid_partial_linkage(
    linkage_id, all_linkages, linkage_members, observations, pure_complete=False
):
    """
    Checks that the given linkage is correctly identified as a pure linkage.
    """
    linkage = all_linkages[all_linkages["linkage_id"] == linkage_id]
    assert len(linkage) == 1
    members = linkage_members[linkage_members["linkage_id"] == linkage_id]
    observations_linkage = observations[observations["obs_id"].isin(members["obs_id"])]

    num_truths = observations_linkage["truth"].nunique()
    assert num_truths > 1
    truth_counts = observations_linkage["truth"].value_counts()
    linked_truth = truth_counts.index[0]
    linked_truth_obs = truth_counts.values[0]
    contamination_percentage = (1 - linked_truth_obs / len(observations_linkage)) * 100

    assert linkage["linked_truth"].values[0] == linked_truth
    assert linkage["num_obs"].values[0] == members["obs_id"].nunique()
    assert linkage["num_members"].values[0] == num_truths
    assert linkage["pure"].values[0] == 0
    assert linkage["pure_complete"].values[0] == 0
    assert linkage["partial"].values[0] == 1
    assert linkage["mixed"].values[0] == 0
    assert linkage["contamination_percentage"].values[0] == contamination_percentage


def assert_valid_mixed_linkage(linkage_id, all_linkages, linkage_members, observations, pure_complete=False):
    """
    Checks that the given linkage is correctly identified as a pure linkage.
    """
    linkage = all_linkages[all_linkages["linkage_id"] == linkage_id]
    assert len(linkage) == 1
    members = linkage_members[linkage_members["linkage_id"] == linkage_id]
    observations_linkage = observations[observations["obs_id"].isin(members["obs_id"])]

    num_truths = observations_linkage["truth"].nunique()
    assert num_truths > 1

    assert linkage["linked_truth"].values[0] == "nan"
    assert linkage["num_obs"].values[0] == members["obs_id"].nunique()
    assert linkage["num_members"].values[0] == num_truths
    assert linkage["pure"].values[0] == 0
    assert linkage["pure_complete"].values[0] == 0
    assert linkage["partial"].values[0] == 0
    assert linkage["mixed"].values[0] == 1
    assert np.isnan(linkage["contamination_percentage"].values[0])


def test_analyzeLinkages_no_classes_0pct_contamination(test_observations, test_linkages):
    test_linkage_members, expected_all_linkages = test_linkages

    # Test analyzeLinkages when no classes are given
    all_linkages, all_truths, summary = analyzeLinkages(
        test_observations,
        test_linkage_members,
        min_obs=5,
        contamination_percentage=0.0,
        classes=None,
    )

    # --- Test the summary dataframe ---
    # No all_truths dataframe was passed so completeness should be NaN
    all_mask = summary["class"] == "All"
    for col in [
        "completeness",
        "findable",
        "findable_found",
        "findable_missed",
        "not_findable_found",
        "not_findable_missed",
    ]:
        assert np.isnan(summary[all_mask][col].values[0])

    assert summary[all_mask]["num_members"].values[0] == 3
    assert summary[all_mask]["num_obs"].values[0] == 30
    assert summary[all_mask]["linkages"].values[0] == test_linkage_members["linkage_id"].nunique()
    assert summary[all_mask]["pure_complete_linkages"].values[0] == 3
    assert summary[all_mask]["pure_linkages"].values[0] == 6
    # With 0% contamination there should be no partial linkages
    assert summary[all_mask]["partial_linkages"].values[0] == 0
    assert summary[all_mask]["mixed_linkages"].values[0] == 6

    assert summary[all_mask]["unique_in_pure_linkages"].values[0] == 3
    assert summary[all_mask]["unique_in_pure_complete_linkages"].values[0] == 3
    assert summary[all_mask]["unique_in_pure_and_partial_linkages"].values[0] == 0
    # With 0% contamination there should be no partial linkages
    assert summary[all_mask]["unique_in_partial_linkages"].values[0] == 0
    assert summary[all_mask]["unique_in_partial_contaminant_linkages"].values[0] == 0
    assert summary[all_mask]["unique_in_mixed_linkages"].values[0] == 3
    # Pure complete + pure linkages
    assert summary[all_mask]["obs_in_pure_linkages"].values[0] == 30 + 30 - 2 * 3
    assert summary[all_mask]["obs_in_pure_complete_linkages"].values[0] == 30
    assert summary[all_mask]["obs_in_partial_linkages"].values[0] == 0
    assert summary[all_mask]["obs_in_partial_contaminant_linkages"].values[0] == 0
    # With 0% contamination there should be no partial linkages so they should count
    # as mixed
    assert summary[all_mask]["obs_in_mixed_linkages"].values[0] == 42

    # --- Test the all_linkages dataframe ---
    pure_linkages = ["pure_23636", "pure_58177", "pure_82134"]
    pure_complete_linkages = ["pure_complete_23636", "pure_complete_58177", "pure_complete_82134"]
    partial_linkages = ["partial_23636", "partial_58177", "partial_82134"]
    mixed_linkages = ["mixed_0", "mixed_1", "mixed_2"]

    for linkage_id in pure_linkages:
        assert_valid_pure_linkage(linkage_id, all_linkages, test_linkage_members, test_observations)

    for linkage_id in pure_complete_linkages:
        assert_valid_pure_linkage(
            linkage_id, all_linkages, test_linkage_members, test_observations, pure_complete=True
        )

    # There should be no partial linkages with 0% contamination
    for linkage_id in partial_linkages + mixed_linkages:
        assert_valid_mixed_linkage(linkage_id, all_linkages, test_linkage_members, test_observations)

    return


def test_analyzeLinkages_no_classes_30pct_contamination(test_observations, test_linkages):
    test_linkage_members, expected_all_linkages = test_linkages

    # Test analyzeLinkages when no classes are given
    all_linkages, all_truths, summary = analyzeLinkages(
        test_observations,
        test_linkage_members,
        min_obs=5,
        contamination_percentage=30.0,
        classes=None,
    )

    # --- Test the summary dataframe ---
    # No all_truths dataframe was passed so completeness should be NaN
    all_mask = summary["class"] == "All"
    for col in [
        "completeness",
        "findable",
        "findable_found",
        "findable_missed",
        "not_findable_found",
        "not_findable_missed",
    ]:
        assert np.isnan(summary[all_mask][col].values[0])

    assert summary[all_mask]["num_members"].values[0] == 3
    assert summary[all_mask]["num_obs"].values[0] == 30
    assert summary[all_mask]["linkages"].values[0] == test_linkage_members["linkage_id"].nunique()
    assert summary[all_mask]["pure_complete_linkages"].values[0] == 3
    assert summary[all_mask]["pure_linkages"].values[0] == 6
    # With 0% contamination there should be no partial linkages
    assert summary[all_mask]["partial_linkages"].values[0] == 3
    assert summary[all_mask]["mixed_linkages"].values[0] == 3

    assert summary[all_mask]["unique_in_pure_linkages"].values[0] == 3
    assert summary[all_mask]["unique_in_pure_complete_linkages"].values[0] == 3
    assert summary[all_mask]["unique_in_pure_and_partial_linkages"].values[0] == 3
    # With 0% contamination there should be no partial linkages
    assert summary[all_mask]["unique_in_partial_linkages"].values[0] == 3
    assert summary[all_mask]["unique_in_partial_contaminant_linkages"].values[0] == 3
    assert summary[all_mask]["unique_in_mixed_linkages"].values[0] == 3
    # Pure complete + pure linkages
    assert summary[all_mask]["obs_in_pure_linkages"].values[0] == 30 + 30 - 2 * 3
    assert summary[all_mask]["obs_in_pure_complete_linkages"].values[0] == 30
    assert summary[all_mask]["obs_in_partial_linkages"].values[0] == 15
    assert summary[all_mask]["obs_in_partial_contaminant_linkages"].values[0] == 6
    # With 0% contamination there should be no partial linkages so they should count
    # as mixed
    assert summary[all_mask]["obs_in_mixed_linkages"].values[0] == 21

    # --- Test the all_linkages dataframe ---
    pure_linkages = ["pure_23636", "pure_58177", "pure_82134"]
    pure_complete_linkages = ["pure_complete_23636", "pure_complete_58177", "pure_complete_82134"]
    partial_linkages = ["partial_23636", "partial_58177", "partial_82134"]
    mixed_linkages = ["mixed_0", "mixed_1", "mixed_2"]

    for linkage_id in pure_linkages:
        assert_valid_pure_linkage(linkage_id, all_linkages, test_linkage_members, test_observations)

    for linkage_id in pure_complete_linkages:
        assert_valid_pure_linkage(
            linkage_id, all_linkages, test_linkage_members, test_observations, pure_complete=True
        )

    for linkage_id in partial_linkages:
        assert_valid_partial_linkage(linkage_id, all_linkages, test_linkage_members, test_observations)

    # There should be no partial linkages with 0% contamination
    for linkage_id in mixed_linkages:
        assert_valid_mixed_linkage(linkage_id, all_linkages, test_linkage_members, test_observations)

    return


def test_analyzeLinkages_classes_column_0pct_contamination(test_observations, test_linkages):
    test_linkage_members, expected_all_linkages = test_linkages

    # Test analyzeLinkages when a class column is given

    # Add class column to test observations
    for i, object_id in enumerate(["23636", "58177", "82134"]):
        test_observations.loc[test_observations["truth"] == object_id, "class"] = "Class_{}".format(i)

    all_linkages, all_truths, summary = analyzeLinkages(
        test_observations,
        test_linkage_members,
        min_obs=5,
        contamination_percentage=0.0,
        classes="class",
    )

    # --- Test the summary dataframe ---
    # No all_truths dataframe was passed so completeness should be NaN
    all_mask = summary["class"] == "All"
    for col in [
        "completeness",
        "findable",
        "findable_found",
        "findable_missed",
        "not_findable_found",
        "not_findable_missed",
    ]:
        assert np.isnan(summary[all_mask][col].values[0])

    assert summary[all_mask]["num_members"].values[0] == 3
    assert summary[all_mask]["num_obs"].values[0] == 30
    assert summary[all_mask]["linkages"].values[0] == test_linkage_members["linkage_id"].nunique()
    assert summary[all_mask]["pure_complete_linkages"].values[0] == 3
    assert summary[all_mask]["pure_linkages"].values[0] == 6
    # With 0% contamination there should be no partial linkages
    assert summary[all_mask]["mixed_linkages"].values[0] == 6
    assert summary[all_mask]["partial_linkages"].values[0] == 0

    assert summary[all_mask]["unique_in_pure_linkages"].values[0] == 3
    assert summary[all_mask]["unique_in_pure_complete_linkages"].values[0] == 3
    assert summary[all_mask]["unique_in_pure_and_partial_linkages"].values[0] == 0
    # With 0% contamination there should be no partial linkages
    assert summary[all_mask]["unique_in_partial_linkages"].values[0] == 0
    assert summary[all_mask]["unique_in_partial_contaminant_linkages"].values[0] == 0
    assert summary[all_mask]["unique_in_mixed_linkages"].values[0] == 3
    # Pure complete + pure linkages
    assert summary[all_mask]["obs_in_pure_linkages"].values[0] == 30 + 30 - 2 * 3
    assert summary[all_mask]["obs_in_pure_complete_linkages"].values[0] == 30
    assert summary[all_mask]["obs_in_partial_linkages"].values[0] == 0
    assert summary[all_mask]["obs_in_partial_contaminant_linkages"].values[0] == 0
    # With 0% contamination there should be no partial linkages so they should count
    # as mixed
    assert summary[all_mask]["obs_in_mixed_linkages"].values[0] == 42

    for class_id in ["Class_0", "Class_1", "Class_2"]:
        class_mask = summary["class"] == class_id
        assert summary[class_mask]["num_members"].values[0] == 1
        assert summary[class_mask]["num_obs"].values[0] == len(
            test_observations[test_observations["class"] == class_id]
        )

    # --- Test the all_linkages dataframe ---
    pure_linkages = ["pure_23636", "pure_58177", "pure_82134"]
    pure_complete_linkages = ["pure_complete_23636", "pure_complete_58177", "pure_complete_82134"]
    partial_linkages = ["partial_23636", "partial_58177", "partial_82134"]
    mixed_linkages = ["mixed_0", "mixed_1", "mixed_2"]

    for linkage_id in pure_linkages:
        assert_valid_pure_linkage(linkage_id, all_linkages, test_linkage_members, test_observations)

    for linkage_id in pure_complete_linkages:
        assert_valid_pure_linkage(
            linkage_id, all_linkages, test_linkage_members, test_observations, pure_complete=True
        )

    # There should be no partial linkages with 0% contamination
    for linkage_id in partial_linkages + mixed_linkages:
        assert_valid_mixed_linkage(linkage_id, all_linkages, test_linkage_members, test_observations)

    return


def test_analyzeLinkages_errors(test_observations, test_linkages):
    # Test analyzeLinkages when incorrect data products are given
    test_linkage_members, expected_all_linkages = test_linkages

    # Make expected observations_test empty
    test_observations.drop(test_observations.index, inplace=True)

    # Check for ValueError when observations are empty
    with pytest.raises(ValueError):
        all_linkages, all_truths, summary = analyzeLinkages(
            test_observations,
            test_linkage_members,
            # all_truths=all_truths_test[["truth", "num_obs", "findable"]],
            min_obs=5,
            contamination_percentage=20.0,
            classes=None,
        )

    return
