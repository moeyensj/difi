import pytest

from ..difi import analyzeLinkages


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
    benchmark, test_observations, test_linkages, min_obs, contamination_percentage
):
    test_linkage_members, expected_all_linkages = test_linkages

    # Test analyzeLinkages when no classes are given and no all_objects are given
    all_linkages, all_objects, summary = benchmark(
        analyzeLinkages,
        test_observations,
        test_linkage_members,
        min_obs=min_obs,
        contamination_percentage=contamination_percentage,
        classes=None,
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
    benchmark, test_observations, test_linkages, test_all_objects, min_obs, contamination_percentage
):
    test_linkage_members, expected_all_linkages = test_linkages

    # Test analyzeLinkages when no classes are given and all_objects are given
    # Set findable column in test all truths
    test_all_objects.loc[:, "findable"] = 1

    all_linkages, all_objects, summary = benchmark(
        analyzeLinkages,
        test_observations,
        test_linkage_members,
        min_obs=min_obs,
        contamination_percentage=contamination_percentage,
        classes=None,
        all_objects=test_all_objects[["object_id", "num_obs", "findable"]],
    )

    return
