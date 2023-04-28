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
def test_benchmark_analyze_linkages_no_classes_no_all_truths(
    benchmark, test_observations, test_linkages, min_obs, contamination_percentage
):
    test_linkage_members, expected_all_linkages = test_linkages

    # Test analyzeLinkages when no classes are given and no all_truths are given
    all_linkages, all_truths, summary = benchmark(
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
def test_benchmark_analyze_linkages_no_classes_all_truths(
    benchmark, test_observations, test_linkages, test_all_truths, min_obs, contamination_percentage
):
    test_linkage_members, expected_all_linkages = test_linkages

    # Test analyzeLinkages when no classes are given and all_truths are given
    # Set findable column in test all truths
    test_all_truths.loc[:, "findable"] = 1

    all_linkages, all_truths, summary = benchmark(
        analyzeLinkages,
        test_observations,
        test_linkage_members,
        min_obs=min_obs,
        contamination_percentage=contamination_percentage,
        classes=None,
        all_truths=test_all_truths[["truth", "num_obs", "findable"]],
    )

    return
