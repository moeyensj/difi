import pytest

from ..cifi import analyzeObservations


@pytest.mark.benchmark(group="analyze_observations")
def test_benchmark_analyze_observations_no_classes(benchmark, test_observations):
    # Test analyzeObservations when no truth classes are given

    all_truths, findable_observations, summary = benchmark(
        analyzeObservations,
        test_observations,
        min_obs=5,
        classes=None,
        detection_window=None,
    )

    return


@pytest.mark.benchmark(group="analyze_observations")
def test_benchmark_analyze_observations_classes_column(benchmark, test_observations):
    # Test analyzeObservations when a column name is given for the truth classes

    # Add class column to test observations
    for i, object_id in enumerate(["23636", "58177", "82134"]):
        test_observations.loc[test_observations["object_id"] == object_id, "class"] = "Class_{}".format(i)

    all_truths, findable_observations, summary = benchmark(
        analyzeObservations,
        test_observations,
        min_obs=5,
        classes="class",
        detection_window=None,
    )

    return


@pytest.mark.benchmark(group="analyze_observations")
def test_benchmark_analyze_observations_classes_dictionary(benchmark, test_observations):
    # Test analyzeObservations when a dictionary is given for the truth classes

    # Add class column to test observations
    classes_dict = {
        "Class_0": ["23636"],
        "Class_1": ["58177"],
        "Class_2": ["82134"],
    }

    all_truths, findable_observations, summary = benchmark(
        analyzeObservations,
        test_observations,
        min_obs=5,
        classes=classes_dict,
        detection_window=None,
    )

    return
