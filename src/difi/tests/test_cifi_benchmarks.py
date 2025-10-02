import pytest

from ..cifi import analyze_observations


@pytest.mark.benchmark(group="analyze_observations")
def test_benchmark_analyze_observations(benchmark, test_observations):
    # Benchmark analyze_observations with default singletons metric

    all_objects, findable_observations, partition_summary = benchmark(
        analyze_observations,
        test_observations,
        partitions=None,
        metric="singletons",
        by_object=True,
        max_processes=1,
    )

    return
