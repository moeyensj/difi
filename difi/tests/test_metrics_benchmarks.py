import numpy as np
import pytest

from ..metrics import (
    MinObsMetric,
    NightlyLinkagesMetric,
    find_observations_within_max_time_separation,
)


@pytest.mark.benchmark(group="metric_helpers")
def test_benchmark_find_observations_within_max_time_separation(benchmark):

    N = 1000
    obs_ids = np.array(["obs_{}".format(i) for i in range(N)])
    times = np.random.uniform(0, 100, N)
    times = times / 24.0 / 60  # Convert to days

    # Test that the function returns the correct observations when max_time_separation is 0.1
    benchmark(find_observations_within_max_time_separation, obs_ids, times, 1.0)

    return


@pytest.mark.parametrize(
    "min_obs",
    [
        5,
        10,
        20,
    ],
)
@pytest.mark.benchmark(group="metrics_min_obs")
def test_benchmark_calcFindableMinObs(benchmark, test_observations, min_obs):

    metric = MinObsMetric(min_obs=min_obs)
    benchmark(metric.run, test_observations)

    return


@pytest.mark.parametrize(
    "min_obs",
    [
        5,
        10,
        20,
    ],
)
@pytest.mark.benchmark(group="metrics_min_obs")
def test_benchmark_calcFindableMinObs_by_object(benchmark, test_observations, min_obs):

    metric = MinObsMetric(min_obs=min_obs)
    benchmark(metric.run, test_observations, by_object=True)

    return


@pytest.mark.parametrize(
    ["linkage_min_obs", "max_obs_separation", "min_linkage_nights"],
    [
        (2, 2 / 24, 2),
        (2, 4 / 24, 3),
        (3, 2 / 24, 2),
        (3, 4 / 24, 3),
        (4, 2 / 24, 2),
        (4, 4 / 24, 3),
    ],
)
@pytest.mark.benchmark(group="metrics_tracklets")
def test_benchmark_calcFindableNightlyLinkages(
    benchmark, test_observations, linkage_min_obs, max_obs_separation, min_linkage_nights
):

    metric = NightlyLinkagesMetric(
        linkage_min_obs=linkage_min_obs,
        max_obs_separation=max_obs_separation,
        min_linkage_nights=min_linkage_nights,
    )
    benchmark(
        metric.run,
        test_observations,
    )
    return


@pytest.mark.parametrize(
    ["linkage_min_obs", "max_obs_separation", "min_linkage_nights"],
    [
        (2, 2 / 24, 2),
        (2, 4 / 24, 3),
        (3, 2 / 24, 2),
        (3, 4 / 24, 3),
        (4, 2 / 24, 2),
        (4, 4 / 24, 3),
    ],
)
@pytest.mark.benchmark(group="metrics_tracklets")
def test_benchmark_calcFindableNightlyLinkages_by_object(
    benchmark, test_observations, linkage_min_obs, max_obs_separation, min_linkage_nights
):

    metric = NightlyLinkagesMetric(
        linkage_min_obs=linkage_min_obs,
        max_obs_separation=max_obs_separation,
        min_linkage_nights=min_linkage_nights,
    )
    benchmark(metric.run, test_observations, by_object=True)
    return
