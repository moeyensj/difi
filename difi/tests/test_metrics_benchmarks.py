import pytest

from ..metrics import calcFindableMinObs, calcFindableNightlyLinkages


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

    benchmark(calcFindableMinObs, test_observations, min_obs=min_obs)

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

    benchmark(
        calcFindableNightlyLinkages,
        test_observations,
        linkage_min_obs=linkage_min_obs,
        max_obs_separation=max_obs_separation,
        min_linkage_nights=min_linkage_nights,
    )
    return
