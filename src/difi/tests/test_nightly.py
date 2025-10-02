import numpy as np
import pytest
from adam_core.coordinates import Residuals
from adam_core.orbit_determination import FittedOrbitMembers
from adam_core.time import Timestamp

from ..nightly import NightlyOrbitSummary, OrbitSummary
from ..observations import Observations


@pytest.fixture
def observations_and_orbit_members() -> tuple[Observations, FittedOrbitMembers]:
    times = Timestamp.from_kwargs(
        days=[59000, 59000, 59001, 59001, 59002, 59003], nanos=[0, 60 * 1e9, 0, 60 * 1e9, 0, 0], scale="utc"
    )

    observations = Observations.from_kwargs(
        id=["1", "2", "3", "4", "5", "6"],
        time=times,
        ra=[10, 10.5, 11, 11.5, 12, 12.5],
        dec=[20, 20.1, 21, 21.1, 22, 23],
        mag=[22, 22.1, 22.8, 22.9, 23, 22.9],
        filter=["g", "r", "g", "g", "r", "z"],
        observatory_code=["X05", "X05", "I41", "X05", "X05", "X05"],
        night=times.days,
    )

    orbit_members = FittedOrbitMembers.from_kwargs(
        orbit_id=["1", "1", "1", "1", "1", "1", "2", "2", "2", "2", "2", "2"],
        obs_id=["1", "2", "3", "4", "5", "6", "1", "2", "3", "4", "5", "6"],
        residuals=Residuals.from_kwargs(
            values=[
                [np.nan, 0.002, 0.002, np.nan, np.nan, np.nan],
                [np.nan, 0.002, 0.002, np.nan, np.nan, np.nan],
                [np.nan, 0.002, 0.002, np.nan, np.nan, np.nan],
                [np.nan, 0.002, 0.002, np.nan, np.nan, np.nan],
                [np.nan, 0.002, 0.002, np.nan, np.nan, np.nan],
                [np.nan, 0.002, 0.002, np.nan, np.nan, np.nan],
                [np.nan, 0.004, 0.004, np.nan, np.nan, np.nan],
                [np.nan, 0.006, 0.006, np.nan, np.nan, np.nan],
                [np.nan, 0.008, 0.008, np.nan, np.nan, np.nan],
                [np.nan, 0.010, 0.010, np.nan, np.nan, np.nan],
                [np.nan, 0.012, 0.012, np.nan, np.nan, np.nan],
                [np.nan, 0.014, 0.014, np.nan, np.nan, np.nan],
            ],
            chi2=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ),
    )
    return observations, orbit_members


def test_NightlyOrbitSummary_create(observations_and_orbit_members):
    # Test that the NightlyOrbitSummary.create method to group
    # orbit members by orbit and night and calculate summary statistics correctly
    observations, orbit_members = observations_and_orbit_members
    nightly_summary = NightlyOrbitSummary.create(orbit_members, observations)

    assert nightly_summary.orbit_id.to_pylist() == ["1", "1", "1", "1", "2", "2", "2", "2"]
    assert nightly_summary.night.to_pylist() == [59000, 59001, 59002, 59003, 59000, 59001, 59002, 59003]
    assert nightly_summary.num_obs.to_pylist() == [2, 2, 1, 1, 2, 2, 1, 1]
    assert nightly_summary.num_filters.to_pylist() == [2, 1, 1, 1, 2, 1, 1, 1]
    np.testing.assert_almost_equal(
        nightly_summary.dtime.to_numpy(zero_copy_only=False),
        [60, 60, 0, 0, 60, 60, 0, 0],
        decimal=6,
    )
    np.testing.assert_almost_equal(
        nightly_summary.dra.to_numpy(zero_copy_only=False),
        [1800.0, 1800.0, 0.0, 0.0, 1800.0, 1800.0, 0.0, 0.0],
    )
    np.testing.assert_almost_equal(
        nightly_summary.ddec.to_numpy(zero_copy_only=False), [360.0, 360.0, 0.0, 0.0, 360.0, 360.0, 0.0, 0.0]
    )
    np.testing.assert_almost_equal(
        nightly_summary.dsky.to_numpy(zero_copy_only=False),
        [1728.8058078, 1718.0213960, 0.0, 0.0, 1728.8058078, 1718.0213960, 0.0, 0.0],
    )
    np.testing.assert_almost_equal(
        nightly_summary.vsky.to_numpy(zero_copy_only=False),
        [1728.8058078 / 60, 1718.0213960 / 60, 0.0, 0.0, 1728.8058078 / 60, 1718.0213960 / 60, 0.0, 0.0],
    )
    np.testing.assert_almost_equal(
        nightly_summary.dmag.to_numpy(zero_copy_only=False),
        [
            0.1,
            0.1,
            0.0,
            0.0,
            0.1,
            0.1,
            0.0,
            0.0,
        ],
    )
    np.testing.assert_almost_equal(
        nightly_summary.mag_mean.to_numpy(zero_copy_only=False),
        [22.05, 22.85, 23.0, 22.9, 22.05, 22.85, 23.0, 22.9],
    )
    np.testing.assert_almost_equal(
        nightly_summary.mag_sigma.to_numpy(zero_copy_only=False), [0.05, 0.05, 0.0, 0.0, 0.05, 0.05, 0.0, 0.0]
    )
    np.testing.assert_almost_equal(
        nightly_summary.chi2_sum.to_numpy(zero_copy_only=False), [2.0, 2.0, 1.0, 1.0, 3.0, 7.0, 5.0, 6.0]
    )
    np.testing.assert_almost_equal(
        nightly_summary.chi2_mean.to_numpy(zero_copy_only=False), [1.0, 1.0, 1.0, 1.0, 1.5, 3.5, 5.0, 6.0]
    )
    np.testing.assert_almost_equal(
        nightly_summary.chi2_sigma.to_numpy(zero_copy_only=False), [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0]
    )
    np.testing.assert_almost_equal(
        nightly_summary.dchi2.to_numpy(zero_copy_only=False), [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    )


def test_OrbitSummary_create(observations_and_orbit_members):
    # Test that the OrbitSummary.create method to group
    # orbit members by orbit and calculate summary statistics correctly
    observations, orbit_members = observations_and_orbit_members
    nightly_summary = NightlyOrbitSummary.create(orbit_members, observations)
    orbit_summary = OrbitSummary.create(nightly_summary)

    assert orbit_summary.orbit_id.to_pylist() == ["1", "2"]
    assert orbit_summary.num_obs.to_pylist() == [6, 6]
    assert orbit_summary.num_nights.to_pylist() == [4, 4]
    assert orbit_summary.num_singletons.to_pylist() == [2, 2]
    assert orbit_summary.num_tracklets.to_pylist() == [2, 2]
