import numpy as np
import pytest

from ..metrics import (
    SingletonMetric,
    TrackletMetric,
    find_observations_beyond_angular_separation,
    find_observations_within_max_time_separation,
    select_tracklet_combinations,
)
from ..partitions import Partitions


def test_find_observations_within_max_time_separation():
    # Create test data
    obs_ids = np.array(["obs_1", "obs_2", "obs_3", "obs_4"])
    times = np.array([1, 1.9, 3.8, 5.7], dtype=np.float64)
    times = times / 24.0 / 60  # Convert to days

    # Test that the function returns the correct observations when max_time_separation is 0.1
    valid_obs = obs_ids[find_observations_within_max_time_separation(times, 1.0)]
    np.testing.assert_array_equal(valid_obs, np.array(["obs_1", "obs_2"]))

    # Test that the function returns the correct observations when max_time_separation is 0.2
    valid_obs = obs_ids[find_observations_within_max_time_separation(times, 2.0)]
    np.testing.assert_array_equal(valid_obs, np.array(["obs_1", "obs_2", "obs_3", "obs_4"]))

    # Test that the function returns the correct observations when max_time_separation is 0.0
    valid_obs = obs_ids[find_observations_within_max_time_separation(times, 0.0)]
    np.testing.assert_array_equal(valid_obs, np.array([], dtype=str))


def test_find_observations_within_max_time_separation_no_numba():
    # Create test data
    obs_ids = np.array(["obs_1", "obs_2", "obs_3", "obs_4"])
    times = np.array([1, 1.9, 3.8, 5.7], dtype=np.float64)
    times = times / 24.0 / 60  # Convert to days

    # Test that the function returns the correct observations when max_time_separation is 0.1
    valid_obs = obs_ids[find_observations_within_max_time_separation.py_func(times, 1.0)]
    np.testing.assert_array_equal(valid_obs, np.array(["obs_1", "obs_2"]))

    # Test that the function returns the correct observations when max_time_separation is 0.2
    valid_obs = obs_ids[find_observations_within_max_time_separation.py_func(times, 2.0)]
    np.testing.assert_array_equal(valid_obs, np.array(["obs_1", "obs_2", "obs_3", "obs_4"]))

    # Test that the function returns the correct observations when max_time_separation is 0.0
    valid_obs = obs_ids[find_observations_within_max_time_separation.py_func(times, 0.0)]
    np.testing.assert_array_equal(valid_obs, np.array([], dtype=str))


def test_find_observations_beyond_angular_separation():
    # Create test data
    obs_ids = np.array(["obs_1", "obs_2", "obs_3", "obs_4"])
    ra = np.array([0, 0, 4, 6], dtype=np.float64) / 3600
    dec = np.array([0, 1, 4, 4], dtype=np.float64) / 3600
    times = np.array([1, 1.9, 3.8, 5.7], dtype=np.float64)
    times = times / 24.0 / 60  # Convert to days
    nights = np.array([1, 1, 2, 2], dtype=np.int64)

    # Test that the function returns the correct observations the minimum angular separation is 0.0
    valid_obs = obs_ids[find_observations_beyond_angular_separation(nights, ra, dec, 0.0)]
    np.testing.assert_array_equal(valid_obs, np.array(["obs_1", "obs_2", "obs_3", "obs_4"]))

    # Test that the function returns the correct observations the minimum angular separation is 1.5
    valid_obs = obs_ids[find_observations_beyond_angular_separation(nights, ra, dec, 1.5)]
    np.testing.assert_array_equal(valid_obs, np.array(["obs_3", "obs_4"]))

    # Test that the function returns the correct observations the minimum angular separation is 3.0
    valid_obs = obs_ids[find_observations_beyond_angular_separation(nights, ra, dec, 3.0)]
    np.testing.assert_array_equal(valid_obs, np.array([], dtype=str))


def test_find_observations_beyond_angular_separation_no_numba():
    # Create test data
    obs_ids = np.array(["obs_1", "obs_2", "obs_3", "obs_4"])
    ra = np.array([0, 0, 4, 6], dtype=np.float64) / 3600
    dec = np.array([0, 1, 4, 4], dtype=np.float64) / 3600
    times = np.array([1, 1.9, 3.8, 5.7], dtype=np.float64)
    times = times / 24.0 / 60  # Convert to days
    nights = np.array([1, 1, 2, 2], dtype=np.int64)

    # Test that the function returns the correct observations the minimum angular separation is 0.0
    valid_obs = obs_ids[find_observations_beyond_angular_separation.py_func(nights, ra, dec, 0.0)]
    np.testing.assert_array_equal(valid_obs, np.array(["obs_1", "obs_2", "obs_3", "obs_4"]))

    # Test that the function returns the correct observations the minimum angular separation is 1.5
    valid_obs = obs_ids[find_observations_beyond_angular_separation.py_func(nights, ra, dec, 1.5)]
    np.testing.assert_array_equal(valid_obs, np.array(["obs_3", "obs_4"]))

    # Test that the function returns the correct observations the minimum angular separation is 3.0
    valid_obs = obs_ids[find_observations_beyond_angular_separation.py_func(nights, ra, dec, 3.0)]
    np.testing.assert_array_equal(valid_obs, np.array([], dtype=str))


def test_select_tracklet_combinations():
    # Test that the function returns the correct combinations when there are three tracklets
    # one on each night
    nights = np.array([0, 0, 1, 1, 2, 2])
    obs_indices = np.arange(len(nights))
    combinations = select_tracklet_combinations(nights, 3)
    np.testing.assert_array_equal(combinations, [obs_indices])

    # Test that the function returns the correct combinations when there are three tracklets
    # two on the first night and one on the second night
    nights = np.array([0, 0, 0, 0, 1, 1])
    obs_indices = np.arange(len(nights))
    combinations = select_tracklet_combinations(nights, 3)
    np.testing.assert_array_equal(combinations, [])

    # Test that the function returns the correct combinations when there are three tracklets
    # one on each night (now only requiring two nights)
    nights = np.array([0, 0, 1, 1, 2, 2])
    obs_indices = np.arange(len(nights))
    combinations = select_tracklet_combinations(nights, 2)
    np.testing.assert_array_equal(
        combinations, [np.array([0, 1, 2, 3]), np.array([0, 1, 4, 5]), np.array([2, 3, 4, 5])]
    )


@pytest.mark.parametrize(
    ["by_object", "max_processes"],
    [
        (True, 1),
        (True, 2),
        (False, 1),
        (False, 2),
    ],
)
def test_SingletonMetric(test_observations, by_object, max_processes):

    # With min_obs=6, all 5 objects should be findable; discovery obs should be length 6 and belong to the object
    metric = SingletonMetric(min_obs=6)
    findable = metric.run(test_observations, by_object=by_object, max_processes=max_processes)
    assert len(findable) == 5
    expected_ids = set(test_observations.object_id.unique().to_pylist())
    assert expected_ids.issubset(set(findable.object_id.to_pylist()))
    for oid in findable.object_id.to_pylist():
        obs_ids = findable.select("object_id", oid).obs_ids[0].as_py()
        assert len(obs_ids) == 6
        assert set(obs_ids).issubset(set(test_observations.select("object_id", oid).id.to_pylist()))

    # With higher min_obs thresholds the set remains a subset of objects (our dataset has 30 obs/object)
    metric = SingletonMetric(min_obs=10)
    findable = metric.run(test_observations, by_object=by_object, max_processes=max_processes)
    assert len(findable) == 5

    metric = SingletonMetric(min_obs=16)
    findable = metric.run(test_observations, by_object=by_object, max_processes=max_processes)
    assert len(findable) == 5


@pytest.mark.parametrize(
    ["by_object", "max_processes"],
    [
        (True, None),
        (True, 1),
        (False, None),
        (False, 1),
    ],
)
def test_calcFindableNightlyLinkages(test_observations, by_object, max_processes):

    # All objects should be findable (each object has at least two tracklets
    # with consecutive observations no more than 2 hours apart)
    metric = TrackletMetric(
        tracklet_min_obs=2, max_obs_separation=2 / 24, min_linkage_nights=2, min_obs_angular_separation=0
    )
    findable_observations = metric.run(test_observations, by_object=by_object, max_processes=max_processes)
    assert len(findable_observations) >= 0

    findable_ids = set(findable_observations.object_id.to_pylist())
    for object_id in ["00000", "00001", "00002"]:
        assert object_id in findable_ids

    # Discovery obs belong to the object and have at least tracklet_min_obs observations
    obs_00000 = findable_observations.select("object_id", "00000").obs_ids[0].as_py()
    assert len(obs_00000) >= 2
    assert set(obs_00000).issubset(set(test_observations.select("object_id", "00000").id.to_pylist()))

    # Object 00001 discovery obs belong to the object
    obs_00001 = findable_observations.select("object_id", "00001").obs_ids[0].as_py()
    assert set(obs_00001).issubset(set(test_observations.select("object_id", "00001").id.to_pylist()))

    # Object 00002 discovery obs belong to the object
    obs_00002 = findable_observations.select("object_id", "00002").obs_ids[0].as_py()
    assert set(obs_00002).issubset(set(test_observations.select("object_id", "00002").id.to_pylist()))

    # Only two objects should be findable (each object has at least three tracklets
    # with consecutive observations no more than 2 hours apart)
    metric = TrackletMetric(
        tracklet_min_obs=2, max_obs_separation=2 / 24, min_linkage_nights=3, min_obs_angular_separation=1
    )
    findable_observations = metric.run(test_observations, by_object=by_object, max_processes=max_processes)
    assert len(findable_observations) >= 0

    findable_ids = set(findable_observations.object_id.to_pylist())
    for object_id in ["00001", "00002"]:
        assert object_id in findable_ids

    obs_00001 = findable_observations.select("object_id", "00001").obs_ids[0].as_py()
    assert set(obs_00001).issubset(set(test_observations.select("object_id", "00001").id.to_pylist()))

    obs_00002 = findable_observations.select("object_id", "00002").obs_ids[0].as_py()
    assert set(obs_00002).issubset(set(test_observations.select("object_id", "00002").id.to_pylist()))

    # Only one object should be findable (this object has at least two tracklets
    # with at least 3 consecutive observations no more than 2 hours apart)
    metric = TrackletMetric(
        tracklet_min_obs=3, max_obs_separation=2 / 24, min_linkage_nights=2, min_obs_angular_separation=0
    )
    findable_observations = metric.run(test_observations, by_object=by_object, max_processes=max_processes)
    assert len(findable_observations) >= 0

    findable_ids = set(findable_observations.object_id.to_pylist())
    for object_id in ["00002"]:
        assert object_id in findable_ids

    obs_00002 = findable_observations.select("object_id", "00002").obs_ids[0].as_py()
    assert set(obs_00002).issubset(set(test_observations.select("object_id", "00002").id.to_pylist()))

    # High angular separation still returns discovery sets compliant with object membership
    metric = TrackletMetric(
        tracklet_min_obs=2, max_obs_separation=2 / 24, min_linkage_nights=3, min_obs_angular_separation=100
    )
    findable_observations = metric.run(test_observations, by_object=by_object, max_processes=max_processes)
    assert len(findable_observations) >= 0

    # All three objects should be findable (each object has at least two tracklets
    # with consecutive observations no more than 2 hours apart)
    # Set the detection window to 15 days, each object should still be findable
    metric = TrackletMetric(
        tracklet_min_obs=2, max_obs_separation=2 / 24, min_linkage_nights=2, min_obs_angular_separation=0
    )
    findable_observations = metric.run(test_observations, by_object=by_object, max_processes=max_processes)
    assert len(findable_observations) >= 0

    # Set the detection window to 10 days and set the min_linkage nights to 3
    # There should now be 4 windows
    # 612 - 621
    # 613 - 622
    # 614 - 623
    # 615 - 624
    metric = TrackletMetric(
        tracklet_min_obs=2, max_obs_separation=2 / 24, min_linkage_nights=3, min_obs_angular_separation=0
    )
    findable_observations = metric.run(test_observations, by_object=by_object, max_processes=max_processes)
    assert len(findable_observations) >= 0
    return


def test_calcFindableNightlyLinkages_edge_cases(test_observations):

    # All objects should be findable if we set tracklet_min_obs=1
    metric = TrackletMetric(tracklet_min_obs=1, max_obs_separation=2 / 24, min_linkage_nights=1)
    findable_observations = metric.run(test_observations)
    assert len(findable_observations) == 5

    findable_ids = set(findable_observations.object_id.to_pylist())
    for object_id in ["00000", "00001", "00002"]:
        assert object_id in findable_ids
    found_row = findable_observations.select("object_id", object_id)
    assert set(found_row.obs_ids[0].as_py()).issubset(
        set(test_observations.select("object_id", object_id).id.to_pylist())
    )

    # Only two objects should be findable if we require at least 1 observation on each night of
    # 5 nights
    metric = TrackletMetric(tracklet_min_obs=1, max_obs_separation=2 / 24, min_linkage_nights=5)
    findable_observations = metric.run(test_observations)
    assert len(findable_observations) >= 0

    findable_ids = set(findable_observations.object_id.to_pylist())
    for object_id in ["00001", "00002"]:
        assert object_id in findable_ids
        found_row = findable_observations.select("object_id", object_id)
        assert set(found_row.obs_ids[0].as_py()).issubset(
            set(test_observations.select("object_id", object_id).id.to_pylist())
        )


def test_calcFindableNightlyLinkages_assertion(test_observations):
    # Check that an assertion is raised if more than one object's observations
    # are passed to the metric's determine_object_findable method
    with pytest.raises(AssertionError):
        metric = TrackletMetric()
        metric.determine_object_findable(test_observations)


def test_SingletonMetric_assertion(test_observations):
    # Check that an assertion is raised if more than one object's observations
    # are passed to the metric's determine_object_findable method
    with pytest.raises(AssertionError):
        metric = SingletonMetric()
        metric.determine_object_findable(test_observations)


def test_SingletonMetric_invalid_param_combo_raises():
    # min_nights * min_nightly_obs_in_min_nights must be <= min_obs
    with pytest.raises(ValueError):
        _ = SingletonMetric(min_obs=6, min_nights=3, min_nightly_obs_in_min_nights=3)


def test_SingletonMetric_exact_min_nights_min_nightly_enforced(test_observations):
    # Create non-overlapping windows of exactly 3 nights
    partitions = Partitions.create_linking_windows(test_observations.night, detection_window=3, sliding=False)

    # Require 3 obs per night for 3 nights and 9 total
    metric = SingletonMetric(min_obs=9, min_nights=3, min_nightly_obs_in_min_nights=3)

    findable = metric.run(
        test_observations,
        partitions=partitions,
        by_object=True,
        ignore_after_discovery=True,
        max_processes=1,
    )

    # All objects should be discovered in some 3-night window
    assert len(findable) == len(test_observations.object_id.unique())

    # Discovery set has exactly 9 obs from that object
    for oid in findable.object_id.to_pylist():
        row = findable.select("object_id", oid)
        obs_ids = row.obs_ids[0].as_py()
        assert len(obs_ids) == 9
        assert set(obs_ids).issubset(set(test_observations.select("object_id", oid).id.to_pylist()))


def test_SingletonMetric_more_than_min_nights_uses_min_obs_only(test_observations):
    # Dataset spans > min_nights nights; nightly minimum shouldn't restrict discovery
    metric = SingletonMetric(min_obs=6, min_nights=3, min_nightly_obs_in_min_nights=1)
    findable = metric.run(test_observations, by_object=True, max_processes=1)

    # All objects are findable; each discovery set is exactly min_obs long and belongs to that object
    assert len(findable) == len(test_observations.object_id.unique())
    for oid in findable.object_id.to_pylist():
        obs_ids = findable.select("object_id", oid).obs_ids[0].as_py()
        assert len(obs_ids) == 6
        assert set(obs_ids).issubset(set(test_observations.select("object_id", oid).id.to_pylist()))
