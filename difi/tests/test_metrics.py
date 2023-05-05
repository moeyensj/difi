import numpy as np
import pytest

from ..metrics import (
    FindabilityMetric,
    MinObsMetric,
    NightlyLinkagesMetric,
    find_observations_beyond_angular_separation,
    find_observations_within_max_time_separation,
    select_tracklet_combinations,
)


def test_FindabilityMetric__compute_windows():
    # Test that the function returns the correct windows when detection_window is None
    nights = np.arange(1, 11)
    windows = FindabilityMetric._compute_windows(nights, detection_window=None)
    assert windows == [(1, 10)]

    # Test that the function returns the correct windows when detection_window is 2
    windows = FindabilityMetric._compute_windows(nights, detection_window=2)
    assert windows == [(1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9), (8, 10)]

    # Test that the function returns the correct windows when detection_window is 3
    windows = FindabilityMetric._compute_windows(nights, detection_window=3)
    assert windows == [(1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10)]

    # Test that the function returns the correct windows when detection_window is 6
    windows = FindabilityMetric._compute_windows(nights, detection_window=6)
    assert windows == [(1, 7), (2, 8), (3, 9), (4, 10)]

    # Test that the function returns the correct windows when detection_window is 15
    windows = FindabilityMetric._compute_windows(nights, detection_window=15)
    assert windows == [(1, 10)]


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
    ["by_object", "num_jobs"],
    [
        (True, None),
        (True, 1),
        (False, None),
        (False, 1),
    ],
)
def test_calcFindableMinObs(test_observations, by_object, num_jobs):

    # All three objects should be findable
    metric = MinObsMetric(min_obs=5)
    findable_observations, window_summary = metric.run(
        test_observations, by_object=by_object, num_jobs=num_jobs
    )
    assert len(findable_observations) == 3

    findable_ids = {k for k in findable_observations["truth"].values}
    for object_id in ["23636", "58177", "82134"]:
        assert object_id in findable_ids
        np.testing.assert_equal(
            findable_observations[findable_observations["truth"] == object_id]["obs_ids"].values[0][0],
            test_observations[test_observations["truth"] == object_id]["obs_id"].values,
        )

    # Only two objects should be findable
    metric = MinObsMetric(min_obs=10)
    findable_observations, window_summary = metric.run(
        test_observations, by_object=by_object, num_jobs=num_jobs
    )

    assert len(findable_observations) == 2
    for object_id in ["58177", "82134"]:
        assert object_id in findable_observations["truth"].values
        np.testing.assert_equal(
            findable_observations[findable_observations["truth"] == object_id]["obs_ids"].values[0][0],
            test_observations[test_observations["truth"] == object_id]["obs_id"].values,
        )

    # No objects should be findable
    metric = MinObsMetric(min_obs=16)
    findable_observations, window_summary = metric.run(
        test_observations, by_object=by_object, num_jobs=num_jobs
    )
    assert len(findable_observations) == 0

    # Set the detection window to 15 days, each object should still be findable
    metric = MinObsMetric(min_obs=5)
    findable_observations, window_summary = metric.run(
        test_observations, by_object=by_object, num_jobs=num_jobs, detection_window=15
    )
    assert len(findable_observations) == 3
    assert len(window_summary) == 1
    assert window_summary["start_night"].values[0] == 612
    assert window_summary["end_night"].values[0] == 624

    # Set the detection window to 10 days, there should now be 3 windows
    metric = MinObsMetric(min_obs=5)
    findable_observations, window_summary = metric.run(
        test_observations, by_object=by_object, num_jobs=num_jobs, detection_window=10
    )
    assert len(window_summary) == 3
    assert window_summary["start_night"].values[0] == 612
    assert window_summary["end_night"].values[0] == 622
    assert window_summary["num_findable"].values[0] == 3
    assert window_summary["num_obs"].values[0] == 26

    assert window_summary["start_night"].values[1] == 613
    assert window_summary["end_night"].values[1] == 623
    assert window_summary["num_findable"].values[1] == 3
    assert window_summary["num_obs"].values[1] == 19

    assert window_summary["start_night"].values[2] == 614
    assert window_summary["end_night"].values[2] == 624
    assert window_summary["num_findable"].values[2] == 3
    assert window_summary["num_obs"].values[2] == 23
    return


@pytest.mark.parametrize(
    ["by_object", "num_jobs"],
    [
        (True, None),
        (True, 1),
        (False, None),
        (False, 1),
    ],
)
def test_calcFindableNightlyLinkages(test_observations, by_object, num_jobs):

    # All three objects should be findable (each object has at least two tracklets
    # with consecutive observations no more than 2 hours apart)
    metric = NightlyLinkagesMetric(
        linkage_min_obs=2, max_obs_separation=2 / 24, min_linkage_nights=2, min_obs_angular_separation=0
    )
    findable_observations, window_summary = metric.run(
        test_observations, by_object=by_object, num_jobs=num_jobs
    )
    assert len(findable_observations) == 3

    findable_ids = {k for k in findable_observations["truth"].values}
    for object_id in ["23636", "58177", "82134"]:
        assert object_id in findable_ids

    # Object 23636 has two tracklets (no more than 2 hours long)
    np.testing.assert_equal(
        findable_observations[findable_observations["truth"] == "23636"]["obs_ids"].values[0][0],
        np.array(
            [
                "obs_000001",  # tracklet 1
                "obs_000002",  # tracklet 1
                "obs_000004",  # tracklet 2
                "obs_000005",  # tracklet 2
            ]
        ),
    )

    # Object 58177 has 5 tracklets no more than 2 hours long (all of its observations)
    np.testing.assert_equal(
        findable_observations[findable_observations["truth"] == "58177"]["obs_ids"].values[0][0],
        test_observations[test_observations["truth"] == "58177"]["obs_id"].values,
    )

    # Object 82134 has 3 tracklets no more than 2 hours long
    np.testing.assert_equal(
        findable_observations[findable_observations["truth"] == "82134"]["obs_ids"].values[0][0],
        np.array(
            [
                "obs_000016",  # tracklet 1
                "obs_000017",  # tracklet 1
                "obs_000018",  # tracklet 1
                "obs_000019",  # tracklet 1
                "obs_000021",  # tracklet 2
                "obs_000022",  # tracklet 2
                "obs_000023",  # tracklet 2
                "obs_000024",  # tracklet 3
                "obs_000025",  # tracklet 3
                "obs_000026",  # tracklet 4
                "obs_000027",  # tracklet 4
                "obs_000028",  # tracklet 5
                "obs_000029",  # tracklet 5
            ]
        ),
    )

    # Only two objects should be findable (each object has at least three tracklets
    # with consecutive observations no more than 2 hours apart)
    metric = NightlyLinkagesMetric(
        linkage_min_obs=2, max_obs_separation=2 / 24, min_linkage_nights=3, min_obs_angular_separation=1
    )
    findable_observations, window_summary = metric.run(
        test_observations, by_object=by_object, num_jobs=num_jobs
    )
    assert len(findable_observations) == 2

    findable_ids = {k for k in findable_observations["truth"].values}
    for object_id in ["58177", "82134"]:
        assert object_id in findable_ids

    # Object 58177 has 5 tracklets no more than 2 hours long (all of its observations)
    np.testing.assert_equal(
        findable_observations[findable_observations["truth"] == "58177"]["obs_ids"].values[0][0],
        test_observations[test_observations["truth"] == "58177"]["obs_id"].values,
    )

    # Object 82134 has 3 tracklets no more than 2 hours long (but several observations that
    # could form tracklets are too close together)
    np.testing.assert_equal(
        findable_observations[findable_observations["truth"] == "82134"]["obs_ids"].values[0][0],
        np.array(
            [
                # "obs_000016",  # tracklet 1
                "obs_000017",  # tracklet 1
                "obs_000018",  # tracklet 1
                # "obs_000019",  # tracklet 1
                "obs_000021",  # tracklet 2
                "obs_000022",  # tracklet 2
                # "obs_000023",  # tracklet 2
                "obs_000024",  # tracklet 3
                "obs_000025",  # tracklet 3
                "obs_000026",  # tracklet 4
                "obs_000027",  # tracklet 4
                "obs_000028",  # tracklet 5
                "obs_000029",  # tracklet 5
            ]
        ),
    )

    # Only one object should be findable (this object has at least two tracklets
    # with at least 3 consecutive observations no more than 2 hours apart)
    metric = NightlyLinkagesMetric(
        linkage_min_obs=3, max_obs_separation=2 / 24, min_linkage_nights=2, min_obs_angular_separation=0
    )
    findable_observations, window_summary = metric.run(
        test_observations, by_object=by_object, num_jobs=num_jobs
    )
    assert len(findable_observations) == 1

    findable_ids = {k for k in findable_observations["truth"].values}
    for object_id in ["82134"]:
        assert object_id in findable_ids

    # Object 82134 has 3 tracklets no more than 2 hours long
    np.testing.assert_equal(
        findable_observations[findable_observations["truth"] == "82134"]["obs_ids"].values[0][0],
        np.array(
            [
                "obs_000016",  # tracklet 1
                "obs_000017",  # tracklet 1
                "obs_000018",  # tracklet 1
                "obs_000019",  # tracklet 1
                "obs_000021",  # tracklet 2
                "obs_000022",  # tracklet 2
                "obs_000023",  # tracklet 2
            ]
        ),
    )

    # No objects should be findable if the minimum separation is 100 arcseconds
    metric = NightlyLinkagesMetric(
        linkage_min_obs=2, max_obs_separation=2 / 24, min_linkage_nights=3, min_obs_angular_separation=100
    )
    findable_observations, window_summary = metric.run(
        test_observations, by_object=by_object, num_jobs=num_jobs
    )
    assert len(findable_observations) == 0

    # All three objects should be findable (each object has at least two tracklets
    # with consecutive observations no more than 2 hours apart)
    # Set the detection window to 15 days, each object should still be findable
    metric = NightlyLinkagesMetric(
        linkage_min_obs=2, max_obs_separation=2 / 24, min_linkage_nights=2, min_obs_angular_separation=0
    )
    findable_observations, window_summary = metric.run(
        test_observations, by_object=by_object, num_jobs=num_jobs, detection_window=15
    )
    assert len(findable_observations) == 3
    assert len(window_summary) == 1
    assert window_summary["start_night"].values[0] == 612
    assert window_summary["end_night"].values[0] == 624

    # Set the detection window to 10 days and set the min_linkage nights to 3
    # There should be 3 windows and object one will never be findable
    metric = NightlyLinkagesMetric(
        linkage_min_obs=2, max_obs_separation=2 / 24, min_linkage_nights=3, min_obs_angular_separation=0
    )
    findable_observations, window_summary = metric.run(
        test_observations, by_object=by_object, num_jobs=num_jobs, detection_window=10
    )
    assert len(window_summary) == 3
    assert window_summary["start_night"].values[0] == 612
    assert window_summary["end_night"].values[0] == 622
    assert window_summary["num_findable"].values[0] == 2
    assert window_summary["num_obs"].values[0] == 26

    assert window_summary["start_night"].values[1] == 613
    assert window_summary["end_night"].values[1] == 623
    assert window_summary["num_findable"].values[1] == 2
    assert window_summary["num_obs"].values[1] == 19

    assert window_summary["start_night"].values[2] == 614
    assert window_summary["end_night"].values[2] == 624
    assert window_summary["num_findable"].values[2] == 2
    assert window_summary["num_obs"].values[2] == 23
    return


def test_calcFindableNightlyLinkages_edge_cases(test_observations):

    # All objects should be findable if we set linkage_min_obs=1
    metric = NightlyLinkagesMetric(linkage_min_obs=1, max_obs_separation=2 / 24, min_linkage_nights=1)
    findable_observations, window_summary = metric.run(test_observations)
    assert len(findable_observations) == 3

    findable_ids = {k for k in findable_observations["truth"].values}
    for object_id in ["23636", "58177", "82134"]:
        assert object_id in findable_ids
        np.testing.assert_equal(
            findable_observations[findable_observations["truth"] == object_id]["obs_ids"].values[0][0],
            test_observations[test_observations["truth"] == object_id]["obs_id"].values,
        )

    # Only two objects should be findable if we require at least 1 observation on each night of
    # 5 nights
    metric = NightlyLinkagesMetric(linkage_min_obs=1, max_obs_separation=2 / 24, min_linkage_nights=5)
    findable_observations, window_summary = metric.run(test_observations)
    assert len(findable_observations) == 2

    findable_ids = {k for k in findable_observations["truth"].values}
    for object_id in ["58177", "82134"]:
        assert object_id in findable_ids
        np.testing.assert_equal(
            findable_observations[findable_observations["truth"] == object_id]["obs_ids"].values[0][0],
            test_observations[test_observations["truth"] == object_id]["obs_id"].values,
        )


def test_calcFindableNightlyLinkages_assertion(test_observations):
    # Check that an assertion is raised if more than one object's observations
    # are passed to the metric's determine_object_findable method
    with pytest.raises(AssertionError):
        metric = NightlyLinkagesMetric()
        metric.determine_object_findable(test_observations)


def test_calcFindableMinObs_assertion(test_observations):
    # Check that an assertion is raised if more than one object's observations
    # are passed to the metric's determine_object_findable method
    with pytest.raises(AssertionError):
        metric = MinObsMetric()
        metric.determine_object_findable(test_observations)
