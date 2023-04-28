import numpy as np

from ..metrics import calcFindableMinObs, calcFindableNightlyLinkages


def test_calcFindableMinObs(test_observations):

    # All three objects should be findable
    findable_observations = calcFindableMinObs(test_observations, min_obs=1)
    assert len(findable_observations) == 3

    findable_ids = {k for k in findable_observations["truth"].values}
    for object_id in ["23636", "58177", "82134"]:
        assert object_id in findable_ids
        np.testing.assert_equal(
            findable_observations[findable_observations["truth"] == object_id]["obs_ids"].values[0],
            test_observations[test_observations["truth"] == object_id]["obs_id"].values,
        )

    # Only two objects should be findable
    findable_observations = calcFindableMinObs(test_observations, min_obs=10)

    assert len(findable_observations) == 2
    for object_id in ["58177", "82134"]:
        assert object_id in findable_observations["truth"].values
        np.testing.assert_equal(
            findable_observations[findable_observations["truth"] == object_id]["obs_ids"].values[0],
            test_observations[test_observations["truth"] == object_id]["obs_id"].values,
        )

    # No objects should be findable
    findable_observations = calcFindableMinObs(test_observations, min_obs=16)
    assert len(findable_observations) == 0

    return


def test_calcFindableNightlyLinkages(test_observations):

    # All three objects should be findable (each object has at least two tracklets
    # with consecutive observations no more than 2 hours apart)
    findable_observations = calcFindableNightlyLinkages(
        test_observations, linkage_min_obs=2, max_obs_separation=2 / 24, min_linkage_nights=2
    )
    assert len(findable_observations) == 3

    findable_ids = {k for k in findable_observations["truth"].values}
    for object_id in ["23636", "58177", "82134"]:
        assert object_id in findable_ids

    # Object 23636 has two tracklets (no more than 2 hours long)
    np.testing.assert_equal(
        findable_observations[findable_observations["truth"] == "23636"]["obs_ids"].values[0],
        np.array(
            [
                "obs_000001",
                "obs_000002",  # tracklet 1
                "obs_000004",
                "obs_000005",  # tracklet 2
            ]
        ),
    )

    # Object 58177 has 5 tracklets no more than 2 hours long (all of its observations)
    np.testing.assert_equal(
        findable_observations[findable_observations["truth"] == "58177"]["obs_ids"].values[0],
        test_observations[test_observations["truth"] == "58177"]["obs_id"].values,
    )

    # Object 82134 has 3 tracklets no more than 2 hours long
    np.testing.assert_equal(
        findable_observations[findable_observations["truth"] == "82134"]["obs_ids"].values[0],
        np.array(
            [
                "obs_000016",
                "obs_000017",
                "obs_000018",
                "obs_000019",  # tracklet 1
                "obs_000021",
                "obs_000022",
                "obs_000023",  # tracklet 2
                "obs_000024",
                "obs_000025",  # tracklet 3
                "obs_000026",
                "obs_000027",  # tracklet 4
                "obs_000028",
                "obs_000029",  # tracklet 5
            ]
        ),
    )

    # Only two objects should be findable (each object has at least three tracklets
    # with consecutive observations no more than 2 hours apart)
    findable_observations = calcFindableNightlyLinkages(
        test_observations, linkage_min_obs=2, max_obs_separation=2 / 24, min_linkage_nights=3
    )
    assert len(findable_observations) == 2

    findable_ids = {k for k in findable_observations["truth"].values}
    for object_id in ["58177", "82134"]:
        assert object_id in findable_ids

    # Object 58177 has 5 tracklets no more than 2 hours long (all of its observations)
    np.testing.assert_equal(
        findable_observations[findable_observations["truth"] == "58177"]["obs_ids"].values[0],
        test_observations[test_observations["truth"] == "58177"]["obs_id"].values,
    )

    # Object 82134 has 3 tracklets no more than 2 hours long
    np.testing.assert_equal(
        findable_observations[findable_observations["truth"] == "82134"]["obs_ids"].values[0],
        np.array(
            [
                "obs_000016",
                "obs_000017",
                "obs_000018",
                "obs_000019",  # tracklet 1
                "obs_000021",
                "obs_000022",
                "obs_000023",  # tracklet 2
                "obs_000024",
                "obs_000025",  # tracklet 3
                "obs_000026",
                "obs_000027",  # tracklet 4
                "obs_000028",
                "obs_000029",  # tracklet 5
            ]
        ),
    )

    # Only one object should be findable (this object has at least two tracklets
    # with at least 3 consecutive observations no more than 2 hours apart)
    findable_observations = calcFindableNightlyLinkages(
        test_observations, linkage_min_obs=3, max_obs_separation=2 / 24, min_linkage_nights=2
    )
    assert len(findable_observations) == 1

    findable_ids = {k for k in findable_observations["truth"].values}
    for object_id in ["82134"]:
        assert object_id in findable_ids

    # Object 82134 has 3 tracklets no more than 2 hours long
    np.testing.assert_equal(
        findable_observations[findable_observations["truth"] == "82134"]["obs_ids"].values[0],
        np.array(
            [
                "obs_000016",
                "obs_000017",
                "obs_000018",
                "obs_000019",  # tracklet 1
                "obs_000021",
                "obs_000022",
                "obs_000023",  # tracklet 2
            ]
        ),
    )

    return


def test_calcFindableNightlyLinkages_edge_cases(test_observations):

    # All objects should be findable if we set linkage_min_obs=1
    findable_observations = calcFindableNightlyLinkages(
        test_observations, linkage_min_obs=1, max_obs_separation=2 / 24, min_linkage_nights=1
    )
    assert len(findable_observations) == 3

    findable_ids = {k for k in findable_observations["truth"].values}
    for object_id in ["23636", "58177", "82134"]:
        assert object_id in findable_ids
        np.testing.assert_equal(
            findable_observations[findable_observations["truth"] == object_id]["obs_ids"].values[0],
            test_observations[test_observations["truth"] == object_id]["obs_id"].values,
        )

    # Only two objects should be findable if we require at least 1 observation on each night of
    # 5 nights
    findable_observations = calcFindableNightlyLinkages(
        test_observations, linkage_min_obs=1, max_obs_separation=2 / 24, min_linkage_nights=5
    )
    assert len(findable_observations) == 2

    findable_ids = {k for k in findable_observations["truth"].values}
    for object_id in ["58177", "82134"]:
        assert object_id in findable_ids
        np.testing.assert_equal(
            findable_observations[findable_observations["truth"] == object_id]["obs_ids"].values[0],
            test_observations[test_observations["truth"] == object_id]["obs_id"].values,
        )
