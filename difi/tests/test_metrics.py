import os

import numpy as np
import pandas as pd
import pytest

from ..metrics import calcFindableMinObs, calcFindableNightlyLinkages


@pytest.fixture
def test_observations():
    """
    Create a test data set of observations. These observations were taken
    a subset of the data used for the tutorial. The observations are:

    obs_id,time,ra,dec,night_id,truth                  # object, tracklet?
    ------------------------------------------------
    obs_000000,58366.402,348.24238,5.44284,612,23636   # 1, singleton
    obs_000001,58368.294,347.76604,5.28888,614,23636   # 1, tracklet 01
    obs_000002,58368.360,347.74873,5.28336,614,23636   # 1, tracklet 01
    obs_000003,58371.312,347.00285,5.02589,617,23636   # 1, singleton
    obs_000004,58374.276,346.25985,4.74993,620,23636   # 1, tracklet 02
    obs_000005,58374.318,346.24901,4.74593,620,23636   # 1, tracklet 02
    obs_000006,58366.338,353.88551,1.57526,612,58177   # 2, tracklet 01
    obs_000007,58366.402,353.86922,1.57478,612,58177   # 2, tracklet 01
    obs_000008,58369.324,353.14697,1.54406,615,58177   # 2, tracklet 02
    obs_000009,58369.361,353.13724,1.54357,615,58177   # 2, tracklet 02
    obs_000010,58372.288,352.39237,1.49640,618,58177   # 2, tracklet 03
    obs_000011,58372.368,352.37078,1.49490,618,58177   # 2, tracklet 03
    obs_000012,58375.310,351.61052,1.43343,621,58177   # 2, tracklet 04
    obs_000013,58375.337,351.60326,1.43283,621,58177   # 2, tracklet 04
    obs_000014,58378.319,350.83215,1.35940,624,58177   # 2, tracklet 05
    obs_000015,58378.389,350.81329,1.35757,624,58177   # 2, tracklet 05
    obs_000016,58366.370,358.60072,8.22694,612,82134   # 3, tracklet 01
    obs_000017,58366.371,358.60060,8.22693,612,82134   # 3, tracklet 01
    obs_000018,58366.400,358.59361,8.22688,612,82134   # 3, tracklet 01
    obs_000019,58366.401,358.59347,8.22688,612,82134   # 3, tracklet 01
    obs_000020,58368.351,358.14871,8.21534,614,82134   # 3, singleton
    obs_000021,58369.326,357.92042,8.20361,615,82134   # 3, tracklet 02
    obs_000022,58369.359,357.91196,8.20314,615,82134   # 3, tracklet 02
    obs_000023,58369.360,357.91173,8.20313,615,82134   # 3, tracklet 02
    obs_000024,58372.287,357.20594,8.14448,618,82134   # 3, tracklet 04
    obs_000025,58372.369,357.18470,8.14241,618,82134   # 3, tracklet 04
    obs_000026,58375.310,356.45433,8.05016,621,82134   # 3, tracklet 05
    obs_000027,58375.358,356.44137,8.04837,621,82134   # 3, tracklet 05
    obs_000028,58378.318,355.69765,7.92566,624,82134   # 3, tracklet 06
    obs_000029,58378.387,355.67936,7.92248,624,82134   # 3, tracklet 06

    Object:
    23636: 7 observations
    58177: 15 observations
    82134: 15 observations
    """
    observations = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "test_observations.csv"),
        index_col=False,
        dtype={
            "obs_id": str,
            "truth": str,
        },
    )
    return observations


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
