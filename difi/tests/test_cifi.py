import os

import pandas as pd
import pytest

from ..cifi import analyzeObservations


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


def test_analyzeObservations_noClasses(test_observations):
    # Test analyzeObservations when no truth classes are given

    all_truths, findable_observations, summary = analyzeObservations(
        test_observations,
        min_obs=5,
        classes=None,
        detection_window=None,
    )

    # Check that all three objects are in the all_truths data frame
    assert all_truths["truth"].nunique() == 3
    for object_id in ["23636", "58177", "82134"]:
        assert object_id in all_truths["truth"].values

    # Check that the number of observations is correct
    assert all_truths["num_obs"].sum() == 30

    # Check that the objects have been correctly marked as findable
    assert all_truths["findable"].sum() == 3

    # Check that the summary data frame is correct, no classes
    # were passed so the length should be one
    assert len(summary) == 1
    assert summary["num_members"].sum() == 3
    assert summary["findable"].sum() == 3
    assert summary["num_obs"].sum() == 30
    assert summary["class"].values[0] == "All"

    all_truths, findable_observations, summary = analyzeObservations(
        test_observations,
        min_obs=10,
        classes=None,
        detection_window=None,
    )

    # Check that all three objects are in the all_truths data frame
    assert all_truths["truth"].nunique() == 3
    for object_id in ["23636", "58177", "82134"]:
        assert object_id in all_truths["truth"].values

    # Check that the number of observations is correct
    assert all_truths["num_obs"].sum() == 30

    # Check that the objects have been correctly marked as findable
    # Only two objects are now findable
    assert all_truths["findable"].sum() == 2
    assert all_truths[all_truths["truth"].isin(["58177", "82134"])]["findable"].sum() == 2

    # Check that the summary data frame is correct, no classes
    # were passed so the length should be one
    assert len(summary) == 1
    assert summary["num_members"].sum() == 3
    assert summary["findable"].sum() == 2
    assert summary["num_obs"].sum() == 30
    assert summary["class"].values[0] == "All"

    return


def test_analyzeObservations_withClassesColumn(test_observations):
    # Test analyzeObservations when a column name is given for the truth classes

    # Add class column to test observations
    for i, object_id in enumerate(["23636", "58177", "82134"]):
        test_observations.loc[test_observations["truth"] == object_id, "class"] = "Class_{}".format(i)

    all_truths, findable_observations, summary = analyzeObservations(
        test_observations,
        min_obs=5,
        classes="class",
        detection_window=None,
    )

    # Check that all three objects are in the all_truths data frame
    assert all_truths["truth"].nunique() == 3
    for object_id in ["23636", "58177", "82134"]:
        assert object_id in all_truths["truth"].values

    # Check that the number of observations is correct
    assert all_truths["num_obs"].sum() == 30

    # Check that the objects have been correctly marked as findable
    assert all_truths["findable"].sum() == 3

    # Check that the summary data frame is correct, there should
    # be a row for each class and a row for the "All" class
    assert len(summary) == 4
    for class_id in ["All", "Class_0", "Class_1", "Class_2"]:
        assert class_id in summary["class"].values

    # Check that the summary data frame is correct for all but
    # the "All" class
    for class_id in ["Class_0", "Class_1", "Class_2"]:
        assert summary[summary["class"] == class_id]["num_members"].sum() == 1
        assert summary[summary["class"] == class_id]["findable"].sum() == 1

    assert summary[summary["class"] == "Class_0"]["num_obs"].sum() == 6
    assert summary[summary["class"] == "Class_1"]["num_obs"].sum() == 10
    assert summary[summary["class"] == "Class_2"]["num_obs"].sum() == 14

    # Check the "All" class
    assert summary[summary["class"] == "All"]["num_obs"].sum() == 30
    assert summary[summary["class"] == "All"]["num_members"].sum() == 3
    assert summary[summary["class"] == "All"]["findable"].sum() == 3

    return


def test_analyzeObservations_withClassesDictionary(test_observations):
    # Test analyzeObservations when a dictionary is given for the truth classes

    # Add class column to test observations
    classes_dict = {
        "Class_0": ["23636"],
        "Class_1": ["58177"],
        "Class_2": ["82134"],
    }

    all_truths, findable_observations, summary = analyzeObservations(
        test_observations,
        min_obs=5,
        classes=classes_dict,
        detection_window=None,
    )

    # Check that all three objects are in the all_truths data frame
    assert all_truths["truth"].nunique() == 3
    for object_id in ["23636", "58177", "82134"]:
        assert object_id in all_truths["truth"].values

    # Check that the number of observations is correct
    assert all_truths["num_obs"].sum() == 30

    # Check that the objects have been correctly marked as findable
    assert all_truths["findable"].sum() == 3

    # Check that the summary data frame is correct, there should
    # be a row for each class and a row for the "All" class
    assert len(summary) == 4
    for class_id in ["All", "Class_0", "Class_1", "Class_2"]:
        assert class_id in summary["class"].values

    # Check that the summary data frame is correct for all but
    # the "All" class
    for class_id in ["Class_0", "Class_1", "Class_2"]:
        assert summary[summary["class"] == class_id]["num_members"].sum() == 1
        assert summary[summary["class"] == class_id]["findable"].sum() == 1

    assert summary[summary["class"] == "Class_0"]["num_obs"].sum() == 6
    assert summary[summary["class"] == "Class_1"]["num_obs"].sum() == 10
    assert summary[summary["class"] == "Class_2"]["num_obs"].sum() == 14

    # Check the "All" class
    assert summary[summary["class"] == "All"]["num_obs"].sum() == 30
    assert summary[summary["class"] == "All"]["num_members"].sum() == 3
    assert summary[summary["class"] == "All"]["findable"].sum() == 3

    return


def test_analyzeObservations_noObservations(test_observations):
    # Test analyzeObservations when the observations data frame is empty
    test_observations = test_observations.drop(test_observations.index)

    with pytest.raises(ValueError):
        # Build the all_truths and summary data frames
        all_truths, findable_observations, summary = analyzeObservations(
            test_observations,
            min_obs=5,
            classes=None,
        )

    return


def test_analyzeObservations_errors(test_observations):
    # Test analyzeObservations the metric is incorrectly defined
    with pytest.raises(ValueError):
        # Build the all_truths and summary data frames
        all_truths, findable_observations, summary = analyzeObservations(
            test_observations,
            min_obs=5,
            metric="wrong_metric",
            classes=None,
        )

    return
