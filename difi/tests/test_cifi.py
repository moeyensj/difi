import pytest

from ..cifi import analyzeObservations


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
        test_observations, min_obs=5, classes="class", detection_window=None, discovery_opportunities=False
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
        discovery_opportunities=False,
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
            test_observations, min_obs=5, classes=None, discovery_opportunities=False
        )

    return


def test_analyzeObservations_errors(test_observations):
    # Test analyzeObservations the metric is incorrectly defined
    with pytest.raises(ValueError):
        # Build the all_truths and summary data frames
        all_truths, findable_observations, summary = analyzeObservations(
            test_observations, min_obs=5, metric="wrong_metric", classes=None, discovery_opportunities=False
        )

    return
