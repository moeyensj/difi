import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from ..cifi import analyzeObservations
from .create_test_data import createTestDataSet

MIN_OBS = range(5, 10)


def test_analyzeObservations_noClasses():
    # --- Test analyzeObservations when no truth classes are given

    # Create test data
    for min_obs in MIN_OBS:
        # Generate test data set
        (
            observations_test,
            all_truths_test,
            linkage_members_test,
            all_linkages_test,
            summary_test,
        ) = createTestDataSet(min_obs, 5, 20)

        # Build the all_truths and summary data frames
        all_truths, findable_observations, summary = analyzeObservations(
            observations_test,
            min_obs=min_obs,
            classes=None,
            detection_window=None,
        )

        # Assert equality among the returned columns
        assert_frame_equal(all_truths, all_truths_test[["truth", "num_obs", "findable"]])
        assert_frame_equal(
            summary,
            summary_test[summary_test["class"] == "All"][["class", "num_members", "num_obs", "findable"]],
        )

    return


def test_analyzeObservations_withClassesColumn():
    # --- Test analyzeObservations when a class column is given

    # Create test data
    for min_obs in MIN_OBS:
        # Generate test data set
        (
            observations_test,
            all_truths_test,
            linkage_members_test,
            all_linkages_test,
            summary_test,
        ) = createTestDataSet(min_obs, 5, 20)

        # Build the all_truths and summary data frames
        all_truths, findable_observations, summary = analyzeObservations(
            observations_test,
            min_obs=min_obs,
            classes="class",
            detection_window=None,
        )

        # Assert equality among the returned columns
        assert_frame_equal(all_truths, all_truths_test[["truth", "num_obs", "findable"]])
        assert_frame_equal(summary, summary_test[["class", "num_members", "num_obs", "findable"]])

    return


def test_analyzeObservations_withClassesDictionary():
    # --- Test analyzeObservations when a class dictionary is given

    # Create test data
    for min_obs in MIN_OBS:
        # Generate test data set
        (
            observations_test,
            all_truths_test,
            linkage_members_test,
            all_linkages_test,
            summary_test,
        ) = createTestDataSet(min_obs, 5, 20)

        classes = {}
        for c in ["blue", "red", "green"]:
            classes[c] = observations_test[observations_test["truth"].str.contains(c)]["truth"].unique()

        # Build the all_truths and summary data frames
        all_truths, findable_observations, summary = analyzeObservations(
            observations_test,
            min_obs=min_obs,
            classes=classes,
            detection_window=None,
        )

        # Assert equality among the returned columns
        assert_frame_equal(all_truths, all_truths_test[["truth", "num_obs", "findable"]])
        assert_frame_equal(summary, summary_test[["class", "num_members", "num_obs", "findable"]])

    return


def test_analyzeObservations_noObservations():
    # --- Test analyzeObservations when the observations data frame is empty

    (
        observations_test,
        all_truths_test,
        linkage_members_test,
        all_linkages_test,
        summary_test,
    ) = createTestDataSet(5, 5, 20)
    observations_test = observations_test.drop(observations_test.index)

    with pytest.raises(ValueError):
        # Build the all_truths and summary data frames
        all_truths, findable_observations, summary = analyzeObservations(
            observations_test,
            min_obs=5,
            classes=None,
        )

    return


def test_analyzeObservations_errors():
    # --- Test analyzeObservations the metric is incorrectly defined

    (
        observations_test,
        all_truths_test,
        linkage_members_test,
        all_linkages_test,
        summary_test,
    ) = createTestDataSet(5, 5, 20)

    with pytest.raises(ValueError):
        # Build the all_truths and summary data frames
        all_truths, findable_observations, summary = analyzeObservations(
            observations_test,
            min_obs=5,
            metric="wrong_metric",
            classes=None,
        )

    return


def test_analyzeObservations_metrics():
    # --- Test analyzeObservations with built in metrics (this only tests that no errors
    # are raised when calling them) actual metric tests are in test_metrics.py

    column_mapping = {
        "obs_id": "obs_id",
        "truth": "truth",
        "time": "time",
        "night": "night",
    }

    # Create test data
    (
        observations_test,
        all_truths_test,
        linkage_members_test,
        all_linkages_test,
        summary_test,
    ) = createTestDataSet(5, 5, 20)
    observations_test["night"] = np.arange(0, len(observations_test))
    observations_test["time"] = np.arange(0, len(observations_test))

    # Build the all_truths and summary data frames and make sure no metric errors are returned
    all_truths, findable_observations, summary = analyzeObservations(
        observations_test,
        metric="min_obs",
        min_obs=5,
        classes=None,
        column_mapping=column_mapping,
    )

    # Build the all_truths and summary data frames and make sure no metric errors are returned
    all_truths, findable_observations, summary = analyzeObservations(
        observations_test,
        metric="nightly_linkages",
        linkage_min_obs=1,
        max_obs_separation=10,
        min_linkage_nights=1,
        classes=None,
        column_mapping=column_mapping,
    )
    return


def test_analyzeObservations_customMetric():
    # --- Test analyzeObservations when a custom metric is given

    def _customMetric(observations, min_observations=5, column_mapping={}):
        # Same as minObs metric, just testing if a custom made function can be sent to analyzeObservations
        object_num_obs = observations[column_mapping["truth"]].value_counts().to_frame("num_obs")
        object_num_obs = object_num_obs[object_num_obs["num_obs"] >= min_obs]
        findable_objects = object_num_obs.index.values
        findable_observations = observations[observations[column_mapping["truth"]].isin(findable_objects)]
        findable = (
            findable_observations.groupby(by=[column_mapping["truth"]])[column_mapping["obs_id"]]
            .apply(np.array)
            .to_frame("obs_ids")
        )
        findable.reset_index(inplace=True, drop=False)
        return findable

    # Create test data
    for min_obs in MIN_OBS:
        # Generate test data set
        (
            observations_test,
            all_truths_test,
            linkage_members_test,
            all_linkages_test,
            summary_test,
        ) = createTestDataSet(min_obs, 5, 20)

        # Build the all_truths and summary data frames
        all_truths, findable_observations, summary = analyzeObservations(
            observations_test,
            metric=_customMetric,
            min_observations=min_obs,
            classes=None,
            detection_window=None,
        )

        # Assert equality among the returned columns
        assert_frame_equal(all_truths, all_truths_test[["truth", "num_obs", "findable"]])
        assert_frame_equal(
            summary,
            summary_test[summary_test["class"] == "All"][["class", "num_members", "num_obs", "findable"]],
        )

    return
