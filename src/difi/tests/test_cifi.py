import pyarrow as pa
import pyarrow.compute as pc
import pytest

from ..cifi import analyze_observations
from ..observations import Observations
from ..partitions import Partitions


def test_analyze_observations_no_partitions(test_observations):

    all_objects, findable_observations, summary = analyze_observations(
        test_observations,
        partitions=None,
        metric="singletons",
        by_object=True,
        ignore_after_discovery=False,
        max_processes=1,
    )

    num_objects = len(test_observations.object_id.unique())
    # Check that all objects have been correctly analyzed
    assert len(all_objects.object_id.unique()) == num_objects
    assert pc.all(pc.equal(all_objects.partition_id, pa.array(["0"] * num_objects))).as_py()
    assert pc.all(pc.greater_equal(all_objects.num_observatories, pa.array([1] * num_objects))).as_py()
    # Our generator makes 30 obs per object
    assert pc.all(pc.equal(all_objects.num_obs, pa.array([30] * num_objects))).as_py()
    assert pc.all(pc.equal(all_objects.findable, pa.array([True] * num_objects))).as_py()

    assert len(findable_observations) == num_objects

    assert len(summary) == 1
    assert pc.all(pc.equal(summary.id, pa.array(["0"]))).as_py()
    assert pc.all(pc.equal(summary.findable, pa.array([num_objects]))).as_py()
    # Total observations equals table length
    assert summary.observations[0].as_py() == len(test_observations)

    return


def test_analyze_observations_simple_partition(test_observations):

    # Use a detection window of 10 nights to create exactly one window for our 10-day dataset
    partitions = Partitions.create_linking_windows(test_observations.night, detection_window=10)

    all_objects, findable_observations, summary = analyze_observations(
        test_observations,
        partitions=partitions,
        metric="singletons",
        by_object=True,
        ignore_after_discovery=False,
        max_processes=1,
    )

    num_objects = len(test_observations.object_id.unique())
    assert len(all_objects.object_id.unique()) == num_objects
    assert len(findable_observations) == num_objects
    assert len(summary) == 1
    assert summary.findable[0].as_py() == num_objects

    return


def test_analyze_observations_no_observations():
    # Test analyze_observations when the observations data frame is empty
    test_observations = Observations.empty()

    all_objects, findable_observations, summary = analyze_observations(
        test_observations,
    )
    assert len(all_objects) == 0
    assert len(findable_observations) == 0
    assert len(summary) == 0

    return


def test_analyze_observations_raises(test_observations):
    # Test analyze_observations the metric is incorrectly defined
    with pytest.raises(ValueError):
        # Build the all_objects and summary data frames
        all_objects, findable_observations, summary = analyze_observations(
            test_observations,
            metric="wrong_metric",
        )

    return
