import pytest
from adam_core.time import Timestamp

from ..metrics import FindableObservations
from ..observations import Observations
from ..partitions import Partitions, PartitionSummary


def test_Partitions_create_single() -> None:
    # Test the creation of a partition that spans the full
    # range of nights
    nights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    partitions = Partitions.create_single(nights)
    assert partitions.id[0].as_py() == "0"
    assert partitions.start_night[0].as_py() == 1
    assert partitions.end_night[0].as_py() == 10

    nights = [30, 20, 100]
    partitions = Partitions.create_single(nights)
    assert len(partitions) == 1
    assert partitions.id[0].as_py() == "0"
    assert partitions.start_night[0].as_py() == 20
    assert partitions.end_night[0].as_py() == 100


def test_Partitions_create_linking_windows() -> None:
    # Test the creation of a linking window sized partitions
    nights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    partitions = Partitions.create_linking_windows(nights, detection_window=5, sliding=False)

    assert len(partitions) == 2
    assert partitions.id.to_pylist() == ["0", "1"]
    assert partitions.start_night.to_pylist() == [1, 6]
    assert partitions.end_night.to_pylist() == [5, 10]

    partitions = Partitions.create_linking_windows(nights, detection_window=None)
    assert len(partitions) == 1
    assert partitions.id.to_pylist() == ["0"]
    assert partitions.start_night.to_pylist() == [1]
    assert partitions.end_night.to_pylist() == [10]


def test_Partitions_create_linking_windows_sliding() -> None:
    # Test the creation of a linking window sized partitions
    nights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    partitions = Partitions.create_linking_windows(nights, detection_window=5, min_nights=3, sliding=True)

    assert len(partitions) == 8
    assert partitions.id.to_pylist() == ["0", "1", "2", "3", "4", "5", "6", "7"]
    assert partitions.start_night.to_pylist() == [1, 1, 1, 2, 3, 4, 5, 6]
    assert partitions.end_night.to_pylist() == [3, 4, 5, 6, 7, 8, 9, 10]

    partitions = Partitions.create_linking_windows(nights, detection_window=3, min_nights=3, sliding=True)
    assert len(partitions) == 8
    assert partitions.id.to_pylist() == ["0", "1", "2", "3", "4", "5", "6", "7"]
    assert partitions.start_night.to_pylist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert partitions.end_night.to_pylist() == [3, 4, 5, 6, 7, 8, 9, 10]

    partitions = Partitions.create_linking_windows(nights, detection_window=3, min_nights=None, sliding=True)
    assert len(partitions) == 8
    assert partitions.id.to_pylist() == ["0", "1", "2", "3", "4", "5", "6", "7"]
    assert partitions.start_night.to_pylist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert partitions.end_night.to_pylist() == [3, 4, 5, 6, 7, 8, 9, 10]


def test_Partitions_create_linking_window_raises() -> None:
    # Test that the function raises an error when the detection window is less than the min_nights
    nights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    with pytest.raises(ValueError):
        Partitions.create_linking_windows(nights, detection_window=1, min_nights=3, sliding=False)


def test_PartitionSummary_create() -> None:
    # Test filtering observations by partition
    observations = Observations.from_kwargs(
        id=["1", "2", "3", "4", "5"],
        time=Timestamp.from_mjd(
            [59000.0, 59001.0, 59002.0, 59003.0, 59004.0],
            scale="utc",
        ),
        ra=[0.0, 0.0, 0.0, 0.0, 0.0],
        dec=[0.0, 0.0, 0.0, 0.0, 0.0],
        observatory_code=["000", "000", "000", "000", "000"],
        night=[59000, 59001, 59002, 59003, 59004],
    )

    partitions = Partitions.from_kwargs(
        id=["1", "2"],
        start_night=[59000, 59003],
        end_night=[59002, 59004],
    )

    partition_summary = PartitionSummary.create(observations, partitions)
    assert len(partition_summary) == 2
    assert partition_summary.id.to_pylist() == ["1", "2"]
    assert partition_summary.start_night.to_pylist() == [59000, 59003]
    assert partition_summary.end_night.to_pylist() == [59002, 59004]
    assert partition_summary.observations.to_pylist() == [3, 2]


def test_PartitionSummary_update_findable():
    # Test that adding in information on what objects are findable
    # gets updated correctly
    observations = Observations.from_kwargs(
        id=["1", "2", "3", "4", "5"],
        time=Timestamp.from_mjd(
            [59000.0, 59001.0, 59002.0, 59003.0, 59004.0],
            scale="utc",
        ),
        ra=[0.0, 0.0, 0.0, 0.0, 0.0],
        dec=[0.0, 0.0, 0.0, 0.0, 0.0],
        observatory_code=["000", "000", "000", "000", "000"],
        night=[59000, 59001, 59002, 59003, 59004],
        object_id=["1", "1", "2", "2", "2"],
    )

    partitions = Partitions.from_kwargs(
        id=["1", "2"],
        start_night=[59000, 59003],
        end_night=[59002, 59004],
    )

    partition_summary = PartitionSummary.create(observations, partitions)

    findable_observations = FindableObservations.from_kwargs(
        partition_id=["1", "2"],
        object_id=["1", "2"],
        discovery_night=[59001, 59004],
        obs_ids=[
            ["1", "2"],
            ["4", "5"],
        ],
    )

    partition_summary = partition_summary.update_findable(findable_observations)
    assert partition_summary.id.to_pylist() == ["1", "2"]
    assert partition_summary.findable.to_pylist() == [1, 1]
