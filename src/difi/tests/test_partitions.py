import pytest

from ..partitions import Partitions


def test_Partitions_create_single():
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


def test_Partitions_create_linking_windows():
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


def test_Partitions_create_linking_windows_sliding():
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


def test_Partitions_create_linking_window_raises():
    # Test that the function raises an error when the detection window is less than the min_nights
    nights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    with pytest.raises(ValueError):
        Partitions.create_linking_windows(nights, detection_window=1, min_nights=3, sliding=False)
