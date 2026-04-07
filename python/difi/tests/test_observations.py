from adam_core.time import Timestamp

from ..observations import Observations
from ..partitions import Partitions


def test_Observations_filter_partition() -> None:
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

    partition = Partitions.from_kwargs(
        id=["1"],
        start_night=[59001],
        end_night=[59002],
    )

    filtered_observations = observations.filter_partition(partition)
    assert len(filtered_observations) == 2
    assert filtered_observations.id.to_pylist() == ["2", "3"]

    partition = Partitions.from_kwargs(
        id=["1"],
        start_night=[59000],
        end_night=[59004],
    )
    filtered_observations = observations.filter_partition(partition)
    assert len(filtered_observations) == 5
    assert filtered_observations.id.to_pylist() == ["1", "2", "3", "4", "5"]
