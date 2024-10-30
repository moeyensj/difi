from typing import Optional

import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
from quivr.validators import and_, ge, le


class Partitions(qv.Table):
    #: Partition ID - Represents a division of the data whether by a linking window
    #: or some other criteria.
    id = qv.LargeStringColumn()
    #: Start night of the partition.
    start_night = qv.Int64Column()
    #: End night of the partition (inclusive).
    end_night = qv.Int64Column()

    @classmethod
    def create_single(cls, nights: pa.Array) -> "Partitions":
        """
        Create a single partition that spans the given nights.

        Parameters
        ----------
        nights : pa.Array
            Array of nights to span (does not need to be sorted or unique).

        Returns
        -------
        paritions : Partitions
            Single partition spanning the given nights.
        """
        return cls.from_kwargs(
            id=["0"],
            start_night=[pc.min(nights).as_py()],
            end_night=[pc.max(nights).as_py()],
        )

    @classmethod
    def create_linking_windows(
        cls,
        nights: pa.Array,
        detection_window: Optional[int] = None,
        min_nights: Optional[int] = None,
        sliding: bool = False,
    ) -> "Partitions":
        """
        Create partitions that represent linking windows of observations. If detection_window is
        None, then the entire range of nights is used. If detection_window is larger than the range
        of the observations, then the entire range of nights is used. If detection_window is smaller
        than the range of the observations, then one of two things can happen. If sliding is True,
        then the windows will shift by one night at a time with a minimum length of min_nights. If
        sliding is False, then the windows will be non-overlapping with a minimum length of detection_window.

        Parameters
        ----------
        nights : pa.Array
            Array of nights on which observations occur.
        detection_window : int, optional
            The number of nights of observations within a single window. If None, then the entire range
            of nights is used as a single window (both min_nights and sliding are ignored).
        min_nights : int, optional
            Minimum length of a detection window measured from the earliest night (only applies when
            sliding is True). If the detection window is set to 15 but min_nights is 3
            then the first window will be 3 nights long and the second window will be 4 nights long, etc...
            Once the detection_window length has been reached then all windows will be of length detection_window.
        sliding : bool, optional
            If True, then the windows will be sliding windows of length detection_window. If False, then the
            windows will be non-overlapping windows of length detection window.

        Returns
        -------
        windows : Partitions
            Table detailing the start and end night of each window.
        """
        # Calculate the unique number of nights
        min_night = pc.min(nights).as_py()
        max_night = pc.max(nights).as_py()

        # If the detection window is not specified, then use the entire
        # range of nights
        if detection_window is None:
            partitions = Partitions.create_single(nights)
            return partitions

        if detection_window >= (max_night - min_night):
            detection_window = max_night - min_night + 1

        if min_nights is None:
            min_nights = detection_window

        if detection_window < min_nights:
            raise ValueError("Detection window must be greater than or equal to min_nights.")

        partitions = cls.empty()

        if sliding:

            i = 0
            start_night = min_night
            end_night = start_night + min_nights - 1
            while True:
                if end_night > max_night:
                    break

                partitions_i = Partitions.from_kwargs(
                    id=[f"{i}"], start_night=[start_night], end_night=[end_night]
                )
                partitions = qv.concatenate([partitions, partitions_i])

                i += 1
                end_night += 1
                if end_night - detection_window == start_night:
                    start_night += 1

        else:
            for i, start_night in enumerate(range(min_night, max_night, detection_window)):

                # End night is inclusive
                end_night = start_night + detection_window - 1
                if end_night > max_night:
                    end_night = max_night

                partitions_i = Partitions.from_kwargs(
                    id=[f"{i}"], start_night=[start_night], end_night=[end_night]
                )
                partitions = qv.concatenate([partitions, partitions_i])

        return partitions


class PartitionSummary(qv.Table):
    #: Partition ID - Represents a division of the data whether by a linking window
    #: or some other criteria.
    id = qv.LargeStringColumn()
    #: Start night of the partition.
    start_night = qv.Int64Column()
    #: End night of the partition (inclusive).
    end_night = qv.Int64Column()
    #: Number of observations in the partition.
    observations = qv.Int64Column()
    #: Number of unique objects that are deemed findable in the partition.
    findable = qv.Int64Column(nullable=True)
    #: Number of unique objects that were found in the partition.
    found = qv.Int64Column(nullable=True)
    #: The completeness of the partition (found / findable).
    completeness = qv.Float64Column(nullable=True, validator=and_(ge(0), le(1)))
    #: Number of pure linkages of known objects.
    pure_known = qv.Int64Column(nullable=True)
    #: Number of pure linkages of unassociated observations.
    pure_unknown = qv.Int64Column(nullable=True)
    #: Number of contaminated linkages.
    contaminated = qv.Int64Column(nullable=True)
    #: Number of mixed linkages.
    mixed = qv.Int64Column(nullable=True)
