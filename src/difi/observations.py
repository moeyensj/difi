import pyarrow.compute as pc
import quivr as qv
from adam_core.time import Timestamp
from quivr.validators import and_, ge, le

from .partitions import Partitions


class Observations(qv.Table):
    id = qv.LargeStringColumn()
    time = Timestamp.as_column()
    ra = qv.Float64Column(validator=and_(ge(0), le(360)))
    dec = qv.Float64Column(validator=and_(ge(-90), le(90)))
    ra_sigma = qv.Float64Column(nullable=True)
    dec_sigma = qv.Float64Column(nullable=True)
    radec_corr = qv.Float64Column(nullable=True, validator=and_(ge(-1), le(1)))
    mag = qv.Float64Column(nullable=True)
    mag_sigma = qv.Float64Column(nullable=True)
    filter = qv.LargeStringColumn(nullable=True)
    observatory_code = qv.LargeStringColumn()
    object_id = qv.LargeStringColumn(nullable=True)
    night = qv.Int64Column()

    def filter_partition(self, partition: Partitions) -> "Observations":
        """
        Filter the observations to only include those within the given partition.

        Parameters
        ----------
        partition : Partitions
            Partition defining the start and end night (both inclusive) of the observations to include.

        Returns
        -------
        Observations
            Observations within the given partition.
        """
        if len(partition) != 1:
            raise ValueError("Only a single partition can be used to filter observations.")
        return self.apply_mask(
            pc.and_(
                pc.greater_equal(self.night, partition.start_night[0].as_py()),
                pc.less_equal(self.night, partition.end_night[0].as_py()),
            )
        )
