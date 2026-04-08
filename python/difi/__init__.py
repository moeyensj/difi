"""difi: Did I Find It?

Evaluate linkage completeness and purity for astronomical surveys.

v3 is a Rust rewrite with Python bindings via PyO3/maturin.
"""

from difi._core import version
from difi.analyze import analyze_linkages, analyze_observations
from difi.cifi import AllObjects
from difi.difi import AllLinkages, LinkageMembers
from difi.findability import SingletonMetric, TrackletMetric
from difi.metrics import FindableObservations
from difi.observations import Observations
from difi.partitions import Partitions, PartitionSummary

__version__ = version()

__all__ = [
    "version",
    "analyze_observations",
    "analyze_linkages",
    "AllObjects",
    "AllLinkages",
    "FindableObservations",
    "LinkageMembers",
    "Observations",
    "Partitions",
    "PartitionSummary",
    "SingletonMetric",
    "TrackletMetric",
]
