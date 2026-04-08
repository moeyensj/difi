"""High-level analysis functions that wrap the Rust core.

Each function accepts either file paths (str/Path) or quivr Table objects.
File paths are passed directly to Rust; quivr objects are serialized to
temporary Parquet files first.
"""

import os
import tempfile
from pathlib import Path
from typing import Tuple, Union

from difi._core import (
    analyze_linkages as _analyze_linkages_rust,
    analyze_observations as _analyze_observations_rust,
)
from difi.cifi import AllObjects
from difi.difi import AllLinkages, LinkageMembers
from difi.metrics import FindableObservations, SingletonMetric, TrackletMetric
from difi.observations import Observations
from difi.partitions import PartitionSummary

# Types accepted for observations / linkage_members arguments
ObservationsInput = Union[str, os.PathLike, Observations]
LinkageMembersInput = Union[str, os.PathLike, LinkageMembers]
MetricInput = Union[str, SingletonMetric, TrackletMetric]


def analyze_observations(
    observations: ObservationsInput,
    metric: MetricInput = "singletons",
    min_obs: int = 6,
    min_nights: int = 3,
) -> Tuple[AllObjects, FindableObservations, PartitionSummary]:
    """Can I Find It? -- Determine findability of objects.

    Parameters
    ----------
    observations : str, Path, or Observations
        Path to observations Parquet file, or a quivr Observations table.
    metric : str or metric instance
        "singletons" or "tracklets".
    min_obs : int
        Minimum observations for findability.
    min_nights : int
        Minimum nights for findability.

    Returns
    -------
    all_objects : AllObjects
    findable : FindableObservations
    partition_summary : PartitionSummary
    """
    obs_path = _resolve_path(observations, "observations")
    metric_name = _resolve_metric_name(metric)

    return _analyze_observations_rust(
        obs_path,
        metric=metric_name,
        min_obs=min_obs,
        min_nights=min_nights,
    )


def analyze_linkages(
    observations: ObservationsInput,
    linkage_members: LinkageMembersInput,
    min_obs: int = 6,
    contamination_percentage: float = 20.0,
    metric: MetricInput = "singletons",
    min_nights: int = 3,
) -> Tuple[AllObjects, AllLinkages, PartitionSummary]:
    """Did I Find It? -- Classify linkages and compute completeness.

    Parameters
    ----------
    observations : str, Path, or Observations
        Path to observations Parquet file, or a quivr Observations table.
    linkage_members : str, Path, or LinkageMembers
        Path to linkage members Parquet file, or a quivr LinkageMembers table.
    min_obs : int
        Minimum observations for "found".
    contamination_percentage : float
        Max contamination % for "contaminated" classification.
    metric : str or metric instance
        Findability metric for CIFI phase.
    min_nights : int
        Minimum nights for findability.

    Returns
    -------
    all_objects : AllObjects
    all_linkages : AllLinkages
    partition_summary : PartitionSummary
    """
    obs_path = _resolve_path(observations, "observations")
    lm_path = _resolve_path(linkage_members, "linkage_members")
    metric_name = _resolve_metric_name(metric)

    return _analyze_linkages_rust(
        obs_path,
        lm_path,
        min_obs=min_obs,
        contamination_percentage=contamination_percentage,
        metric=metric_name,
        min_nights=min_nights,
    )


def _resolve_path(input, name: str) -> str:
    """Convert a path or quivr table to a file path string."""
    if isinstance(input, (str, os.PathLike)):
        return str(input)

    # quivr Table — write to a temp file
    # The temp file persists until the process exits (no auto-cleanup needed
    # since these are small metadata-only writes for the Rust core to read)
    tmp = tempfile.NamedTemporaryFile(suffix=f"_{name}.parquet", delete=False)
    tmp.close()
    input.to_parquet(tmp.name)
    return tmp.name


def _resolve_metric_name(metric) -> str:
    if isinstance(metric, str):
        return metric
    if isinstance(metric, SingletonMetric):
        return "singletons"
    if isinstance(metric, TrackletMetric):
        return "tracklets"
    raise ValueError(f"Unknown metric type: {type(metric)}")
