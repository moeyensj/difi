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
from difi.findability import SingletonMetric, TrackletMetric, resolve_metric_json
from difi.metrics import FindableObservations
from difi.observations import Observations
from difi.partitions import PartitionSummary

# Types accepted for arguments
ObservationsInput = Union[str, os.PathLike, Observations]
LinkageMembersInput = Union[str, os.PathLike, LinkageMembers]
MetricInput = Union[str, SingletonMetric, TrackletMetric]


def analyze_observations(
    observations: ObservationsInput,
    metric: MetricInput = SingletonMetric(),
) -> Tuple[AllObjects, FindableObservations, PartitionSummary]:
    """Can I Find It? -- Determine findability of objects.

    Parameters
    ----------
    observations : str, Path, or Observations
        Path to observations Parquet file, or a quivr Observations table.
    metric : str, SingletonMetric, or TrackletMetric
        Findability metric. Pass a string ("singletons", "tracklets") for
        defaults, or a configured metric instance.

    Returns
    -------
    result : dict
        Analysis results including object counts and findability.
    """
    obs_path = _resolve_path(observations, "observations")
    metric_json = resolve_metric_json(metric)

    return _analyze_observations_rust(obs_path, metric_json)


def analyze_linkages(
    observations: ObservationsInput,
    linkage_members: LinkageMembersInput,
    metric: MetricInput = SingletonMetric(),
    min_obs: int = 6,
    contamination_percentage: float = 20.0,
) -> Tuple[AllObjects, AllLinkages, PartitionSummary]:
    """Did I Find It? -- Classify linkages and compute completeness.

    Parameters
    ----------
    observations : str, Path, or Observations
        Path to observations Parquet file, or a quivr Observations table.
    linkage_members : str, Path, or LinkageMembers
        Path to linkage members Parquet file, or a quivr LinkageMembers table.
    metric : str, SingletonMetric, or TrackletMetric
        Findability metric for CIFI phase.
    min_obs : int
        Minimum observations for a linkage to be considered "found".
    contamination_percentage : float
        Max contamination % for "contaminated" classification.

    Returns
    -------
    result : dict
        Analysis results including linkage classifications and completeness.
    """
    obs_path = _resolve_path(observations, "observations")
    lm_path = _resolve_path(linkage_members, "linkage_members")
    metric_json = resolve_metric_json(metric)

    return _analyze_linkages_rust(
        obs_path,
        lm_path,
        metric_json,
        min_obs=min_obs,
        contamination_percentage=contamination_percentage,
    )


def _resolve_path(input, name: str) -> str:
    """Convert a path or quivr table to a file path string."""
    if isinstance(input, (str, os.PathLike)):
        return str(input)

    # quivr Table — write to a temp file
    tmp = tempfile.NamedTemporaryFile(suffix=f"_{name}.parquet", delete=False)
    tmp.close()
    input.to_parquet(tmp.name)
    return tmp.name
