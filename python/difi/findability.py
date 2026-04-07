"""Findability metric configuration classes.

Each metric defines what observation pattern makes an object "findable".
Pass a configured metric instance to analyze_observations() or
analyze_linkages().
"""

import json
from dataclasses import dataclass, field


@dataclass
class SingletonMetric:
    """Object is findable if it has enough individual detections.

    Parameters
    ----------
    min_obs : int
        Minimum number of observations.
    min_nights : int
        Minimum number of distinct observing nights.
    min_nightly_obs_in_min_nights : int
        When exactly min_nights nights are present, each night must have
        at least this many observations.
    """

    min_obs: int = 6
    min_nights: int = 3
    min_nightly_obs_in_min_nights: int = 1

    def to_json(self) -> str:
        return json.dumps(
            {
                "type": "singletons",
                "min_obs": self.min_obs,
                "min_nights": self.min_nights,
                "min_nightly_obs_in_min_nights": self.min_nightly_obs_in_min_nights,
            }
        )


@dataclass
class TrackletMetric:
    """Object is findable if it has intra-night tracklets on enough nights.

    A tracklet is a group of observations within max_obs_separation time
    that show angular motion >= min_obs_angular_separation.

    Parameters
    ----------
    tracklet_min_obs : int
        Minimum observations per tracklet.
    max_obs_separation : float
        Maximum time between consecutive tracklet observations (days).
    min_linkage_nights : int
        Minimum number of distinct nights with valid tracklets.
    min_obs_angular_separation : float
        Minimum angular separation within a tracklet (arcseconds).
    """

    tracklet_min_obs: int = 2
    max_obs_separation: float = 1.5 / 24.0
    min_linkage_nights: int = 3
    min_obs_angular_separation: float = 1.0

    def to_json(self) -> str:
        return json.dumps(
            {
                "type": "tracklets",
                "tracklet_min_obs": self.tracklet_min_obs,
                "max_obs_separation": self.max_obs_separation,
                "min_linkage_nights": self.min_linkage_nights,
                "min_obs_angular_separation": self.min_obs_angular_separation,
            }
        )


# Default metric instances
_METRIC_DEFAULTS = {
    "singletons": SingletonMetric,
    "tracklets": TrackletMetric,
}


def resolve_metric_json(metric) -> str:
    """Convert a metric name, class, or instance to a JSON config string."""
    if isinstance(metric, str):
        if metric not in _METRIC_DEFAULTS:
            raise ValueError(f"Unknown metric: {metric!r}. Choose from: {list(_METRIC_DEFAULTS)}")
        return _METRIC_DEFAULTS[metric]().to_json()
    if isinstance(metric, (SingletonMetric, TrackletMetric)):
        return metric.to_json()
    raise TypeError(f"Expected str, SingletonMetric, or TrackletMetric, got {type(metric)}")
