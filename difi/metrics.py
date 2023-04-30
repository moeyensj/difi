from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ["FindabilityMetric", "NightlyLinkagesMetric", "MinObsMetric"]


class FindabilityMetric(ABC):
    @abstractmethod
    def determine_object_findable(self, observations):
        pass

    @staticmethod
    def _compute_windows(observations, detection_window: Optional[int] = None) -> List[Tuple[int, int]]:
        """
        Calculate the minimum and maximum night for windows of observations of length
        detection_window. If detection_window is None, then the entire range of nights
        is used.

        Parameters
        ----------
        observations : `~pandas.DataFrame`
            Observations dataframe containing at least the following columns:
            `obs_id`, `time`, `night`, `truth`.
        detection_window : int, optional
            The number of nights of observations within a single window. If None, then
            the entire range of nights is used.

        Returns
        -------
        windows : list of tuples
            List of tuples containing the start and end night of each window.
        """
        # Calculate the unique number of nights
        nights = observations["night"].unique()
        min_night = nights.min()
        max_night = nights.max()

        windows = []
        # If the detection window is not specified, then use the entire
        # range of nights
        if detection_window is None:
            windows = [(min_night, max_night)]
        else:
            for night in range(min_night, max_night):
                if night + detection_window > max_night:
                    window = (night, max_night)
                else:
                    window = (night, night + detection_window)
                windows.append(window)

        return windows

    @staticmethod
    def _create_window_summary(
        observations: pd.DataFrame, windows: List[Tuple[int, int]], findable: List[pd.DataFrame]
    ):
        """
        Create a summary dataframe of the windows, their start and end nights, the number of observations
        and findable truths in each window.

        Parameters
        ----------
        observations : `~pandas.DataFrame`
            Observations dataframe containing at least the following columns:
            `obs_id`, `time`, `night`, `truth`.
        windows : List[tuples]
            List of tuples containing the start and end night of each window.
        findable : List`~pandas.DataFrame`]
            List of dataframes containing the findable truths for each window.

        Returns
        -------
        window_summary : `~pandas.DataFrame`
            Dataframe containing the window summary.
        """
        windows_dict: Dict[str, List[Any]] = {
            "window_id": [],
            "start_night": [],
            "end_night": [],
            "num_obs": [],
            "num_findable": [],
        }

        for i, (window, findable_i) in enumerate(zip(windows, findable)):
            night_min, night_max = window
            observations_in_window = observations[
                observations["night"].between(night_min, night_max, inclusive="both")
            ]

            windows_dict["window_id"].append(i)
            windows_dict["start_night"].append(night_min)
            windows_dict["end_night"].append(night_max)
            windows_dict["num_obs"].append(len(observations_in_window))
            windows_dict["num_findable"].append(findable_i["findable"].sum())

        return pd.DataFrame(windows_dict)

    def run_by_object(self, observations: pd.DataFrame, windows: List[Tuple[int, int]]) -> List[pd.DataFrame]:
        """
        Run the findability metric on the observations split by objects. For windows where there are many
        observations, this may be faster than running the metric on each window individually
        (with all objects' observations).

        Parameters
        ----------
        observations : `~pandas.DataFrame`
            Observations dataframe containing at least the following columns:
            `obs_id`, `time`, `night`, `truth`.
        windows : List[tuples]
            List of tuples containing the start and end night of each window.

        Returns
        -------
        findable_dfs : List[`~pandas.DataFrame`]
            List of dataframes containing the findable truths and the observations
            that made them findable for each window.
        """
        grouped_observations = observations.groupby(by=["truth"])
        object_observations = [grouped_observations.get_group(x) for x in grouped_observations.groups]

        findable_dfs = []
        for object_obs in object_observations:
            findable_dfs.append(pd.concat(self.run_by_window(object_obs, windows), ignore_index=True))

        return findable_dfs

    def run_by_window(self, observations: pd.DataFrame, windows: List[Tuple[int, int]]) -> List[pd.DataFrame]:
        """
        Run the findability metric on the observations split by windows where each window will
        contain all of the observations within a span of detection_window nights.

        Parameters
        ----------
        observations : `~pandas.DataFrame`
            Observations dataframe containing at least the following columns:
            `obs_id`, `time`, `night`, `truth`.
        windows : List[tuples]
            List of tuples containing the start and end night of each window.

        Returns
        -------
        findable_dfs : List[`~pandas.DataFrame`]
            List of dataframes containing the findable truths and the observations
            that made them findable for each window.
        """
        findable_dfs = []
        for i, window in enumerate(windows):
            night_min, night_max = window
            window_obs = observations[
                (observations["night"] >= night_min) & (observations["night"] <= night_max)
            ]
            window_obs.sort_values(by=["truth", "time"])

            findable = (
                window_obs.groupby(by=["truth"]).apply(self.determine_object_findable).to_frame("findable")
            )
            findable.reset_index(inplace=True, drop=False)
            expanded = pd.DataFrame(
                findable["findable"].values.tolist(), index=findable.index, columns=["findable", "obs_ids"]
            )
            findable = findable[["truth"]].merge(expanded, left_index=True, right_index=True)
            findable.insert(0, "window_id", i)
            findable = findable[findable["findable"] == True]  # noqa: E712
            findable.reset_index(inplace=True, drop=True)
            findable_dfs.append(findable)

        return findable_dfs

    def run(
        self, observations: pd.DataFrame, detection_window: Optional[int] = None, by_object: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the findability metric on the observations.

        Parameters
        ----------
        observations : `~pandas.DataFrame`
            Observations dataframe containing at least the following columns:
            `obs_id`, `time`, `night`, `truth`.
        detection_window : int, optional
            The number of days of observations to consider when
            determining if a truth is findable. If the number of consecutive days
            of observations exceeds the detection_window, then a rolling window
            of size detection_window is used to determine if the truth is findable.
            If None, then the detection_window is the entire range observations.
        by_object : bool, optional
            If True, run the metric on the observations split by objects. For windows where there are many
            observations, this may be faster than running the metric on each window individually
            (with all objects' observations).

        Returns
        -------
        findable : `~pandas.DataFrame`
            A dataframe containing the truth IDs that are findable and a column
            with a list of the observation IDs that made the object findable.
        window_summary : `~pandas.DataFrame`
            A dataframe containing the number of observations, number of findable
            objects and the start and end night of each window.
        """
        observations_sorted = observations.sort_values(by=["time"])

        windows = self._compute_windows(observations_sorted, detection_window)
        if by_object:
            findable_dfs = self.run_by_object(observations_sorted, windows)
        else:
            findable_dfs = self.run_by_window(observations_sorted, windows)

        window_summary = self._create_window_summary(observations_sorted, windows, findable_dfs)

        findable = pd.concat(findable_dfs, ignore_index=True)
        return findable, window_summary


class NightlyLinkagesMetric(FindabilityMetric):
    def __init__(
        self,
        linkage_min_obs: int = 2,
        max_obs_separation: float = 1.5 / 24,
        min_linkage_nights: int = 3,
    ):
        """
        Given observations belonging to one object, finds all observations that are within
        max_obs_separation of each other.

        If linkage_min_obs is 1 then the object is findable if there are at least
        min_linkage_nights of observations.

        Parameters
        ----------
        linkage_min_obs : int, optional
            Minimum number of observations needed to make a intra-night
            linkage (tracklet).
        tracklet_max_time_span : float, optional
            Maximum temporal separation between two observations for them
            to be considered to be in a linkage (in the same units of decimal days).
            Maximum timespan between two observations.
        min_linkage_nights : int, optional
            Minimum number of nights on which a linkage should appear.
        """
        super().__init__()
        self.linkage_min_obs = linkage_min_obs
        self.min_linkage_nights = min_linkage_nights
        self.max_obs_separation = max_obs_separation

    def determine_object_findable(self, observations: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Given observations belonging to one object, finds all observations that are within
        max_obs_separation of each other.

        If linkage_min_obs is 1 then the object is findable if there are at least
        min_linkage_nights of observations.

        Parameters
        ----------
        observations : `~pandas.DataFrame` or `~pandas.core.groupby.generic.DataFrameGroupBy`
            Pandas DataFrame with at least two columns for a single unique truth: observation IDs
            and the observation times in units of decimal days.

        Returns
        -------
        linkage_obs : `~numpy.ndarray`
            Array of observation IDs that made the object findable.
        """
        # If the len of observations is 0 then the object is not findable
        if len(observations) == 0:
            return False, []

        assert observations["truth"].nunique() == 1

        # Exit early if there are not enough observations
        total_required_obs = self.linkage_min_obs * self.min_linkage_nights
        if len(observations) < total_required_obs:
            return False, []
        
        # Grab times and observation IDs from grouped observations
        times = observations["time"].values
        obs_ids = observations["obs_id"].values
        nights = observations["night"].values

        if self.linkage_min_obs > 1:
            # Calculate the time difference between observations
            # (assumes observations are sorted by ascending time)
            delta_t = times[1:] - times[:-1]

            # Create mask that selects all observations within max_obs_separation of
            # each other
            mask = delta_t <= self.max_obs_separation
            start_times = times[np.where(mask)[0]]
            end_times = times[np.where(mask)[0] + 1]

            # Combine times and select all observations match the linkage times
            linkage_times = np.unique(np.concatenate([start_times, end_times]))
            linkage_obs = obs_ids[np.isin(times, linkage_times)]
            linkage_nights, night_counts = np.unique(
                nights[np.isin(obs_ids, linkage_obs)], return_counts=True
            )

            # Make sure that there are enough observations on each night to make a linkage
            valid_nights = linkage_nights[night_counts >= self.linkage_min_obs]
            linkage_obs = obs_ids[np.isin(nights, valid_nights)]
        else:
            # If linkage_min_obs is 1, then we don't need to check for time separation
            # All nights with at least one observation are valid
            valid_nights = np.unique(nights)
            linkage_obs = obs_ids

        # Make sure that the number of observations is still linkage_min_obs * min_linkage_nights
        enough_obs = len(linkage_obs) >= total_required_obs

        # Make sure that the number of unique nights on which a linkage is made
        # is still equal to or greater than the minimum number of nights.
        enough_nights = len(valid_nights) >= self.min_linkage_nights

        if not enough_obs or not enough_nights:
            return False, []
        else:
            return True, linkage_obs.tolist()


class MinObsMetric(FindabilityMetric):
    def __init__(
        self,
        min_obs: int = 5,
    ):
        """
        Create a metric that finds all truths with a minimum of min_obs observations.

        Parameters
        ----------
        min_obs : int, optional
            Minimum number of observations needed to make a truth findable.
        """
        super().__init__()
        self.min_obs = min_obs

    def determine_object_findable(self, observations: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Finds all truths with a minimum of self.min_obs observations and the observations
        that makes them findable.

        Parameters
        ----------
        observations : `~pandas.DataFrame`
            Pandas DataFrame with at least two columns: observation IDs and the truth values
            (the object to which the observation belongs to).

        Returns
        -------
        findable : bool
            Whether or not the object is findable.
        obs_ids : List[str]
            A list of the observation IDs that made the object findable within the window.
        """
        # If the len of observations is 0 then the object is not findable
        if len(observations) == 0:
            return False, []

        assert observations["truth"].nunique() == 1
        if len(observations) >= self.min_obs:
            return True, observations["obs_id"].unique().tolist()
        else:
            return False, []
