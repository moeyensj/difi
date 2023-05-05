import multiprocessing as mp
from abc import ABC, abstractmethod
from itertools import combinations, repeat
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import numpy as np
import pandas as pd
from numba import njit

__all__ = ["FindabilityMetric", "NightlyLinkagesMetric", "MinObsMetric"]

Metrics = TypeVar("Metrics", bound="FindabilityMetric")

__DIFI_ARRAY: np.ndarray


@njit(cache=True)
def haversine_distance(ra1: np.ndarray, dec1: np.ndarray, ra2: np.ndarray, dec2: np.ndarray) -> np.ndarray:
    """
    Calculate the great circle distance between two points on a sphere.

    Parameters
    ----------
    ra1 : `~numpy.ndarray`
        Right ascension of the first point in degrees.
    dec1 : `~numpy.ndarray`
        Declination of the first point in degrees.
    ra2 : `~numpy.ndarray`
        Right ascension of the second point in degrees.
    dec2 : `~numpy.ndarray`
        Declination of the second point in degrees.

    Returns
    -------
    distance : `~numpy.ndarray`
        The great circle distance between the two points in degrees.
    """
    ra1, dec1, ra2, dec2 = map(np.radians, [ra1, dec1, ra2, dec2])

    dlon = ra2 - ra1
    dlat = dec2 - dec1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return np.degrees(c)


@njit(cache=True)
def find_observations_within_max_time_separation(times: np.ndarray, max_obs_separation: float) -> np.ndarray:
    """
    Find all observation IDs that are within max_obs_separation of another observation.

    Parameters
    ----------
    times : `~numpy.ndarray`
        Array of observation times in days.
    max_obs_separation : float
        Maximum time separation between observations in minutes.

    Returns
    -------
    valid_obs : `~numpy.ndarray
        Array of indices of observations that are within max_obs_separation of another observation.
    """
    # Create array of observation indices
    obs_indices = np.arange(len(times), dtype=np.int32)

    # Calculate the time difference between observations
    # (assumes observations are sorted by ascending time)
    delta_t = times[1:] - times[:-1]
    delta_t_minutes = delta_t * 24 * 60

    # Create mask that selects all observations within max_obs_separation of
    # each other
    mask = delta_t_minutes <= max_obs_separation
    start_times_indices = obs_indices[np.where(mask)[0]]
    end_times_indices = obs_indices[np.where(mask)[0] + 1]

    # Combine times and select all observations match the linkage times
    # Pass tuple for numba compatibility
    valid_obs = np.unique(np.concatenate((start_times_indices, end_times_indices)))

    return valid_obs


@njit(cache=True)
def find_observations_beyond_angular_separation(
    nights: np.ndarray, ra: np.ndarray, dec: np.ndarray, min_angular_separation: float
) -> np.ndarray:
    """
    Find all observation IDs that are separated by at least min_angular_separation in a night.

    Parameters
    ----------
    nights : `~numpy.ndarray`
        Array of observation nights.
    ra : `~numpy.ndarray`
        Array of observation right ascensions.
    dec : `~numpy.ndarray`
        Array of observation declinations.
    min_angular_separation : float
        Minimum angular separation between observations in arcseconds.

    Returns
    -------
    valid_obs : `~numpy.ndarray`
        Array of indices of observations that are separated by at least min_angular_separation in a night.
    """
    obs_indices = np.arange(len(nights), dtype=np.int32)
    valid_obs = set()
    for night in nights:
        obs_ids_night = obs_indices[nights == night]
        ra_night = ra[nights == night]
        dec_night = dec[nights == night]

        # Calculate the angular separation between consecutive observations
        distances = haversine_distance(ra_night[:-1], dec_night[:-1], ra_night[1:], dec_night[1:])
        distance_mask = distances >= min_angular_separation / 3600

        valid_obs_start = obs_ids_night[np.where(distance_mask)[0]]
        valid_obs_end = obs_ids_night[np.where(distance_mask)[0] + 1]
        for obs_id in valid_obs_start:
            valid_obs.add(obs_id)
        for obs_id in valid_obs_end:
            valid_obs.add(obs_id)

    return np.array(list(valid_obs), dtype=np.int32)


def select_tracklet_combinations(nights: np.ndarray, min_nights: int) -> List[np.ndarray]:
    """
    Select all possible combinations of tracklets that span at least
    min_nights.

    All detections within one night are considered to be part of the same
    tracklet.

    Parameters
    ----------
    nights : `~numpy.ndarray`
        Array of nights on which observations occur.
    min_nights : int
        Minimum number of nights on which a tracklet must occur.

    Returns
    -------
    linkages : List[`~numpy.ndarray`]
        List of arrays of indices of observations .
    """
    linkages = []
    unique_nights = np.unique(nights)
    obs_indices = np.arange(len(nights), dtype=np.int32)
    for combination in combinations(unique_nights, min_nights):
        linkage_i = []
        for night in combination:
            linkage_i.append(obs_indices[nights == night])
        linkages.append(np.concatenate(linkage_i))
    return linkages


class FindabilityMetric(ABC):
    @abstractmethod
    def determine_object_findable(
        self, observations: np.ndarray, discovery_opportunities: bool = False
    ) -> List[List[str]]:
        pass

    @staticmethod
    def _compute_windows(nights: np.ndarray, detection_window: Optional[int] = None) -> List[Tuple[int, int]]:
        """
        Calculate the minimum and maximum night for windows of observations of length
        detection_window. If detection_window is None, then the entire range of nights
        is used.

        If the detection_window is larger than the range of the observations, then
        the entire range of nights is used.

        Parameters
        ----------
        nights : `~numpy.ndarray`
            Array of nights on which observations occur.
        detection_window : int, optional
            The number of nights of observations within a single window. If None, then
            the entire range of nights is used.

        Returns
        -------
        windows : list of tuples
            List of tuples containing the start and end night of each window.
        """
        # Calculate the unique number of nights
        min_night = nights.min()
        max_night = nights.max()

        # If the detection window is not specified, then use the entire
        # range of nights
        if detection_window is None:
            windows = [(min_night, max_night)]
        elif detection_window > max_night - min_night:
            windows = [(min_night, max_night)]
        else:
            windows = []
            for night in range(min_night, max_night):
                if night + detection_window >= max_night:
                    window = (night, max_night)
                    windows.append(window)
                    break
                else:
                    window = (night, night + detection_window)
                    windows.append(window)

        return windows

    @staticmethod
    def _create_window_summary(
        observations: pd.DataFrame, windows: List[Tuple[int, int]], findable: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create a summary dataframe of the windows, their start and end nights, the number of observations
        and findable objects in each window.

        Parameters
        ----------
        observations : `~pandas.DataFrame`
            Observations dataframe containing at least the following columns:
            `obs_id`, `time`, `night`, `object_id`.
        windows : List[tuples]
            List of tuples containing the start and end night of each window.
        findable : ~pandas.DataFrame`
            Dataframes containing the findable objects for each window.

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

        for i, window in enumerate(windows):
            night_min, night_max = window
            observations_in_window = observations[
                observations["night"].between(night_min, night_max, inclusive="both")
            ]
            findable_i = findable[findable["window_id"] == i]

            windows_dict["window_id"].append(i)
            windows_dict["start_night"].append(night_min)
            windows_dict["end_night"].append(night_max)
            windows_dict["num_obs"].append(len(observations_in_window))
            windows_dict["num_findable"].append(findable_i["findable"].sum().astype(int))

        return pd.DataFrame(windows_dict)

    @staticmethod
    def _sort_by_object_times(objects, obs_ids, times, ra, dec, nights):
        """
        Sort the observations by object and then by time. This is optimal for
        computing the findability metrics by object.

        Parameters
        ----------
        objects : `~numpy.ndarray`
            Array of object values for each observation.
        obs_ids : `~numpy.ndarray`
            Array of observation IDs.
        times : `~numpy.ndarray`
            Array of times of each observation.
        ra : `~numpy.ndarray`
            Array of right ascensions of each observation.
        dec : `~numpy.ndarray`
            Array of declinations of each observation.
        nights : `~numpy.ndarray`
            Array of nights on which observations occur.

        Returns
        -------
        objects : `~numpy.ndarray`
            Sorted array of object values for each observation.
        obs_ids : `~numpy.ndarray`
            Sorted array of observation IDs.
        times : `~numpy.ndarray`
            Sorted array of times of each observation.
        ra : `~numpy.ndarray`
            Sorted array of right ascensions of each observation.
        dec : `~numpy.ndarray`
            Sorted array of declinations of each observation.
        nights : `~numpy.ndarray`
            Sorted array of nights on which observations occur.
        """
        # Sort order is declared in reverse order
        sorted_indices = np.lexsort((times, objects))

        return (
            objects[sorted_indices],
            obs_ids[sorted_indices],
            times[sorted_indices],
            ra[sorted_indices],
            dec[sorted_indices],
            nights[sorted_indices],
        )

    @staticmethod
    def _sort_by_times_object(objects, obs_ids, times, ra, dec, nights):
        """
        Sort the observations by time and then by object. This is optimal for
        computing the findability metrics by windows of time.

        Parameters
        ----------
        objects : `~numpy.ndarray`
            Array of object values for each observation.
        obs_ids : `~numpy.ndarray`
            Array of observation IDs.
        times : `~numpy.ndarray`
            Array of times of each observation.
        ra : `~numpy.ndarray`
            Array of right ascensions of each observation.
        dec : `~numpy.ndarray`
            Array of declinations of each observation.
        nights : `~numpy.ndarray`
            Array of nights on which observations occur.

        Returns
        -------
        objects : `~numpy.ndarray`
            Sorted array of object values for each observation.
        obs_ids : `~numpy.ndarray`
            Sorted array of observation IDs.
        times : `~numpy.ndarray`
            Sorted array of times of each observation.
        ra : `~numpy.ndarray`
            Sorted array of right ascensions of each observation.
        dec : `~numpy.ndarray`
            Sorted array of declinations of each observation.
        nights : `~numpy.ndarray`
            Sorted array of nights on which observations occur.
        """
        # Sort order is declared in reverse order
        sorted_indices = np.lexsort((objects, times))

        return (
            objects[sorted_indices],
            obs_ids[sorted_indices],
            times[sorted_indices],
            ra[sorted_indices],
            dec[sorted_indices],
            nights[sorted_indices],
        )

    @staticmethod
    def _split_by_object(objects: np.ndarray) -> List[np.ndarray]:
        """
        Create a list of arrays of indices for each object in the observations.

        Parameters
        ----------
        objects : `~numpy.ndarray`
            Array of object values for each observation.

        Returns
        -------
        split_by_object_indices : List[`~numpy.ndarray`]
            List of arrays of indices for each object in the observations.
        """
        unique_objects, object_indices = np.unique(objects, return_index=True)
        split_by_object_indices = np.split(np.arange(len(objects)), object_indices[1:])

        return split_by_object_indices

    @staticmethod
    def _store_as_record_array(object_ids, obs_ids, times, ra, dec, nights):
        """
        Store the observations as a record array.

        Parameters
        ----------
        object_ids : `~numpy.ndarray`
            Array of object IDs for each observation.
        obs_ids : `~numpy.ndarray`
            Array of observation IDs.
        times : `~numpy.ndarray`
            Array of times of each observation.
        ra : `~numpy.ndarray`
            Array of right ascensions of each observation.
        dec : `~numpy.ndarray`
            Array of declinations of each observation.
        nights : `~numpy.ndarray`
            Array of nights on which observations occur.

        Returns
        -------
        observations_array : `~numpy.ndarray`
            Record array of observations.
        """
        # Store arrays as global variables
        observations_array = np.empty(
            len(object_ids),
            dtype=[
                ("object_id", object_ids.dtype),
                ("obs_id", obs_ids.dtype),
                ("time", np.float64),
                ("ra", np.float64),
                ("dec", np.float64),
                ("night", np.int64),
            ],
        )

        observations_array["object_id"] = object_ids
        observations_array["obs_id"] = obs_ids
        observations_array["time"] = times
        observations_array["ra"] = ra
        observations_array["dec"] = dec
        observations_array["night"] = nights
        return observations_array

    def _run_object_worker(
        self,
        object_indices: np.ndarray,
        windows: List[Tuple[int, int]],
        discovery_opportunities: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Run the metric on a single object.

        Parameters
        ----------
        observations : `~pandas.DataFrame`
            Observations dataframe containing at least the following columns:
            `obs_id`, `time`, `night`, `object_id`.
        windows : list of tuples
            A list of tuples containing the start and end nights of each window.
        discovery_opportunities : bool, optional
            If True, then return the combinations of observations that made each object findable.
            Note that if the window is large, this can greatly increase the computation time.

        Returns
        -------
        findable : List[Dict[str, Any]]]
            A list of dictionaries containing the findable objects and the observations
            that made them findable for each window.
        """
        findable_dicts = []
        observations = __DIFI_ARRAY[object_indices]
        for i, window in enumerate(windows):
            night_min, night_max = window
            window_obs = observations[
                (observations["night"] >= night_min) & (observations["night"] <= night_max)
            ]

            obs_ids = self.determine_object_findable(
                window_obs, discovery_opportunities=discovery_opportunities
            )
            chances = len(obs_ids)
            if chances > 0:
                findable = {
                    "window_id": i,
                    "object_id": observations["object_id"][0],
                    "findable": 1,
                    "discovery_opportunities": chances,
                    "obs_ids": obs_ids,
                }
            else:
                findable = {
                    "window_id": np.NaN,
                    "object_id": np.NaN,
                    "findable": np.NaN,
                    "discovery_opportunities": np.NaN,
                    "obs_ids": np.NaN,
                }

            findable_dicts.append(findable)

        return findable_dicts

    def run_by_object(
        self,
        observations: pd.DataFrame,
        windows: List[Tuple[int, int]],
        discovery_opportunities: bool = False,
        num_jobs: Optional[int] = 1,
    ) -> List[pd.DataFrame]:
        """
        Run the findability metric on the observations split by objects. For windows where there are many
        observations, this may be faster than running the metric on each window individually
        (with all objects' observations).

        Parameters
        ----------
        observations : `~pandas.DataFrame`
            Observations dataframe containing at least the following columns:
            `obs_id`, `time`, `night`, `object_id`.
        windows : List[tuples]
            List of tuples containing the start and end night of each window.
        discovery_opportunities : bool, optional
            If True, then return the combinations of observations that made each object findable.
            Note that if the window is large, this can greatly increase the computation time.
        num_jobs : int, optional
            The number of jobs to run in parallel. If 1, then run in serial. If None, then use the number of
            CPUs on the machine.

        Returns
        -------
        findable : List[`~pandas.DataFrame`]
            Dataframe containing the findable objects and the observations
            that made them findable for each window.
        """
        # Sort arrays by object and then by times
        objects, obs_ids, times, ra, dec, nights = self._sort_by_object_times(
            observations["object_id"].values.astype(str),
            observations["obs_id"].values.astype(str),
            observations["time"].values,
            observations["ra"].values,
            observations["dec"].values,
            observations["night"].values,
        )

        # Split arrays by object
        split_by_object_indices = self._split_by_object(objects)

        # Store the observations in a global variable so that the worker functions can access them
        global __DIFI_ARRAY
        __DIFI_ARRAY = self._store_as_record_array(objects, obs_ids, times, ra, dec, nights)

        findable_lists: List[List[Dict[str, Any]]] = []
        if num_jobs is None or num_jobs > 1:
            pool = mp.Pool(num_jobs)
            findable_lists = pool.starmap(
                self._run_object_worker,
                zip(split_by_object_indices, repeat(windows), repeat(discovery_opportunities)),
            )

            pool.close()
            pool.join()

        else:
            for object_indices in split_by_object_indices:
                findable_lists.append(
                    self._run_object_worker(
                        object_indices, windows, discovery_opportunities=discovery_opportunities
                    )
                )

        findable_flattened = [item for sublist in findable_lists for item in sublist]

        findable = pd.DataFrame(findable_flattened)
        findable.dropna(subset=["window_id"], inplace=True)
        if len(findable) > 0:
            findable.loc[:, "window_id"] = findable["window_id"].astype(int)
            findable.loc[:, "findable"] = findable["findable"].astype(int)
            findable.loc[:, "discovery_opportunities"] = findable["discovery_opportunities"].astype(int)
            findable.sort_values(by=["window_id", "object_id"], inplace=True, ignore_index=True)

        return findable

    def _run_window_worker(
        self, window_indices, window_id: int, discovery_opportunities: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Run the metric on a single window of observations.

        Parameters
        ----------
        observations : `~pandas.DataFrame`
            Observations dataframe containing at least the following columns:
            `obs_id`, `time`, `night`, `object_id`.
        window_id : int
            The ID of this window.
        discovery_opportunities : bool, optional
            If True, then return the combinations of observations that made each object findable.
            Note that if the window is large, this can greatly increase the computation time.

        Returns
        -------
        findable : List[Dict[str, Any]]]
            A list of dictionaries containing the findable objects and the observations
            that made them findable for each window.
        """
        observations = __DIFI_ARRAY[window_indices]
        if len(observations) == 0:
            return [
                {
                    "window_id": np.nan,
                    "object_id": np.nan,
                    "findable": np.nan,
                    "discovery_opportunities": np.nan,
                    "obs_ids": np.nan,
                }
            ]

        findable_dicts = []
        for object_id in np.unique(observations["object_id"]):

            obs_ids = self.determine_object_findable(
                observations[np.where(observations["object_id"] == object_id)[0]],
                discovery_opportunities=discovery_opportunities,
            )
            chances = len(obs_ids)
            if chances > 0:
                findable = {
                    "window_id": window_id,
                    "object_id": object_id,
                    "findable": 1,
                    "discovery_opportunities": chances,
                    "obs_ids": obs_ids,
                }
            else:
                findable = {
                    "window_id": np.nan,
                    "object_id": np.nan,
                    "findable": np.nan,
                    "discovery_opportunities": np.nan,
                    "obs_ids": np.nan,
                }

            findable_dicts.append(findable)

        return findable_dicts

    def run_by_window(
        self,
        observations: pd.DataFrame,
        windows: List[Tuple[int, int]],
        discovery_opportunities: bool = False,
        num_jobs: Optional[int] = 1,
    ) -> List[pd.DataFrame]:
        """
        Run the findability metric on the observations split by windows where each window will
        contain all of the observations within a span of detection_window nights.

        Parameters
        ----------
        observations : `~pandas.DataFrame`
            Observations dataframe containing at least the following columns:
            `obs_id`, `time`, `night`, `object_id`.
        windows : List[tuples]
            List of tuples containing the start and end night of each window.
        discovery_opportunities : bool, optional
            If True, then return the combinations of observations that made each object findable.
            Note that if the window is large, this can greatly increase the computation time.
        num_jobs : int, optional
            The number of jobs to run in parallel. If 1, then run in serial. If None, then use the number of
            CPUs on the machine.

        Returns
        -------
        findable : List[`~pandas.DataFrame`]
            Dataframe containing the findable objects and the observations
            that made them findable for each window.
        """
        # Sort arrays by times then objects
        objects, obs_ids, times, ra, dec, nights = self._sort_by_times_object(
            observations["object_id"].values.astype(str),
            observations["obs_id"].values.astype(str),
            observations["time"].values,
            observations["ra"].values,
            observations["dec"].values,
            observations["night"].values,
        )

        # Split the observations by window
        split_by_window_indices = []
        for window in windows:
            night_min, night_max = window
            window_indices = np.where((nights >= night_min) & (nights <= night_max))[0]
            split_by_window_indices.append(window_indices)

        # Store the observations in a global variable so that the worker functions can access them
        global __DIFI_ARRAY
        __DIFI_ARRAY = self._store_as_record_array(objects, obs_ids, times, ra, dec, nights)

        findable_lists: List[List[Dict[str, Any]]] = []
        if num_jobs is None or num_jobs > 1:
            pool = mp.Pool(num_jobs)
            findable_lists = pool.starmap(
                self._run_window_worker,
                zip(split_by_window_indices, range(len(windows)), repeat(discovery_opportunities)),
            )
            pool.close()
            pool.join()

        else:
            for i, window_indices in enumerate(split_by_window_indices):
                findable_lists.append(
                    self._run_window_worker(
                        window_indices, i, discovery_opportunities=discovery_opportunities
                    )
                )

        findable_flattened = [item for sublist in findable_lists for item in sublist]

        findable = pd.DataFrame(findable_flattened)
        findable.dropna(subset=["window_id"], inplace=True)
        if len(findable) > 0:
            findable.loc[:, "window_id"] = findable["window_id"].astype(int)
            findable.loc[:, "findable"] = findable["findable"].astype(int)
            findable.loc[:, "discovery_opportunities"] = findable["discovery_opportunities"].astype(int)
            findable.sort_values(by=["window_id", "object_id"], inplace=True, ignore_index=True)

        return findable

    def run(
        self,
        observations: pd.DataFrame,
        detection_window: Optional[int] = None,
        discovery_opportunities: bool = False,
        by_object: bool = False,
        num_jobs: Optional[int] = 1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the findability metric on the observations.

        Parameters
        ----------
        observations : `~pandas.DataFrame`
            Observations dataframe containing at least the following columns:
            `obs_id`, `time`, `night`, `object_id`.
        detection_window : int, optional
            The number of days of observations to consider when
            determining if a object is findable. If the number of consecutive days
            of observations exceeds the detection_window, then a rolling window
            of size detection_window is used to determine if the object is findable.
            If None, then the detection_window is the entire range observations.
        discovery_opportunities : bool, optional
            If True, then return the combinations of observations that made each object findable.
            Note that if the window is large, this can greatly increase the computation time.
        by_object : bool, optional
            If True, run the metric on the observations split by objects. For windows where there are many
            observations, this may be faster than running the metric on each window individually
            (with all objects' observations).
        num_jobs : int, optional
            The number of jobs to run in parallel. If 1, then run in serial. If None, then use the number of
            CPUs on the machine.

        Returns
        -------
        findable : `~pandas.DataFrame`
            A dataframe containing the object IDs that are findable and a column
            with a list of the observation IDs that made the object findable.
        window_summary : `~pandas.DataFrame`
            A dataframe containing the number of observations, number of findable
            objects and the start and end night of each window.
        """
        # Extract arrays
        nights = observations["night"].values
        windows = self._compute_windows(nights, detection_window)

        if by_object:
            observations_sorted = observations.sort_values(by=["object_id", "time"], ascending=True)
            findable = self.run_by_object(
                observations_sorted,
                windows,
                discovery_opportunities=discovery_opportunities,
                num_jobs=num_jobs,
            )
        else:
            observations_sorted = observations.sort_values(by=["time"], ascending=True)
            findable = self.run_by_window(
                observations_sorted,
                windows,
                discovery_opportunities=discovery_opportunities,
                num_jobs=num_jobs,
            )

        window_summary = self._create_window_summary(observations_sorted, windows, findable)
        return findable, window_summary


class NightlyLinkagesMetric(FindabilityMetric):
    def __init__(
        self,
        linkage_min_obs: int = 2,
        max_obs_separation: float = 1.5 / 24,
        min_linkage_nights: int = 3,
        min_obs_angular_separation: float = 1.0,
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
        min_obs_angular_separation : float, optional
            Minimum angular separation between two consecutie observations for them
            to be considered to be in a linkage (in arcseconds).
        """
        super().__init__()
        self.linkage_min_obs = linkage_min_obs
        self.min_linkage_nights = min_linkage_nights
        self.max_obs_separation = max_obs_separation
        self.min_obs_angular_separation = min_obs_angular_separation

    def determine_object_findable(
        self, observations: np.ndarray, discovery_opportunities: bool = False
    ) -> List[List[str]]:
        """
        Given observations belonging to one object, finds all observations that are within
        max_obs_separation of each other.

        If linkage_min_obs is 1 then the object is findable if there are at least
        min_linkage_nights of observations.

        Parameters
        ----------
        observations : `~numpy.ndarray`
            Numpy record array with at least the following columns:
            `object_id`, `obs_id`, `time`, `night`, `ra`, `dec`.
        discovery_opportunities : bool, optional
            If True, return the observation combinations that represent unique discovery
            opportunites.

        Returns
        -------
        obs_ids : List[List[str]]
            List of lists containing observation IDs for each discovery opportunity.
            If discovery_opportunities is False, then the list will contain a single
            list of all valid observation IDs.
        """
        # If the len of observations is 0 then the object is not findable
        if len(observations) == 0:
            return []

        assert len(np.unique(observations["object_id"])) == 1

        # Exit early if there are not enough observations
        total_required_obs = self.linkage_min_obs * self.min_linkage_nights
        if len(observations) < total_required_obs:
            return []

        # Grab times and observation IDs from grouped observations
        times = observations["time"]
        obs_ids = observations["obs_id"]
        nights = observations["night"]
        ra = observations["ra"]
        dec = observations["dec"]

        if self.linkage_min_obs > 1:

            linkage_obs = obs_ids[
                find_observations_within_max_time_separation(
                    times,
                    self.max_obs_separation * (60.0 * 24),
                )
            ]

            # Find the number of observations on each night
            linkage_nights, night_counts = np.unique(
                nights[np.isin(obs_ids, linkage_obs)], return_counts=True
            )

            # Make sure that there are enough observations on each night to make a linkage
            valid_unique_nights = linkage_nights[night_counts >= self.linkage_min_obs]
            valid_mask = np.isin(nights, valid_unique_nights)
            valid_nights = nights[valid_mask]
            linkage_obs = obs_ids[valid_mask]

            if self.min_obs_angular_separation > 0:

                linkage_obs = linkage_obs[
                    find_observations_beyond_angular_separation(
                        valid_nights,
                        ra[valid_mask],
                        dec[valid_mask],
                        self.min_obs_angular_separation,
                    )
                ]
                # If there are no valid observations then the object is not findable
                if len(linkage_obs) == 0:
                    return []

                # Update the valid nights
                valid_nights = nights[np.isin(obs_ids, linkage_obs)]
                valid_unique_nights = np.unique(valid_nights)

        else:
            # If linkage_min_obs is 1, then we don't need to check for time separation
            # All nights with at least one observation are valid
            valid_nights = nights
            valid_unique_nights = np.unique(valid_nights)
            linkage_obs = obs_ids

        # Make sure that the number of observations is still linkage_min_obs * min_linkage_nights
        enough_obs = len(linkage_obs) >= total_required_obs

        # Make sure that the number of unique nights on which a linkage is made
        # is still equal to or greater than the minimum number of nights.
        enough_nights = len(valid_unique_nights) >= self.min_linkage_nights

        if not enough_obs or not enough_nights:
            return []
        else:
            if discovery_opportunities:
                obs_indices = select_tracklet_combinations(valid_nights, self.min_linkage_nights)
                obs_ids = [linkage_obs[ind].tolist() for ind in obs_indices]
                return obs_ids
            else:
                return [linkage_obs.tolist()]


class MinObsMetric(FindabilityMetric):
    def __init__(
        self,
        min_obs: int = 5,
    ):
        """
        Create a metric that finds all objects with a minimum of min_obs observations.

        Parameters
        ----------
        min_obs : int, optional
            Minimum number of observations needed to make a object findable.
        """
        super().__init__()
        self.min_obs = min_obs

    def determine_object_findable(
        self, observations: np.ndarray, discovery_opportunities: bool = False
    ) -> List[List[str]]:
        """
        Finds all objects with a minimum of self.min_obs observations and the observations
        that makes them findable.

        Parameters
        ----------
        observations : `~numpy.ndarray`
            Numpy record array with at least the following columns:
            `object_id`, `obs_id`, `time`, `night`, `ra`, `dec`.
        discovery_opportunities : bool, optional
            If True, return the observation combinations that represent unique discovery
            opportunites.

        Returns
        -------
        obs_ids : List[List[str]]
            List of lists containing observation IDs for each discovery opportunity.
            If discovery_opportunities is False, then the list will contain a single
            list of all valid observation IDs.
        """
        # If the len of observations is 0 then the object is not findable
        if len(observations) == 0:
            return []

        assert len(np.unique(observations["object_id"])) == 1
        if len(observations) >= self.min_obs:
            obs_ids = observations["obs_id"]
            if discovery_opportunities:
                obs_ids = list(combinations(obs_ids, self.min_obs))
                return obs_ids
            else:
                obs_ids = [obs_ids.tolist()]
                return obs_ids
        else:
            return []
