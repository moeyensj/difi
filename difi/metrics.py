import hashlib
import multiprocessing as mp
import warnings
from abc import ABC, abstractmethod
from itertools import combinations, repeat
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import numpy as np
import pandas as pd
from numba import njit

__all__ = ["FindabilityMetric", "NightlyLinkagesMetric", "MinObsMetric"]

Metrics = TypeVar("Metrics", bound="FindabilityMetric")


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


def calculate_random_seed_from_object_id(object_id: str) -> int:
    """
    Caculate a random seed from an object ID.

    Parameters
    ----------
    object_id : str
        Object ID.

    Returns
    -------
    seed : int
        Random seed.
    """
    seed = int(hashlib.md5(object_id.encode("utf-8")).hexdigest(), base=16) % (10**8)
    return seed


def apply_discovery_probability(
    discovery_obs_ids: List[List[str]], object_id: str, probability: float = 1.0
) -> Tuple[List[str], List[List[str]]]:
    """
    Given a list of lists containing observation IDs, apply a discovery probability to each list.

    Parameters
    ----------
    discovery_obs_ids : List[List[str]]
        List of lists containing observation IDs for each discovery chance.
    object_id : str
        Object ID. Used to calculate a random seed.
    probability : float, optional
        Probability of a discovery chance being selected, by default 1.0. If less
        than 1.0, a random number is generated for each discovery chance and if
        the random number is greater than the probability, the discovery chance
        is removed.

    Returns
    -------
    obs_ids_unique : List[str]
        List of unique observation IDs remaining in the discovery chances.
    discovery_obs_ids : List[List[str]]
        List of lists containing observation IDs for each discovery chance.
    """
    if probability < 1.0:
        random_seed = calculate_random_seed_from_object_id(object_id)
        rng = np.random.default_rng(random_seed)
        p_chance = rng.random(len(discovery_obs_ids))
        del_mask = np.where(p_chance > probability)[0]
        discovery_obs_ids = [discovery_obs_ids[i] for i in range(len(discovery_obs_ids)) if i not in del_mask]
    else:
        discovery_obs_ids = discovery_obs_ids

    # Get the unique observation IDs
    if len(discovery_obs_ids) > 0:
        obs_ids_unique = np.unique(np.concatenate(discovery_obs_ids)).tolist()
    else:
        obs_ids_unique = []

    return obs_ids_unique, discovery_obs_ids


class FindabilityMetric(ABC):
    @abstractmethod
    def determine_object_findable(
        self, observations: np.ndarray, windows: Optional[List[Tuple[int, int]]] = None
    ) -> Dict[int, List[List[str]]]:
        pass

    @staticmethod
    def _compute_windows(
        nights: np.ndarray, detection_window: Optional[int] = None, min_nights: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        """
        Calculate the minimum and maximum night for windows of observations of length
        detection_window. If detection_window is None, then the entire range of nights
        is used.

        If the detection_window is larger than the range of the observations, then
        the entire range of nights is used. If the detection_window is smaller than
        the range of the observations, then the windows are calculated such that there
        is a rolling window of length detection_window that starts at the earliest night
        and ends at the latest night.

        Parameters
        ----------
        nights : `~numpy.ndarray`
            Array of nights on which observations occur.
        detection_window : int, optional
            The number of nights of observations within a single window. If None, then
            the entire range of nights is used.
        min_nights : int, optional
            Minimum length of a detection window measured from the earliest night. If
            the detection window is set to 15 but min_nights is 3 then the first window
            will be 3 nights long and the second window will be 4 nights long, etc... Once
            the detection_window length has been reached then all windows will be of length
            detection_window.

        Returns
        -------
        windows : list of tuples
            List of tuples containing the start and end night of each window (inclusive).
        """
        # Calculate the unique number of nights
        min_night = nights.min()
        max_night = nights.max()

        # If the detection window is not specified, then use the entire
        # range of nights
        if detection_window is None:
            windows = [(min_night, max_night)]
            return windows
        elif isinstance(detection_window, int):
            detection_window_ = detection_window
        else:
            raise TypeError("detection_window must be an integer or None.")

        if detection_window_ >= (max_night - min_night):
            detection_window_ = max_night - min_night + 1

        if min_nights is None:
            min_nights_ = detection_window_
        elif isinstance(min_nights, int):
            min_nights_ = min_nights
        else:
            raise TypeError("min_nights must be an integer or None.")

        windows = []
        for night in range(min_night, max_night):
            night_start = night
            night_end = night + detection_window_ - 1  # inclusive

            if night_start == min_night:
                assert min_nights_ <= detection_window_
                for i in range(min_nights_ - 1, detection_window_):
                    windows.append((night_start, night_start + i))
            else:
                window = (night_start, night_end)
                if night_end > max_night:
                    break
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

        # Extract the counts of observations in each night
        nights = observations["night"].values
        nights, night_counts = np.unique(nights, return_counts=True)

        # Sort the nights and night counts
        night_counts = night_counts[np.argsort(nights)]
        nights = np.sort(nights)

        for i, window in enumerate(windows):
            night_min, night_max = window

            observations_in_window = night_counts[
                np.where((nights >= night_min) & (nights <= night_max))
            ].sum()

            findable_i = findable[findable["window_id"] == i]

            windows_dict["window_id"].append(i)
            windows_dict["start_night"].append(night_min)
            windows_dict["end_night"].append(night_max)
            windows_dict["num_obs"].append(observations_in_window)
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
    def _split_by_object(objects: np.ndarray) -> List[slice]:
        """
        Create a list of slices for each object in the observations.

        Parameters
        ----------
        objects : `~numpy.ndarray`
            Array of object values for each observation.

        Returns
        -------
        split_by_object_slices : List[slice]
            List of slices for each object in the observations.
        """
        unique_objects, object_indices = np.unique(objects, return_index=True)
        split_by_object_indices = np.split(np.arange(len(objects)), object_indices[1:])

        split_by_object_slices = []
        for object_i_indices in split_by_object_indices:
            assert np.all(np.diff(object_i_indices) == 1)

            object_slice = slice(object_i_indices[0], object_i_indices[-1] + 1)
            split_by_object_slices.append(object_slice)

        return split_by_object_slices

    @staticmethod
    def _split_by_window(windows: List[Tuple[int, int]], nights: np.ndarray) -> List[slice]:
        """
        Create a list of slices for each window in the observations.

        Parameters
        ----------
        windows : list of tuples
            List of tuples containing the start and end night of each window.
        nights : `~numpy.ndarray`
            Array of nights on which observations occur.

        Returns
        -------
        split_by_window_slices : List[slice]
            List of slices for each window in the observations.
        """
        # Split the observations by window
        split_by_window_slices = []
        empty_slice = slice(0, 0)
        for window in windows:
            night_min, night_max = window
            window_indices = np.where((nights >= night_min) & (nights <= night_max))[0]

            if len(window_indices) == 0:
                window_slice = empty_slice
            else:
                assert np.all(np.diff(window_indices) == 1)
                window_slice = slice(window_indices[0], window_indices[-1] + 1)

            split_by_window_slices.append(window_slice)

        return split_by_window_slices

    def _store_as_shared_record_array(self, object_ids, obs_ids, times, ra, dec, nights):
        """
        Store the observations as a record array in shared memory.

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
        dtypes = [
            ("object_id", object_ids.dtype),
            ("obs_id", obs_ids.dtype),
            ("time", np.float64),
            ("ra", np.float64),
            ("dec", np.float64),
            ("night", np.int64),
        ]
        self._dtypes = dtypes
        observations_array = np.empty(
            len(object_ids),
            dtype=dtypes,
        )
        self._num_observations = observations_array.shape[0]
        self._itemsize = observations_array.itemsize

        shared_mem = shared_memory.SharedMemory("DIFI_ARRAY", create=True, size=observations_array.nbytes)
        shared_memory_array = np.ndarray(
            observations_array.shape, dtype=observations_array.dtype, buffer=shared_mem.buf
        )
        shared_memory_array["object_id"] = object_ids
        shared_memory_array["obs_id"] = obs_ids
        shared_memory_array["time"] = times
        shared_memory_array["ra"] = ra
        shared_memory_array["dec"] = dec
        shared_memory_array["night"] = nights
        shared_mem.close()
        return

    @staticmethod
    def _clear_shared_record_array():
        shared_mem = shared_memory.SharedMemory("DIFI_ARRAY")
        shared_mem.unlink()

    def _run_object_worker(
        self,
        object_slice: slice,
        windows: List[Tuple[int, int]],
        discovery_opportunities: bool = False,
        discovery_probability: float = 1.0,
        ignore_after_discovery: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Run the metric on a single object.

        Parameters
        ----------
        object_slice : slice
            The slice in the observations array corresponding to the observations of a single object.
            A slice with start and stop indices will be interpreted as `observations[start:stop]`.
            If start == stop, then no observations will be used.
        windows : list of tuples
            A list of tuples containing the start and end nights of each window.
        discovery_opportunities : bool, optional
            If True, then return the combinations of observations that made each object findable.
            Note, that if True, this can greatly increase the memory consumption.
            If False, just return the unique observation IDs across all discovery chances.
        discovery_probability : float, optional
            The probability applied to a single discovery opportunity that this object will be discovered.
            Each object will have a random seed generated from its object ID, and for each discovery
            opportunity a random number will be drawn between 0 and 1. If the random number is less
            than the discovery probability, then the object will be discovered. If not, then it will
            not be discovered.
        ignore_after_discovery : bool, optional
            If True, then ignore all observations after each object's initial discovery. If False,
            then each object's observations will continue to be tested for discovery in each window.

        Returns
        -------
        findable : List[Dict[str, Any]]]
            A list of dictionaries containing the findable objects and the observations
            that made them findable for each window.
        """
        num_obs = object_slice.stop - object_slice.start
        if num_obs == 0:
            return [
                {
                    "window_id": np.nan,
                    "object_id": np.nan,
                    "findable": np.nan,
                    "discovery_opportunities": np.nan,
                    "obs_ids": np.nan,
                }
            ]

        # Load the observations from shared memory
        existing_shared_mem = shared_memory.SharedMemory(name="DIFI_ARRAY")
        observations = np.ndarray(
            num_obs,
            dtype=self._dtypes,
            buffer=existing_shared_mem.buf,
            offset=object_slice.start * self._itemsize,
        )
        object_id = observations["object_id"][0]

        findable_dicts = []
        discovery_obs_ids = self.determine_object_findable(observations, windows=windows)

        for i, window in enumerate(windows):

            discovery_obs_ids_window = discovery_obs_ids[i]

            if len(discovery_obs_ids_window) > 0:

                obs_ids_unique, discovery_obs_ids_window = apply_discovery_probability(
                    discovery_obs_ids_window, object_id, discovery_probability
                )
                num_opportunities = len(discovery_obs_ids_window)

                if discovery_opportunities:
                    obs_ids = discovery_obs_ids_window
                else:
                    obs_ids = [obs_ids_unique]

            else:
                num_opportunities = 0

            if num_opportunities > 0:
                findable = {
                    "window_id": i,
                    "object_id": object_id,
                    "findable": 1,
                    "discovery_opportunities": num_opportunities,
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
            if ignore_after_discovery and num_opportunities > 0:
                break

        existing_shared_mem.close()
        return findable_dicts

    def run_by_object(
        self,
        observations: pd.DataFrame,
        windows: List[Tuple[int, int]],
        discovery_opportunities: bool = False,
        discovery_probability: float = 1.0,
        ignore_after_discovery: bool = False,
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
            Note, that if True, this can greatly increase the memory consumption.
            If False, just return the unique observation IDs across all discovery chances.
        discovery_probability : float, optional
            The probability applied to a single discovery opportunity that this object will be discovered.
            Each object will have a random seed generated from its object ID, and for each discovery
            opportunity a random number will be drawn between 0 and 1. If the random number is less
            than the discovery probability, then the object will be discovered. If not, then it will
            not be discovered.
        ignore_after_discovery : bool, optional
            If True, then ignore all observations after each object's initial discovery. If False,
            then each object's observations will continue to be tested for discovery in each window.
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
        split_by_object_slices = self._split_by_object(objects)

        # Store the observations in a global variable so that the worker functions can access them
        self._store_as_shared_record_array(objects, obs_ids, times, ra, dec, nights)

        findable_lists: List[List[Dict[str, Any]]] = []
        if num_jobs is None or num_jobs > 1:
            pool = mp.Pool(num_jobs)
            findable_lists = pool.starmap(
                self._run_object_worker,
                zip(
                    split_by_object_slices,
                    repeat(windows),
                    repeat(discovery_opportunities),
                    repeat(discovery_probability),
                    repeat(ignore_after_discovery),
                ),
            )

            pool.close()
            pool.join()

        else:
            for object_indices in split_by_object_slices:
                findable_lists.append(
                    self._run_object_worker(
                        object_indices,
                        windows,
                        discovery_opportunities=discovery_opportunities,
                        discovery_probability=discovery_probability,
                        ignore_after_discovery=ignore_after_discovery,
                    )
                )

        self._clear_shared_record_array()

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
        self,
        window_slice: slice,
        window_id: int,
        discovery_opportunities: bool = False,
        discovery_probability: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Run the metric on a single window of observations.

        Parameters
        ----------
        window_slice: slice
            The slice in the observations array that contains the observations for this window.
            A slice with start and stop indices will be interpreted as `observations[start:stop]`.
            If start == stop, then no observations will be used.
        window_id : int
            The ID of this window.
        discovery_opportunities : bool, optional
            If True, then return the combinations of observations that made each object findable.
            Note, that if True, this can greatly increase the memory consumption.
            If False, just return the unique observation IDs across all discovery chances.
        discovery_probability : float, optional
            The probability applied to a single discovery opportunity that this object will be discovered.
            Each object will have a random seed generated from its object ID, and for each discovery
            opportunity a random number will be drawn between 0 and 1. If the random number is less
            than the discovery probability, then the object will be discovered. If not, then it will
            not be discovered.

        Returns
        -------
        findable : List[Dict[str, Any]]]
            A list of dictionaries containing the findable objects and the observations
            that made them findable for each window.
        """
        num_obs = window_slice.stop - window_slice.start
        if num_obs == 0:
            return [
                {
                    "window_id": np.nan,
                    "object_id": np.nan,
                    "findable": np.nan,
                    "discovery_opportunities": np.nan,
                    "obs_ids": np.nan,
                }
            ]

        # Read observations from shared memory array
        existing_shared_mem = shared_memory.SharedMemory(name="DIFI_ARRAY")
        observations = np.ndarray(
            num_obs,
            dtype=self._dtypes,
            buffer=existing_shared_mem.buf,
            offset=window_slice.start * self._itemsize,
        )

        findable_dicts = []
        for object_id in np.unique(observations["object_id"]):

            discovery_obs_ids = self.determine_object_findable(
                observations[np.where(observations["object_id"] == object_id)[0]],
                # We are running on a single window so don't need to pass in windows
                windows=None,
            )
            # self.determine_object_findable returns a list of lists, with one element (also a list)
            # per window, so lets just grab the first element representing the current window
            discovery_obs_ids_window = discovery_obs_ids[0]

            # If there are discovery opportunities, then apply the discovery probability
            if len(discovery_obs_ids_window) > 0:

                obs_ids_unique, discovery_obs_ids_window = apply_discovery_probability(
                    discovery_obs_ids_window, object_id, discovery_probability
                )
                num_opportunities = len(discovery_obs_ids_window)

                if discovery_opportunities:
                    obs_ids = discovery_obs_ids_window
                else:
                    obs_ids = [obs_ids_unique]

            else:
                num_opportunities = 0

            if num_opportunities > 0:
                findable = {
                    "window_id": window_id,
                    "object_id": object_id,
                    "findable": 1,
                    "discovery_opportunities": num_opportunities,
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

        existing_shared_mem.close()
        return findable_dicts

    def run_by_window(
        self,
        observations: pd.DataFrame,
        windows: List[Tuple[int, int]],
        discovery_opportunities: bool = False,
        discovery_probability: float = 1.0,
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
            Note, that if True, this can greatly increase the memory consumption.
            If False, just return the unique observation IDs across all discovery chances.
        discovery_probability : float, optional
            The probability applied to a single discovery opportunity that this object will be discovered.
            Each object will have a random seed generated from its object ID, and for each discovery
            opportunity a random number will be drawn between 0 and 1. If the random number is less
            than the discovery probability, then the object will be discovered. If not, then it will
            not be discovered.
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

        # Store the observations in a global variable so that the worker functions can access them
        self._store_as_shared_record_array(objects, obs_ids, times, ra, dec, nights)

        # Find indices that split the observations into windows
        split_by_window_slices = self._split_by_window(windows, nights)

        findable_lists: List[List[Dict[str, Any]]] = []
        if num_jobs is None or num_jobs > 1:
            pool = mp.Pool(num_jobs)
            findable_lists = pool.starmap(
                self._run_window_worker,
                zip(
                    split_by_window_slices,
                    range(len(windows)),
                    repeat(discovery_opportunities),
                    repeat(discovery_probability),
                ),
            )
            pool.close()
            pool.join()

        else:
            for i, window_slice in enumerate(split_by_window_slices):
                findable_lists.append(
                    self._run_window_worker(
                        window_slice,
                        i,
                        discovery_opportunities=discovery_opportunities,
                        discovery_probability=discovery_probability,
                    )
                )

        self._clear_shared_record_array()

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
        min_window_nights: Optional[int] = None,
        discovery_opportunities: bool = False,
        discovery_probability: float = 1.0,
        by_object: bool = False,
        ignore_after_discovery: bool = False,
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
            The number of nights of observations to consider when
            determining if a object is findable. If the number of consecutive days
            of observations exceeds the detection_window, then a rolling window
            of size detection_window is used to determine if the object is findable.
            If None, then the detection_window is the entire range observations.
        min_window_nights : int, optional
            The minimum number of nights that must be in a window starting at the first window.
            For example, if detection_window is 10 and min_window_nights is 3, then the first
            window will be nights 1-3, the second window will be nights 1-4, and so on. If None, and
            detection_window is 10, then the first window will be nights 1-10,
            the second window will be nights 2-11, and so on.
        discovery_opportunities : bool, optional
            If True, then return the combinations of observations that made each object findable.
            Note, that if True, this can greatly increase the memory consumption.
            If False, just return the unique observation IDs across all discovery chances.
        discovery_probability : float, optional
            The probability applied to a single discovery opportunity that this object will be discovered.
            Each object will have a random seed generated from its object ID, and for each discovery
            opportunity a random number will be drawn between 0 and 1. If the random number is less
            than the discovery probability, then the object will be discovered. If not, then it will
            not be discovered.
        by_object : bool, optional
            If True, run the metric on the observations split by objects. For windows where there are many
            observations, this may be faster than running the metric on each window individually
            (with all objects' observations).
        ignore_after_discovery : bool, optional
            If True, then ignore observations that occur after the object has been discovered. Only applies
            when by_object is True. If False, then the objects will be tested for discovery chances again.
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
        windows = self._compute_windows(nights, detection_window, min_nights=min_window_nights)

        if not by_object and ignore_after_discovery:
            warnings.warn(
                (
                    "ignore_after_discovery only applies when by_object is True."
                    "Setting ignore_after_discovery to False."
                )
            )
            ignore_after_discovery = False

        if by_object:
            findable = self.run_by_object(
                observations,
                windows,
                discovery_opportunities=discovery_opportunities,
                discovery_probability=discovery_probability,
                ignore_after_discovery=ignore_after_discovery,
                num_jobs=num_jobs,
            )
        else:
            findable = self.run_by_window(
                observations,
                windows,
                discovery_opportunities=discovery_opportunities,
                discovery_probability=discovery_probability,
                num_jobs=num_jobs,
            )

        window_summary = self._create_window_summary(observations, windows, findable)
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
        self,
        observations: np.ndarray,
        windows: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict[int, List[List[str]]]:
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
        windows: List[Tuple[int, int]], optional
            List of windows of time (in MJD) during which to consider observations.

        Returns
        -------
        obs_ids : dict[int, List[List[str]]]
            Dictionary keyed on window IDs, with values of lists containining observation IDs
            for each discovery opportunity (if discovery_opportunities is True).
            If no discovery opportunities are found, then the list will be empty.
            If discovery_opportunities is False, then the list will contain a single
            list of all valid observation IDs.
        """
        assert len(np.unique(observations["object_id"])) == 1
        if windows is None:
            windows = [(np.min(observations["night"]), np.max(observations["night"]))]

        obs_ids_window: Dict[int, List[List[str]]] = {}
        for i, window in enumerate(windows):
            mask = (observations["night"] >= window[0]) & (observations["night"] <= window[1])
            observations_window = observations[mask]

            # Exit early if there are not enough observations
            total_required_obs = self.linkage_min_obs * self.min_linkage_nights
            if len(observations_window) < total_required_obs:
                obs_ids_window[i] = []
                continue

            # Grab times and observation IDs from grouped observations
            times = observations_window["time"]
            obs_ids = observations_window["obs_id"]
            nights = observations_window["night"]
            ra = observations_window["ra"]
            dec = observations_window["dec"]

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
                        obs_ids = []

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
                obs_ids = []
            else:
                obs_indices = select_tracklet_combinations(valid_nights, self.min_linkage_nights)
                obs_ids = [linkage_obs[ind].tolist() for ind in obs_indices]

            obs_ids_window[i] = obs_ids

        return obs_ids_window


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
        self, observations: np.ndarray, windows: Optional[List[Tuple[int, int]]] = None
    ) -> Dict[int, List[List[str]]]:
        """
        Finds all objects with a minimum of self.min_obs observations and the observations
        that makes them findable.

        Parameters
        ----------
        observations : `~numpy.ndarray`
            Numpy record array with at least the following columns:
            `object_id`, `obs_id`, `time`, `night`, `ra`, `dec`.
        windows: List[Tuple[int, int]], optional
            List of windows of time (in MJD) during which to consider observations.

        Returns
        -------
        obs_ids : dict[int, List[List[str]]]
            Dictionary keyed on window IDs, with values of lists containining observation IDs
            for each discovery opportunity (if discovery_opportunities is True).
            If no discovery opportunities are found, then the list will be empty.
            If discovery_opportunities is False, then the list will contain a single
            list of all valid observation IDs.
        """
        # If the len of observations is 0 then the object is not findable
        assert len(np.unique(observations["object_id"])) == 1

        if windows is None:
            windows = [(np.min(observations["night"]), np.max(observations["night"]))]

        obs_ids_window = {}
        for i, window in enumerate(windows):
            mask = (observations["night"] >= window[0]) & (observations["night"] <= window[1])
            observations_window = observations[mask]

            if len(observations_window) >= self.min_obs:
                obs_ids = observations_window["obs_id"]
                obs_ids = list(combinations(obs_ids, self.min_obs))

            else:
                obs_ids = []

            obs_ids_window[i] = obs_ids

        return obs_ids_window
