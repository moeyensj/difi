import hashlib
import multiprocessing as mp
import warnings
from abc import ABC, abstractmethod
from itertools import combinations
from typing import Dict, List, Optional, Tuple, TypeVar

import numpy as np
import pandas as pd
import quivr as qv
import ray
from adam_core.ray_cluster import initialize_use_ray
from numba import njit

from .observations import Observations
from .partitions import Partitions

__all__ = ["FindabilityMetric", "NightlyLinkagesMetric", "MinObsMetric"]

Metrics = TypeVar("Metrics", bound="FindabilityMetric")


class FindableObservations(qv.Table):
    partition_id = qv.LargeStringColumn()
    object_id = qv.LargeStringColumn()
    discovery_opportunities = qv.Int64Column()
    obs_ids = qv.ListColumn(qv.ListColumn(qv.LargeStringColumn()), nullable=True)


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


def partition_worker(
    metric: "FindabilityMetric",
    partition: Partitions,
    observations: Observations,
    discovery_opportunities: bool = False,
    discovery_probability: float = 1.0,
) -> FindableObservations:
    """
    Run the metric on a single window of observations.

    Parameters
    ----------
    metric : FindabilityMetric
        Findability metric that defines how an object is considered findable.
    partition : Partitions
        Partition defining the start and end night (both inclusive) of the observations to include.
    observations : Observations
        Observations to run the metric on.
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
    findable : FindableObservations
        Findable observations for this window.
    """
    assert len(partition) == 1
    partition_id = partition.id[0].as_py()
    observations_in_partition = observations.filter_partition(partition)
    if len(observations_in_partition) == 0:
        return FindableObservations.empty()

    findable_observations = FindableObservations.empty()
    for object_id in observations_in_partition.object_id.unique():

        discovery_obs_ids = metric.determine_object_findable(
            observations_in_partition.select("object_id", object_id),
            # We are running on a single partition so don't need to pass in partitions
            partitions=None,
        )
        # self.determine_object_findable returns a list of lists, with one element (also a list)
        # per partition, so lets just grab the first element representing the current partition
        discovery_obs_ids_partition = discovery_obs_ids[partition_id]

        # If there are discovery opportunities, then apply the discovery probability
        if len(discovery_obs_ids_partition) > 0:

            obs_ids_unique, discovery_obs_ids_partition = apply_discovery_probability(
                discovery_obs_ids_partition, object_id, discovery_probability
            )
            num_opportunities = len(discovery_obs_ids_partition)

            if discovery_opportunities:
                obs_ids = discovery_obs_ids_partition
            else:
                obs_ids = [obs_ids_unique]

        else:
            num_opportunities = 0

        if num_opportunities > 0:
            findable = FindableObservations.from_kwargs(
                partition_id=[partition_id],
                object_id=[object_id],
                discovery_opportunities=[num_opportunities],
                obs_ids=[obs_ids],
            )

            findable_observations = qv.concatenate([findable_observations, findable])
            if findable_observations.fragmented():
                findable_observations = qv.defragment(findable_observations)

    findable_observations = findable_observations.sort_by(
        [
            ("partition_id", "ascending"),
            ("object_id", "ascending"),
        ]
    )
    return findable_observations


partition_worker_remote = ray.remote(partition_worker)
partition_worker_remote.options(num_cpus=1)


def object_worker(
    metric: "FindabilityMetric",
    object_id: str,
    partitions: Partitions,
    observations: Observations,
    discovery_opportunities: bool = False,
    discovery_probability: float = 1.0,
    ignore_after_discovery: bool = False,
) -> FindableObservations:
    """
    Run the metric on a single object.

    Parameters
    ----------
    metric : FindabilityMetric
        Findability metric that defines how an object is considered findable.
    object_id : str
        Object ID to run the metric on.
    partitions : Partitions
        Partitions defining the start and end night (both inclusive) of the observations to include.
        These partitions are used to filter the observations to only include those within the given partition.
    observations : Observations
        Observations to run the metric on.
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
    findable : FindableObservations
        Findable observations for this object.
    """
    object_observations = observations.select("object_id", object_id)

    num_obs = len(object_observations)
    if num_obs == 0:
        return FindableObservations.empty()

    discovery_obs_ids = metric.determine_object_findable(object_observations, partitions=partitions)

    findable_observations = FindableObservations.empty()
    for partition in partitions:
        partition_id = partition.id[0].as_py()

        discovery_obs_ids_partition = discovery_obs_ids[partition_id]

        if len(discovery_obs_ids_partition) > 0:

            obs_ids_unique, discovery_obs_ids_partition = apply_discovery_probability(
                discovery_obs_ids_partition, object_id, discovery_probability
            )
            num_opportunities = len(discovery_obs_ids_partition)

            if discovery_opportunities:
                obs_ids = discovery_obs_ids_partition
            else:
                obs_ids = [obs_ids_unique]

        else:
            num_opportunities = 0

        if num_opportunities > 0:
            findable = FindableObservations.from_kwargs(
                partition_id=partition.id,
                object_id=[object_id],
                discovery_opportunities=[num_opportunities],
                obs_ids=[obs_ids],
            )

            findable_observations = qv.concatenate([findable_observations, findable])
            if findable_observations.fragmented():
                findable_observations = qv.defragment(findable_observations)

        if ignore_after_discovery and num_opportunities > 0:
            break

    findable_observations = findable_observations.sort_by(
        [
            ("partition_id", "ascending"),
            ("object_id", "ascending"),
        ]
    )
    return findable_observations


object_worker_remote = ray.remote(object_worker)
object_worker_remote.options(num_cpus=1)


class FindabilityMetric(ABC):
    @abstractmethod
    def determine_object_findable(
        self, observations: Observations, partitions: Optional[Partitions] = None
    ) -> Dict[str, List[List[str]]]:
        pass

    def run_by_object(
        self,
        observations: Observations,
        partitions: Partitions,
        discovery_opportunities: bool = False,
        discovery_probability: float = 1.0,
        ignore_after_discovery: bool = False,
        max_processes: Optional[int] = 1,
    ) -> FindableObservations:
        """
        Run the metric on the observations split by objects.

        Parameters
        ----------
        observations : Observations
            Observations to run the metric on.
        partitions : Partitions
            Partitions defining the start and end night (both inclusive) of the observations to include.
            These partitions are used to filter the observations to only include those within the given partition.
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
            If True, then ignore observations that occur after the object has been discovered. Only applies
            when by_object is True. If False, then the objects will be tested for discovery chances again.
        max_processes : int, optional
            The maximum number of processes to run in parallel. If None, then use the number of CPUs on the machine.

        Returns
        -------
        findable_observations : FindableObservations
            Findable objects and their observations.
        """
        if max_processes is None:
            max_processes = mp.cpu_count()

        use_ray = initialize_use_ray(num_cpus=max_processes)
        if use_ray:

            observations_ref = ray.put(observations)

            object_ids = observations.object_id.unique()
            findable_observations = FindableObservations.empty()
            futures = []
            for object_id in object_ids:
                futures.append(
                    object_worker_remote.remote(
                        self,
                        object_id,
                        partitions,
                        observations_ref,
                        discovery_opportunities=discovery_opportunities,
                        discovery_probability=discovery_probability,
                        ignore_after_discovery=ignore_after_discovery,
                    )
                )

                if len(futures) >= max_processes * 1.5:
                    finished, futures = ray.wait(futures, num_returns=1)
                    findable_observations = qv.concatenate([findable_observations, ray.get(finished[0])])
                    if findable_observations.fragmented():
                        findable_observations = qv.defragment(findable_observations)

            while futures:
                finished, futures = ray.wait(futures, num_returns=1)
                findable_observations = qv.concatenate([findable_observations, ray.get(finished[0])])
                if findable_observations.fragmented():
                    findable_observations = qv.defragment(findable_observations)

            ray.internal.free(observations_ref)

        else:

            object_ids = observations.object_id.unique()
            findable_observations = FindableObservations.empty()

            for object_id in object_ids:
                findable_observations_i = object_worker(
                    self,
                    object_id,
                    partitions,
                    observations,
                    discovery_opportunities=discovery_opportunities,
                    discovery_probability=discovery_probability,
                    ignore_after_discovery=ignore_after_discovery,
                )

                findable_observations = qv.concatenate([findable_observations, findable_observations_i])
                if findable_observations.fragmented():
                    findable_observations = qv.defragment(findable_observations)

        return findable_observations

    def run_by_partition(
        self,
        observations: Observations,
        partitions: Partitions,
        discovery_opportunities: bool = False,
        discovery_probability: float = 1.0,
        max_processes: Optional[int] = 1,
    ) -> FindableObservations:
        """
        Run the metric on the observations split by partition.

        Parameters
        ----------
        observations : Observations
            Observations to run the metric on.
        partitions : Partitions
            Partitions defining the start and end night (both inclusive) of the observations to include.
            These partitions are used to filter the observations to only include those within the given partition.
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
        max_processes : int, optional
            The maximum number of processes to run in parallel. If None, then use the number of CPUs on the machine.

        Returns
        -------
        findable_observations : FindableObservations
            Findable objects and their observations.
        """
        if max_processes is None:
            max_processes = mp.cpu_count()

        use_ray = initialize_use_ray(num_cpus=max_processes)
        if use_ray:

            observations_ref = ray.put(observations)

            findable_observations = FindableObservations.empty()
            futures = []
            for partition in partitions:
                futures.append(
                    partition_worker_remote.remote(
                        self,
                        partition,
                        observations_ref,
                        discovery_opportunities=discovery_opportunities,
                        discovery_probability=discovery_probability,
                    )
                )

                if len(futures) >= max_processes * 1.5:
                    finished, futures = ray.wait(futures, num_returns=1)
                    findable_observations = qv.concatenate([findable_observations, ray.get(finished[0])])
                    if findable_observations.fragmented():
                        findable_observations = qv.defragment(findable_observations)

            while futures:

                finished, futures = ray.wait(futures, num_returns=1)
                findable_observations = qv.concatenate([findable_observations, ray.get(finished[0])])
                if findable_observations.fragmented():
                    findable_observations = qv.defragment(findable_observations)

            ray.internal.free(observations_ref)

        else:

            findable_observations = FindableObservations.empty()
            for partition in partitions:
                findable_observations_i = partition_worker(
                    self,
                    partition,
                    observations,
                    discovery_opportunities=discovery_opportunities,
                    discovery_probability=discovery_probability,
                )
                findable_observations = qv.concatenate([findable_observations, findable_observations_i])
                if findable_observations.fragmented():
                    findable_observations = qv.defragment(findable_observations)

        return findable_observations

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
        return_summary: bool = True,
        clear_on_failure: bool = True,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
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
        return_summary : bool, optional
            If True, then return a summary of the number of observations, number of findable
            objects and the start and end night of each window.
        clear_on_failure : bool, optional
            If a failure occurs and this is False, then the shared memory array will not be cleared.
            If True, then the shared memory array will be cleared.

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
                clear_on_failure=clear_on_failure,
            )
        else:
            findable = self.run_by_window(
                observations,
                windows,
                discovery_opportunities=discovery_opportunities,
                discovery_probability=discovery_probability,
                num_jobs=num_jobs,
                clear_on_failure=clear_on_failure,
            )

        if return_summary:
            window_summary = self._create_window_summary(observations, windows, findable)
        else:
            window_summary = None
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
        observations: Observations,
        partitions: Optional[Partitions] = None,
    ) -> Dict[str, List[List[str]]]:
        """
        Given observations belonging to one object, finds all observations that are within
        max_obs_separation of each other.

        If linkage_min_obs is 1 then the object is findable if there are at least
        min_linkage_nights of observations.

        Parameters
        ----------
        observations : Observations
            Observations to run the metric on.
        partitions : Partitions, optional
            Partitions defining the start and end night (both inclusive) of the observations to include.
            These partitions are used to filter the observations to only include those within the given partition.
            If None, then the observations are partitioned into a single partition spanning the full range of nights.

        Returns
        -------
        obs_ids : dict[str, List[List[str]]]
            Dictionary keyed on partition IDs, with values of lists containining observation IDs
            for each discovery opportunity (if discovery_opportunities is True).
            If no discovery opportunities are found, then the list will be empty.
            If discovery_opportunities is False, then the list will contain a single
            list of all valid observation IDs.
        """
        assert len(observations.object_id.unique()) == 1
        if partitions is None:
            partitions = Partitions.create_single(observations.night)

        obs_ids_partition: Dict[int, List[List[str]]] = {}
        for partition in partitions:

            partition_id = partition.id[0].as_py()
            observations_in_window = observations.filter_partition(partition)

            # Exit early if there are not enough observations
            total_required_obs = self.linkage_min_obs * self.min_linkage_nights
            if len(observations_in_window) < total_required_obs:
                obs_ids_partition[partition_id] = []
                continue

            # Grab times and observation IDs from grouped observations
            times = observations_in_window.time.mjd().to_numpy(zero_copy_only=False)
            obs_ids = observations_in_window.id.to_numpy(zero_copy_only=False)
            nights = observations_in_window.night.to_numpy(zero_copy_only=False)
            ra = observations_in_window.ra.to_numpy(zero_copy_only=False)
            dec = observations_in_window.dec.to_numpy(zero_copy_only=False)

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

            obs_ids_partition[partition_id] = obs_ids

        return obs_ids_partition


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
        self, observations: Observations, partitions: Optional[Partitions] = None
    ) -> FindableObservations:
        """
        Finds all objects with a minimum of self.min_obs observations and the observations
        that makes them findable.

        Parameters
        ----------
        observations : Observations
            Observations to run the metric on.
        partitions : Partitions, optional
            Partitions defining the start and end night (both inclusive) of the observations to include.
            These partitions are used to filter the observations to only include those within the given partition.
            If None, then the observations are partitioned into a single partition spanning the full range of nights.

        Returns
        -------
        obs_ids : dict[str, List[List[str]]]
            Dictionary keyed on partition IDs, with values of lists containining observation IDs
            for each discovery opportunity (if discovery_opportunities is True).
            If no discovery opportunities are found, then the list will be empty.
            If discovery_opportunities is False, then the list will contain a single
            list of all valid observation IDs.
        """
        assert len(observations.object_id.unique()) == 1
        if partitions is None:
            partitions = Partitions.create_single(observations.night)

        obs_ids_partition = {}
        for partition in partitions:

            partition_id = partition.id[0].as_py()

            observations_in_partition = observations.filter_partition(partition)

            if len(observations_in_partition) >= self.min_obs:
                obs_ids = observations_in_partition.id.to_numpy(zero_copy_only=False).tolist()
                obs_ids = list(combinations(obs_ids, self.min_obs))

            else:
                obs_ids = []

            obs_ids_partition[partition_id] = obs_ids

        return obs_ids_partition
