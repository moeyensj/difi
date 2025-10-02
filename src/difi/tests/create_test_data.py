if __name__ == "__main__":

    import argparse
    import uuid
    from pathlib import Path

    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    from adam_assist import ASSISTPropagator
    from adam_core.observers import Observers
    from adam_core.observers.utils import calculate_observing_night
    from adam_core.orbits import Orbits
    from adam_core.time import Timestamp
    from adam_core.utils.helpers import make_real_orbits

    from difi.difi import LinkageMembers
    from difi.observations import Observations

    def create_observations(
        orbits: Orbits,
        start_night: int = 61000,
        nights: int = 10,
        obs_times_in_night: np.ndarray = np.arange(0, 90, 30),
        observatory_codes: list[str] = ["X05", "W84"],
        seed: int | None = None,
    ) -> Observers:
        """
        Create observers for a given start night, number of nights, and observatory codes.
        The observatory code for a given night is randomly chosen from the list of observatory codes.

        Parameters
        ----------
        start_night : int
            The start night.
        obs_times_in_night : np.ndarray
            The times in the night.
        nights : int
            The number of nights.
        observatory_codes : list[str]
            The list of observatory codes.
        seed : int | None, optional
            The seed for the random number generator.

        Returns
        -------
        Observers
            The observers.
        """
        num_obs = len(obs_times_in_night) * nights
        observation_times = np.empty(num_obs, dtype=float)
        observatory_codes_observations = np.empty(num_obs, dtype=object)
        exposure_ids = np.empty(num_obs, dtype=object)

        rng = np.random.default_rng(seed=seed)
        for i in range(nights):
            observatory_code = rng.choice(observatory_codes)
            observation_times[i * len(obs_times_in_night) : (i + 1) * len(obs_times_in_night)] = (
                start_night + obs_times_in_night / (24 * 60) + i
            )
            observatory_codes_observations[
                i * len(obs_times_in_night) : (i + 1) * len(obs_times_in_night)
            ] = observatory_code
            exposure_ids[i * len(obs_times_in_night) : (i + 1) * len(obs_times_in_night)] = [
                f"exp_{i}_{j:02d}" for j in range(len(obs_times_in_night))
            ]

        times = Timestamp.from_mjd(observation_times, scale="utc")
        observers = Observers.from_codes(pa.array(observatory_codes_observations), times)

        propagator = ASSISTPropagator()
        ephemeris = propagator.generate_ephemeris(orbits, observers, max_processes=1)
        ephemeris = ephemeris.sort_by(["orbit_id", "coordinates.time.days", "coordinates.time.nanos"])
        exposure_ids = np.tile(exposure_ids, len(orbits))

        observations = Observations.from_kwargs(
            id=[uuid.uuid4().hex for _ in range(len(ephemeris))],
            time=ephemeris.coordinates.time,
            ra=ephemeris.coordinates.lon,
            dec=ephemeris.coordinates.lat,
            observatory_code=ephemeris.coordinates.origin.code,
            object_id=ephemeris.orbit_id,
            night=calculate_observing_night(ephemeris.coordinates.origin.code, ephemeris.coordinates.time),
        )
        observations = observations.sort_by(["time.days", "time.nanos"])
        return observations

    def create_linkages(
        observations: Observations,
        num_mixed: int = 5,
        partial_contamination_percents: list[float] | None = None,
        seed: int | None = None,
    ) -> LinkageMembers:
        """
        Create linkages from observations.

        For each unique object_id in the observations, this function creates:
        - one pure linkage that includes all observations for the object
        - several pure-incomplete linkages that include subsets of observations for the object
        - one partial (contaminated) linkage with a specific contamination percentage

        It also creates a configurable number of mixed linkages that contain
        observations from multiple different objects.

        Parameters
        ----------
        observations : Observations
            Table of observations to build linkages from.
        num_mixed : int
            Number of mixed linkages to create. Each mixed linkage will contain a
            random number of observations between 6 and 10 (inclusive) drawn from
            at least 3 distinct objects.
        partial_contamination_percents : list[float] | None
            If provided, must have length equal to number of unique objects. Values are percentages [0-100]
            specifying contamination for each object's partial linkage. If None, values will be spaced
            evenly between 5 and 50% across objects.
        seed : int | None
            Seed for reproducible random selection.

        Returns
        -------
        LinkageMembers
            Table of linkage members with columns linkage_id and obs_id.
        """
        rng = np.random.default_rng(seed=seed)

        linkage_ids: list[str] = []
        obs_ids: list[str] = []

        unique_object_ids = observations.object_id.unique().to_pylist()

        # Create pure, pure-incomplete, and partial (contaminated) linkages per object
        if partial_contamination_percents is not None:
            if len(partial_contamination_percents) != len(unique_object_ids):
                raise ValueError("partial_contamination_percents length must match number of unique objects")
            contamination_percents = partial_contamination_percents
        else:
            # Evenly space contamination percentages from 5% to 50% (inclusive) across objects
            if len(unique_object_ids) == 1:
                contamination_percents = [25.0]
            else:
                contamination_percents = np.linspace(5.0, 50.0, num=len(unique_object_ids)).tolist()

        for object_id in unique_object_ids:
            object_obs = observations.apply_mask(pc.equal(observations.object_id, object_id))
            object_obs_ids = object_obs.id.to_pylist()

            # Pure linkage: all observations for this object
            pure_linkage_id = f"linkage_pure_{object_id}"
            linkage_ids.extend([pure_linkage_id] * len(object_obs_ids))
            obs_ids.extend(object_obs_ids)

            # Pure-incomplete linkages: strict subsets (no completeness)
            if len(object_obs_ids) > 1:
                max_subset = max(1, len(object_obs_ids) - 1)
                desired_k = int(rng.integers(low=6, high=11))  # 6..10 inclusive
                k = max(1, min(desired_k, max_subset))
                subset_obs_ids = rng.choice(object_obs_ids, size=k, replace=False).tolist()
                pure_incomplete_linkage_id = f"linkage_pure_incomplete_{object_id}"
                linkage_ids.extend([pure_incomplete_linkage_id] * len(subset_obs_ids))
                obs_ids.extend(subset_obs_ids)

            # Partial (contaminated) linkage: choose a target contamination percent for this object
            # Determine this object's contamination target
            obj_index = unique_object_ids.index(object_id)
            contamination_percent = contamination_percents[obj_index]
            # Choose total linkage size reasonably large but <= total obs
            total_obs = len(object_obs_ids)
            if total_obs <= 2:
                total_linkage_size = total_obs
            else:
                # Prefer around 12 or fewer, but at least 3 and at most total_obs
                total_linkage_size = int(min(max(3, 12), total_obs))
            # Compute contaminated count and correct count
            contaminated_count = int(round((contamination_percent / 100.0) * total_linkage_size))
            contaminated_count = max(1, min(contaminated_count, total_linkage_size - 1))
            correct_count = total_linkage_size - contaminated_count

            # Sample correct obs from this object
            partial_correct = rng.choice(object_obs_ids, size=correct_count, replace=False).tolist()

            # Sample contaminant obs from other objects
            other_obs_ids = observations.apply_mask(
                pc.not_equal(observations.object_id, object_id)
            ).id.to_pylist()
            if len(other_obs_ids) > 0:
                partial_contaminants = rng.choice(
                    other_obs_ids, size=contaminated_count, replace=False
                ).tolist()
            else:
                partial_contaminants = []
                # If there are no other objects, fall back to pure-incomplete
                # by moving all contaminated slots into correct slots
                partial_correct = rng.choice(object_obs_ids, size=total_linkage_size, replace=False).tolist()

            partial_obs = partial_correct + partial_contaminants
            partial_linkage_id = f"linkage_partial_{object_id}"
            linkage_ids.extend([partial_linkage_id] * len(partial_obs))
            obs_ids.extend(partial_obs)

        # Create mixed linkages: draw a random size [6..10] with >=3 distinct objects
        num_objects = len(unique_object_ids)
        if num_objects > 0 and num_mixed > 0:
            for i in range(num_mixed):
                desired_k = int(rng.integers(low=6, high=11))  # 6..10 inclusive
                num_distinct = min(max(3, 3), min(desired_k, num_objects))
                chosen_objects = rng.choice(unique_object_ids, size=num_distinct, replace=False).tolist()

                # Start with one observation from each chosen object, then distribute the remainder
                counts = np.ones(num_distinct, dtype=int)
                remaining = desired_k - num_distinct
                while remaining > 0:
                    idx = int(rng.integers(low=0, high=num_distinct))
                    counts[idx] += 1
                    remaining -= 1

                mixed_obs: list[str] = []
                for oid, cnt in zip(chosen_objects, counts.tolist()):
                    obj_obs_ids = observations.apply_mask(
                        pc.equal(observations.object_id, oid)
                    ).id.to_pylist()
                    # Sample without replacement per object
                    sampled = rng.choice(obj_obs_ids, size=min(cnt, len(obj_obs_ids)), replace=False).tolist()
                    mixed_obs.extend(sampled)

                mixed_linkage_id = f"linkage_mixed_{i:05d}"
                linkage_ids.extend([mixed_linkage_id] * len(mixed_obs))
                obs_ids.extend(mixed_obs)

        return LinkageMembers.from_kwargs(
            linkage_id=linkage_ids,
            obs_id=obs_ids,
        )

    parser = argparse.ArgumentParser(description="Generate test observations and linkages parquet files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    testdata_dir = Path(__file__).parent / "testdata"
    testdata_dir.mkdir(parents=True, exist_ok=True)

    # Generate data
    orbits = make_real_orbits(5)
    observations = create_observations(orbits, seed=args.seed)
    linkage_members = create_linkages(observations, seed=args.seed)

    # Save observations
    observations_file = testdata_dir / "observations.parquet"
    observations.to_parquet(observations_file)

    # Save linkage members directly
    linkage_members_file = testdata_dir / "linkage_members.parquet"
    linkage_members.to_parquet(linkage_members_file)
