import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .utils import _checkColumnTypes, _checkColumnTypesEqual, _classHandler

__all__ = ["analyzeLinkages"]


def analyzeLinkages(
    observations: pd.DataFrame,
    linkage_members: pd.DataFrame,
    all_objects: Optional[pd.DataFrame] = None,
    min_obs: int = 5,
    contamination_percentage: float = 20.0,
    classes: Optional[dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Did I Find It?

    Given a data frame of observations and a data frame defining possible linkages made from those
    observations this function identifies each linkage as one of three possible types:
    - pure: a linkage where all constituent observations belong to a single truth
    - partial: a linkage that contains observations belonging to multiple objects but
        equal to or more than min_obs observations of one truth and no more than the contamination
        threshold of observations of other objects. For example, a linkage with ten observations,
        eight of which belong to a single unique truth and two of which belong to other objects
        has contamination percentage 20%. If the threshold is set to 20% or greater, and min_obs
        is less than or equal to eight then the truth with the eight observations
        is considered found and the linkage is considered a partial linkage.
    - mixed: all linkages that are neither pure nor partial.


    Parameters
    ----------
    observations : `~pandas.DataFrame`
        Pandas DataFrame with at least two columns: observation IDs and the truth values
        (the object to which the observation belongs to).
    linkage_members : `~pandas.DataFrame`
        Pandas DataFrame with at least two columns: linkage IDs and the observation
    all_linkages : {`~pandas.DataFrame`, None}, optional
        Pandas DataFrame with one row per linkage with at least one column: linkage IDs.
        If None, all_linkages will be created.
        [Default = None]
    all_objects : {`~pandas.DataFrame`, None}, optional
        Pandas DataFrame with one row per unique truth with at least one column: object_id.
        If None, all_objects will be created.
        [Default = None]
    min_obs : int, optional
        The minimum number of observations belonging to one object in a pure linkage for
        that object to be considered found. In a partial linkage, for an object to be considered
        found it must have equal to or more than this number of observations for it to be considered
        found.
    contamination_percentage : float, optional
        Number of detections expressed as a percentage [0-100] belonging to other objects in a linkage
        for that linkage to considered partial. For example, if contamination_percentage is
        20% then a linkage with 10 members, where 8 belong to one object and 2 belong to other objects,
        will be considered a partial linkage.
        [Default = 20]
    classes : {dict, str, None}
        Analyze observations for objects grouped in different classes.
        str : Name of the column in the observations dataframe which identifies
            the class of each truth.
        dict : A dictionary with class names as keys and a list of unique
            objects belonging to each class as values.
        None : If there are no classes of objects.

    Returns
    -------
    all_linkages : `~pandas.DataFrame`
        A per-linkage summary.

        Columns:
            "linkage_id" : str
                Unique linkage ID.
            "num_obs" : int
                Number of constituent observations contained in the linkage.
            "num_members" : int
                Number of unique object IDs contained in the linkage.
            "pure" : int
                1 if this linkage is pure, 0 otherwise.
            "pure_complete" : int
                1 if this linkage is complete and pure, 0 otherwise.
            "partial" : int
                1 if this linkage is partial, 0 otherwise.
            "contamination_percentage" : float
                Percent of observations that do not belong to the linked truth. For pure
                linkages this number is always 0, for partial linkages it will never exceed
                the contaminationPercentage, for mixed linkages it is always NaN.
            "found_pure" : int
                1 if the pure linkage has equal to or more than min_obs observations (the linked truth
                is then considered found), 0 otherwise. A linkage can be pure but not condisered found
                and pure if it does not have enough observations.
            "found_partial" : int
                1 if the partial linkage has equal to or more than min_obs observations of the linked
                truth (the linked truth is then considered found), 0 otherwise. A linkage can be
                partial but not considered found if it does not have enough observations of the linked
                truth.
            "found" : int
                1 if either found_partial or found_pure is 1, 0 otherwise.
            "linked_object_id" : str
                The truth linked in the linkage if the linkage is pure or partial, NaN otherwise.

    all_objects: `~pandas.DataFrame`
        A per-truth summary.

        Columns:
            "object_id" : str
                Truth
            "num_obs" : int
                Number of observations in the observations dataframe
                for each truth
            "findable" : int
                1 if the object is findable, 0 if the object is not findable.
                (NaN if no findable column is found in the all_objects dataframe)
            "found_pure" : int
                Number of pure linkages that contain at least min_obs observations.
            "found_partial" : int
                Number of partial linkages that contain at least min_obs observations
                and contain no more than the contamination_percentage of observations
                of other object IDs.
            "found" : int
                Sum of found_pure and found_partial.
            "pure" : int
                Number of pure linkages that observations belonging to this truth
                are found in.
            "pure_complete" : int
                Number of pure linkage that contain all of this truth's observations (this number is
                a subset of the the number of pure linkages).
            "partial" : int
                Number of partial linkages.
            "partial_contaminant" : int
                Number of partial linkages that are contaminated by observations belonging to
                this truth.
            "mixed" : int
                Number of mixed linkages that observations belonging to this truth are
                found in.
            "obs_in_pure" : int
                Total number of observations (not-unique) that are contained in pure linkages.
            "obs_in_partial" : int
                Total number of observations (not-unique) that are contained in partial linkages.
            "obs_in_partial_contaminant" : int
                Total number of observations (not-unique) that contaminate other truth's partial
                linkages.
            "obs_in_mixed" : int
                Total number of observations (not-unique) that are contained in mixed linkages.

    summary : `~pandas.DataFrame`
        A per-class summary.

        Columns:
            "class" : str
                Name of class (if none are defined, will only contain) "All".
            "num_members" : int
                Number of unique object IDs that belong to the class.
            "num_obs" : int
                Number of observations of object IDs belonging to the class in
                the observations dataframe.
            "completeness" : float
                Percent of object IDs deemed findable that are found in pure or
                partial linkages that contain more than min_obs observations.
            "findable" : int
                Number of object IDs deemed findable (all_objects must be passed to this
                function with a findable column)
            "found" : int
                Number of object IDs found in pure and partial linkages with equal to or
                more than min_obs observations.
            "findable_found" : int
                Number of object IDs deemed findable that were found.
            "findable_missed" : int
                Number of object IDs deemed findable that were not found.
            "not_findable_found" : int
                Number of object IDs deemed not findable that were found (serendipitous discoveries).
            "not_findable_missed" : int
                Number of object IDs deemed not findable that were not found.
            "linkages" : int
                Number of unique linkages that contain observations of this class of object IDs.
            "pure_linkages" : int
                Number of pure linkages that contain observations of this class of object IDs.
            "pure_complete_linkages" : int
                Number of complete pure linkages that contain observations of this class of object IDs.
            "partial_linkages" : int
                Number of partial linkages that contain observations of this class of object IDs.
            "partial_contaminant_linkages" : int
                Number of partial linkages that are contaminated by observations of this class of object IDs.
            "mixed_linkages" :
                Number of mixed linkages that contain observations of this class of object IDs.
            "unique_in_pure_linkages" : int
                Number of unique object IDs in pure linkages.
            "unique_in_pure_complete_linkages" : int
                Number of unique object IDs in pure complete linkages (subset of unique_in_pure_linkages).
            "unique_in_pure_linkages_only" : int
                Number of unique object IDs in pure linkages only (not in partial linkages but can be in mixed
                linkages).
            "unique_in_partial_linkages_only" : int
                Number of unique object IDs in partial linkages only (not in pure linkages but can be in mixed
                linkages).
            "unique_in_pure_and_partial_linkages" : int
                Number of unique object IDs that appear in both pure and partial linkages.
            "unique_in_partial_linkages" : int
                Number of unique object IDs in partial linkages.
            "unique_in_partial_contaminant_linkages" : int
                Number of unique object IDs that contaminate partial linkages.
            "unique_in_mixed_linkages" : int
                Number of unique object IDs in mixed linkages.
            "obs_in_pure_linkages" : int
                Number of observations of object IDs of this class in pure linkages.
            "obs_in_pure_complete_linkages" : int
                Number of observations of object IDs of this class in complete pure linkages.
            "obs_in_partial_linkages" : int
                Number of observations of object IDs of this class in partial linkages.
            "obs_in_partial_contaminant_linkages" : int
                Number of observations of object IDs of this class that contaminate partial linkages.
            "obs_in_mixed_linkages" : int
                Number of observations of object IDs of this class in mixed linkages.

    Raises
    ------
    TypeError : If the truth column in observations does not have type "Object",
        or if the obs_id columns in observations and linkage_members do not have the same type,
        or if the linkage_id columns in all_linkages (if passed) and linkage_members do not have the same
        type, or if the truth columns in all_objects (if passed) and observations do not have the same type.
    """
    # Raise error if there are no observations
    if len(observations) == 0:
        raise ValueError("There are no observations in the observations DataFrame!")

    findable_present = True
    # If all_objects DataFrame does not exist, create it
    if all_objects is None:
        objects = observations["object_id"].value_counts()

        all_objects = pd.DataFrame(
            {
                "object_id": objects.index.values,
                # "class" : ["None" for i in range(len(objects))],
                "num_obs": np.zeros(len(objects), dtype=int),
                "findable": [np.NaN for i in range(len(objects))],
            }
        )
        all_objects["object_id"] = all_objects["object_id"].astype(str)

        num_obs_per_truth = observations["object_id"].value_counts()
        all_objects.loc[
            all_objects["object_id"].isin(num_obs_per_truth.index.values), "num_obs"
        ] = num_obs_per_truth.values

        all_objects.sort_values(
            by=["num_obs", "object_id"], ascending=[False, True], inplace=True, ignore_index=True
        )

        findable_present = False

    # If it does exist, add columns
    else:
        all_objects = all_objects.copy()
        if "findable" not in all_objects.columns:
            warn = (
                "No findable column found in all_objects. Completeness\n" "statistics can not be calculated."
            )
            warnings.warn(warn, UserWarning)
            all_objects.loc[:, "findable"] = np.NaN
            findable_present = False
        _checkColumnTypesEqual(all_objects, observations, ["object_id"])

    # Check column types
    _checkColumnTypes(observations, ["object_id"])
    _checkColumnTypes(observations, ["obs_id"])
    _checkColumnTypes(linkage_members, ["obs_id"])
    _checkColumnTypesEqual(observations, linkage_members, ["obs_id"])

    # Create a summary dictionary
    summary_cols = [
        "class",
        "num_members",
        "num_obs",
        "completeness",
        "findable",
        "found",
        "findable_found",
        "findable_missed",
        "not_findable_found",
        "not_findable_missed",
        "linkages",
        "found_pure_linkages",
        "found_partial_linkages",
        "pure_linkages",
        "pure_complete_linkages",
        "partial_linkages",
        "partial_contaminant_linkages",
        "mixed_linkages",
        "unique_in_pure_linkages",
        "unique_in_pure_complete_linkages",
        "unique_in_pure_linkages_only",
        "unique_in_partial_linkages_only",
        "unique_in_pure_and_partial_linkages",
        "unique_in_partial_linkages",
        "unique_in_partial_contaminant_linkages",
        "unique_in_mixed_linkages",
        "obs_in_pure_linkages",
        "obs_in_pure_complete_linkages",
        "obs_in_partial_linkages",
        "obs_in_partial_contaminant_linkages",
        "obs_in_mixed_linkages",
    ]
    summary: dict = {c: [] for c in summary_cols}

    if len(linkage_members) > 0:
        # Grab only observation IDs and truth from observations
        all_linkages = observations[["obs_id", "object_id"]].copy()

        # Merge truth from observations with linkage_members on observation IDs
        all_linkages = all_linkages.merge(linkage_members[["linkage_id", "obs_id"]], on="obs_id")

        # Group the data frame of objects, linkage_ids and
        # observation IDs by truth and linkage ID
        # then count the number of occurences
        all_linkages = all_linkages.groupby(by=["object_id", "linkage_id"]).count().reset_index()
        all_linkages.rename(columns={"obs_id": "num_obs"}, inplace=True)

        # Calculate the total number of observations in each linkage
        num_obs_in_linkage = (
            all_linkages.groupby(by=["linkage_id"])["num_obs"].sum().to_frame(name="num_obs_in_linkage")
        )
        num_obs_in_linkage.reset_index(drop=False, inplace=True)

        # Merge with num_obs_in_linkage to get the total
        # number of observations in each linkage
        all_linkages = all_linkages.merge(
            num_obs_in_linkage,
            left_on="linkage_id",
            right_on="linkage_id",
            suffixes=("", "_"),
        )

        # Calculate the number of unique objects in each linkage
        num_object_in_linkage = (
            all_linkages.groupby(by=["linkage_id"])["object_id"].nunique().to_frame(name="num_members")
        )
        num_obs_in_linkage.reset_index(drop=False, inplace=True)

        # Merge with num_object_in_linkage to get the total
        # number of objects in each linkage
        all_linkages = all_linkages.merge(
            num_object_in_linkage,
            left_on="linkage_id",
            right_on="linkage_id",
            suffixes=("", "_"),
        )

        # Merge with all_objects to get the total number of
        # observations in the observations data frame
        all_linkages = all_linkages.merge(
            all_objects[["object_id", "num_obs"]].rename(
                columns={"num_obs": "num_obs_in_observations"},
            ),
            left_on="object_id",
            right_on="object_id",
            suffixes=("", "_"),
        )

        # For each truth calculate the percent of observations
        # in a linkage that belong to that truth
        all_linkages["percentage_in_linkage"] = (
            100.0 * all_linkages["num_obs"] / all_linkages["num_obs_in_linkage"]
        )
        all_linkages["contamination_percentage_in_linkages"] = 100 - all_linkages["percentage_in_linkage"]

        # Sort by linkage_id and the percentage then reset the index
        all_linkages.sort_values(
            by=["linkage_id", "percentage_in_linkage"],
            ascending=[True, False],
            inplace=True,
            ignore_index=True,
        )

        # Initialize the linkage purity columns
        all_linkages.loc[:, "found_pure"] = 0
        all_linkages.loc[:, "found_partial"] = 0
        all_linkages.loc[:, "found"] = 0
        all_linkages.loc[:, "pure"] = 0
        all_linkages.loc[:, "pure_complete"] = 0
        all_linkages.loc[:, "partial"] = 0
        all_linkages.loc[:, "partial_contaminant"] = 0
        all_linkages.loc[:, "mixed"] = 0

        # Pure linkages: any linkage where each observation belongs to the same truth
        all_linkages.loc[(all_linkages["num_obs"] == all_linkages["num_obs_in_linkage"]), "pure"] = 1

        # Complete pure linkages: any linkage where all observations of a truth are linked
        all_linkages.loc[
            (all_linkages["num_obs"] == all_linkages["num_obs_in_observations"])
            & (all_linkages["pure"] == 1),
            "pure_complete",
        ] = 1

        # Partial linkages: any linkage where up to a contamination percentage of observations belong
        # to other objects
        all_linkages.loc[
            (all_linkages["pure"] == 0)
            & (all_linkages["contamination_percentage_in_linkages"] <= contamination_percentage),
            "partial",
        ] = 1
        partial_linkages = all_linkages[all_linkages["partial"] == 1]["linkage_id"].unique()
        all_linkages.loc[
            (
                all_linkages["linkage_id"].isin(partial_linkages)
                & (all_linkages["contamination_percentage_in_linkages"] > contamination_percentage)
            ),
            "partial_contaminant",
        ] = 1

        # If the contamination percentage is high it may set linkages with no clear majority of detections
        # belonging to one object as partials.. these are actually mixed so make sure
        # they are correctly indentified.
        contamination_counts = (
            all_linkages[all_linkages["linkage_id"].isin(partial_linkages)]
            .groupby("linkage_id")["contamination_percentage_in_linkages"]
            .nunique()
        )
        no_majority_partials = contamination_counts[contamination_counts.values == 1].index.values
        all_linkages.loc[all_linkages["linkage_id"].isin(no_majority_partials), "partial"] = 0
        all_linkages.loc[
            all_linkages["linkage_id"].isin(no_majority_partials),
            "partial_contaminant",
        ] = 0
        all_linkages.loc[all_linkages["linkage_id"].isin(no_majority_partials), "mixed"] = 1

        # Mixed linkages: any linkage that is not pure or partial
        all_linkages.loc[
            (all_linkages["pure"] == 0)
            & (all_linkages["partial"] == 0)
            & (all_linkages["partial_contaminant"] == 0),
            "mixed",
        ] = 1

        # Update found columns
        all_linkages.loc[
            (all_linkages["num_obs"] >= min_obs) & (all_linkages["pure"] == 1),
            "found_pure",
        ] = 1
        all_linkages.loc[
            (all_linkages["num_obs"] >= min_obs) & (all_linkages["partial"] == 1),
            "found_partial",
        ] = 1
        all_linkages.loc[
            (all_linkages["found_pure"] == 1) | (all_linkages["found_partial"] == 1),
            "found",
        ] = 1

        # Calculate number of observations in pure linkages for each truth
        pure_obs = (
            all_linkages[all_linkages["pure"] == 1]
            .groupby(by="object_id")["num_obs"]
            .sum()
            .to_frame(name="obs_in_pure")
        )
        pure_obs.reset_index(inplace=True)

        # Calculate number of observations in pure complete linkages for each truth
        pure_complete_obs = (
            all_linkages[all_linkages["pure_complete"] == 1]
            .groupby(by="object_id")["num_obs"]
            .sum()
            .to_frame(name="obs_in_pure_complete")
        )
        pure_complete_obs.reset_index(inplace=True)

        # Calculate number of observations in partial linkages for each truth
        partial_obs = (
            all_linkages[all_linkages["partial"] == 1]
            .groupby(by="object_id")["num_obs"]
            .sum()
            .to_frame(name="obs_in_partial")
        )
        partial_obs.reset_index(inplace=True)

        # Calculate number of observations in partial linkages for each truth
        partial_contaminant_obs = (
            all_linkages[(all_linkages["partial_contaminant"] == 1)]
            .groupby(by="object_id")["num_obs"]
            .sum()
            .to_frame(name="obs_in_partial_contaminant")
        )
        partial_contaminant_obs.reset_index(inplace=True)

        # Calculate number of observations in mixed linkages for each truth
        mixed_obs = (
            all_linkages[all_linkages["mixed"] == 1]
            .groupby(by="object_id")["num_obs"]
            .sum()
            .to_frame(name="obs_in_mixed")
        )
        mixed_obs.reset_index(inplace=True)

        linkage_types = all_linkages.groupby(by=["object_id"])[
            [
                "pure",
                "pure_complete",
                "partial",
                "partial_contaminant",
                "mixed",
                "found_pure",
                "found_partial",
                "found",
            ]
        ].sum()
        linkage_types.reset_index(inplace=True)

        for df in [
            pure_obs,
            pure_complete_obs,
            partial_obs,
            partial_contaminant_obs,
            mixed_obs,
        ]:
            linkage_types = linkage_types.merge(df, on="object_id", how="outer")

    else:
        # Create empty linkage_types dataframe when linkage_members is empty
        dtypes = np.dtype(
            [
                ("object_id", str),
                ("pure", int),
                ("pure_complete", int),
                ("partial", int),
                ("partial_contaminant", int),
                ("mixed", int),
                ("found_pure", int),
                ("found_partial", int),
                ("found", int),
                ("obs_in_pure", int),
                ("obs_in_pure_complete", int),
                ("obs_in_partial", int),
                ("obs_in_partial_contaminant", int),
                ("obs_in_mixed", int),
            ]
        )
        linkage_types = pd.DataFrame(np.empty(0, dtype=dtypes))

        # Create empty all_linkages dataframe when linkage_members is empty
        dtypes = np.dtype(
            [
                ("linkage_id", str),
                ("num_obs", int),
                ("num_obs_in_linkage", int),
                ("num_members", int),
                ("num_obs_in_observations", int),
                ("percentage_in_linkage", float),
                ("pure", int),
                ("pure_complete", int),
                ("partial", int),
                ("partial_contaminant", int),
                ("mixed", int),
                ("contamination_percentage_in_linkages", float),
                ("found_pure", int),
                ("found_partial", int),
                ("found", int),
                ("object_id", str),
            ]
        )
        all_linkages = pd.DataFrame(np.empty(0, dtype=dtypes))

    all_objects = all_objects.merge(linkage_types, on="object_id", how="outer")
    all_objects_int_cols = [
        "pure",
        "pure_complete",
        "partial",
        "partial_contaminant",
        "mixed",
        "found_pure",
        "found_partial",
        "found",
        "obs_in_pure",
        "obs_in_pure_complete",
        "obs_in_partial",
        "obs_in_partial_contaminant",
        "obs_in_mixed",
    ]
    all_objects[all_objects_int_cols] = all_objects[all_objects_int_cols].fillna(0)
    all_objects[all_objects_int_cols] = all_objects[all_objects_int_cols].astype(int)

    class_list, truths_list = _classHandler(classes, observations)

    # Loop through the classes and summarize the results
    for c, v in zip(class_list, truths_list):
        # Create masks for the class of objects
        all_truths_class = all_objects[all_objects["object_id"].isin(v)]
        all_linkages_class = all_linkages[all_linkages["object_id"].isin(v)]
        observations_class = observations[observations["object_id"].isin(v)]

        # Add class and the number of members to the summary
        summary["class"].append(c)
        summary["num_members"].append(len(v))
        summary["num_obs"].append(len(observations_class))

        # Number of objects found
        found = len((all_truths_class[all_truths_class["found"] >= 1]))
        summary["found"].append(found)

        if findable_present:
            # Number of objects findable
            findable = len(all_truths_class[all_truths_class["findable"] == 1])
            summary["findable"].append(findable)

            # Number of findable objects found
            findable_found = len(
                all_truths_class[(all_truths_class["findable"] == 1) & (all_truths_class["found"] >= 1)]
            )
            summary["findable_found"].append(findable_found)

            # Calculate completeness
            if findable == 0:
                completeness = np.NaN
            else:
                completeness = 100.0 * findable_found / findable
            summary["completeness"].append(completeness)

            # Number of findable objects missed
            findable_missed = len(
                all_truths_class[(all_truths_class["findable"] == 1) & (all_truths_class["found"] == 0)]
            )
            summary["findable_missed"].append(findable_missed)

            # Number of not findable objects found
            not_findable_found = len(
                all_truths_class[(all_truths_class["findable"] == 0) & (all_truths_class["found"] >= 1)]
            )
            summary["not_findable_found"].append(not_findable_found)

            # Number of not findable objects missed
            not_findable_missed = len(
                all_truths_class[(all_truths_class["findable"] == 0) & (all_truths_class["found"] == 0)]
            )
            summary["not_findable_missed"].append(not_findable_missed)

        else:
            summary["completeness"].append(np.NaN)
            summary["findable"].append(np.NaN)
            summary["findable_found"].append(np.NaN)
            summary["findable_missed"].append(np.NaN)
            summary["not_findable_found"].append(np.NaN)
            summary["not_findable_missed"].append(np.NaN)

        # Calculate number of linkage types that contain observations of this class
        for linkage_type in [
            "found_pure",
            "found_partial",
            "pure",
            "pure_complete",
            "partial",
            "partial_contaminant",
            "mixed",
        ]:
            summary["{}_linkages".format(linkage_type)].append(
                all_linkages_class[all_linkages_class[linkage_type] == 1]["linkage_id"].nunique()
            )
        summary["linkages"].append(all_linkages_class["linkage_id"].nunique())

        # Calculate number of linkage types that contain observations of this class
        for linkage_type in [
            "pure",
            "pure_complete",
            "partial",
            "partial_contaminant",
            "mixed",
        ]:
            summary["unique_in_{}_linkages".format(linkage_type)].append(
                all_linkages_class[all_linkages_class[linkage_type] == 1]["object_id"].nunique()
            )

        # Calculate number of observations in different linkages for each class
        for linkage_type in [
            "pure",
            "pure_complete",
            "partial",
            "partial_contaminant",
            "mixed",
        ]:
            summary["obs_in_{}_linkages".format(linkage_type)].append(
                all_truths_class["obs_in_{}".format(linkage_type)].sum()
            )

        summary["unique_in_pure_and_partial_linkages"].append(
            all_truths_class[(all_truths_class["pure"] >= 1) & (all_truths_class["partial"] >= 1)][
                "object_id"
            ].nunique()
        )

        summary["unique_in_partial_linkages_only"].append(
            all_truths_class[(all_truths_class["pure"] == 0) & (all_truths_class["partial"] >= 1)][
                "object_id"
            ].nunique()
        )

        summary["unique_in_pure_linkages_only"].append(
            all_truths_class[(all_truths_class["pure"] >= 1) & (all_truths_class["partial"] == 0)][
                "object_id"
            ].nunique()
        )

    all_linkages.loc[all_linkages["mixed"] == 1, "object_id"] = np.NaN
    all_linkages.loc[all_linkages["mixed"] == 1, "contamination_percentage_in_linkages"] = np.NaN
    all_linkages["object_id"] = all_linkages["object_id"].astype(str)

    # Drop all duplicate linkage_id entries which has the effect of
    # dropping all but one of the entries for mixed linkages and dropping
    # the contaminant entries for partial linkages. Pure linkages are already
    # unique at this stage
    all_linkages.drop_duplicates(subset=["linkage_id"], keep="first", inplace=True)

    # Reset index after dataframe size change
    all_linkages.reset_index(drop=True, inplace=True)

    # Organize columns and rename a few
    all_linkages = all_linkages[
        [
            "linkage_id",
            # "num_obs", # UNCOMMENT FOR DEBUG
            "num_obs_in_linkage",
            "num_members",
            # "num_obs_in_observations", # UNCOMMENT FOR DEBUG
            # "percentage_in_linkage", # UNCOMMENT FOR DEBUG
            "pure",
            "pure_complete",
            "partial",
            # "partial_contaminant", # UNCOMMENT FOR DEBUG
            "mixed",
            "contamination_percentage_in_linkages",
            "found_pure",
            "found_partial",
            "found",
            "object_id",
        ]
    ]
    all_linkages.rename(
        columns={
            "object_id": "linked_object_id",
            "num_obs_in_linkage": "num_obs",
            "contamination_percentage_in_linkages": "contamination_percentage",
        },
        inplace=True,
    )

    all_objects = all_objects[
        [
            "object_id",
            # "class",
            "num_obs",
            "findable",
            "found_pure",
            "found_partial",
            "found",
            "pure",
            "pure_complete",
            "partial",
            "partial_contaminant",
            "mixed",
            "obs_in_pure",
            "obs_in_pure_complete",
            "obs_in_partial",
            "obs_in_partial_contaminant",
            "obs_in_mixed",
        ]
    ]

    summary_df = pd.DataFrame(summary)
    summary_df.sort_values(by=["num_obs", "class"], ascending=False, inplace=True, ignore_index=True)

    return all_linkages, all_objects, summary_df
