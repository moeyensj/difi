import warnings
import numpy as np
import pandas as pd

from .utils import _checkColumnTypes
from .utils import _checkColumnTypesEqual
from .utils import _classHandler
from .utils import _percentHandler

__all__ = ["analyzeLinkages"]

def analyzeLinkages(observations, 
                    linkage_members, 
                    all_truths=None,
                    min_obs=5, 
                    contamination_percentage=20., 
                    classes=None,
                    column_mapping={
                        "linkage_id": "linkage_id",
                        "obs_id": "obs_id",
                        "truth": "truth"
                    }):
    """
    Did I Find It? 
    
    Given a data frame of observations and a data frame defining possible linkages made from those observations
    this function identifies each linkage as one of three possible types:
    - pure: a linkage where all constituent observations belong to a single truth
    - partial: a linkage that contains observations belonging to multiple truths but 
        equal to or more than min_obs observations of one truth and no more than the contamination threshold
        of observations of other truths. For example, a linkage with ten observations, eight of which belong to
        a single unique truth and two of which belong to other truths has contamination percentage 20%. If the threshold
        is set to 20% or greater, and min_obs is less than or equal to eight then the truth with the eight observations
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
    all_truths : {`~pandas.DataFrame`, None}, optional
        Pandas DataFrame with one row per unique truth with at least one column: truths.
        If None, all_truths will be created.
        [Default = None]
    min_obs : int, optional
        The minimum number of observations belonging to one object in a pure linkage for 
        that object to be considered found. In a partial linkage, for an object to be considered
        found it must have equal to or more than this number of observations for it to be considered
        found. 
    contamination_percentage : float, optional 
        Number of detections expressed as a percentage [0-100] belonging to other objects in a linkage 
        for that linkage to considered partial. For example, if contamination_percentage is 
        20% then a linkage with 10 members, where 8 belong to one object and 2 belong to other objects, will 
        be considered a partial linkage.
        [Default = 20]
    classes : {dict, str, None}
        Analyze observations for truths grouped in different classes. 
        str : Name of the column in the observations dataframe which identifies 
            the class of each truth.
        dict : A dictionary with class names as keys and a list of unique 
            truths belonging to each class as values.
        None : If there are no classes of truths.
    column_mapping : dict, optional
        The mapping of columns in observations and linkage_members to internally used names. 
        Needs the following: "linkage_id" : ..., "truth": ... and "obs_id" : ... .
        
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
                Number of unique truths contained in the linkage.
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
            "linked_truth" : str
                The truth linked in the linkage if the linkage is pure or partial, NaN otherwise.

    all_truths: `~pandas.DataFrame`
        A per-truth summary.
        
        Columns:
            "truth" : str
                Truth
            "num_obs" : int
                Number of observations in the observations dataframe
                for each truth
            "findable" : int
                1 if the object is findable, 0 if the object is not findable.
                (NaN if no findable column is found in the all_truths dataframe)
            "found_pure" : int
                Number of pure linkages that contain at least min_obs observations.
            "found_partial" : int
                Number of partial linkages that contain at least min_obs observations
                and contain no more than the contamination_percentage of observations 
                of other truths.
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
                Number of unique truths that belong to the class.
            "num_obs" : int
                Number of observations of truths belonging to the class in 
                the observations dataframe. 
            "completeness" : float
                Percent of truths deemed findable that are found in pure or 
                partial linkages that contain more than min_obs observations.
            "findable" : int
                Number of truths deemed findable (all_truths must be passed to this 
                function with a findable column)
            "found" : int
                Number of truths found in pure and partial linkages with equal to or 
                more than min_obs observations.
            "findable_found" : int
                Number of truths deemed findable that were found. 
            "findable_missed" : int
                Number of truths deemed findable that were not found. 
            "not_findable_found" : int
                Number of truths deemed not findable that were found (serendipitous discoveries).
            "not_findable_missed" : int
                Number of truths deemed not findable that were not found. 
            "linkages" : int
                Number of unique linkages that contain observations of this class of truths.
            "pure_linkages" : int
                Number of pure linkages that contain observations of this class of truths.
            "pure_complete_linkages" : int 
                Number of complete pure linkages that contain observations of this class of truths.
            "partial_linkages" : int
                Number of partial linkages that contain observations of this class of truths.
            "partial_contaminant_linkages" : int
                Number of partial linkages that are contaminated by observations of this class of truths.
            "mixed_linkages" : 
                Number of mixed linkages that contain observations of this class of truths.
            "unique_in_pure_linkages" : int
                Number of unique truths in pure linkages.
            "unique_in_pure_complete_linkages" : int
                Number of unique truths in pure complete linkages (subset of unique_in_pure_linkages).
            "unique_in_pure_linkages_only" : int
                Number of unique truths in pure linkages only (not in partial linkages but can be in mixed
                linkages). 
            "unique_in_partial_linkages_only" : int
                Number of unique truths in partial linkages only (not in pure linkages but can be in mixed
                linkages).
            "unique_in_pure_and_partial_linkages" : int
                Number of unique truths that appear in both pure and partial linkages.
            "unique_in_partial_linkages" : int
                Number of unique truths in partial linkages.
            "unique_in_partial_contaminant_linkages" : int
                Number of unique truths that contaminate partial linkages.
            "unique_in_mixed_linkages" : int
                Number of unique truths in mixed linkages.
            "obs_in_pure_linkages" : int
                Number of observations of truths of this class in pure linkages.
            "obs_in_pure_complete_linkages" : int
                Number of observations of truths of this class in complete pure linkages.
            "obs_in_partial_linkages" : int
                Number of observations of truths of this class in partial linkages.
            "obs_in_partial_contaminant_linkages" : int
                Number of observations of truths of this class that contaminate partial linkages.
            "obs_in_mixed_linkages" : int
                Number of observations of truths of this class in mixed linkages.
            
    Raises
    ------
    TypeError : If the truth column in observations does not have type "Object", 
        or if the obs_id columns in observations and linkage_members do not have the same type, 
        or if the linkage_id columns in all_linkages (if passed) and linkage_members do not have the same type, 
        or if the truth columns in all_truths (if passed) and observations do not have the same type.
    """
    # Get column names
    linkage_id_col = column_mapping["linkage_id"]
    truth_col = column_mapping["truth"]
    obs_id_col = column_mapping["obs_id"]
    
    # Raise error if there are no observations
    if len(observations) == 0: 
        raise ValueError("There are no observations in the observations DataFrame!")

    findable_present = True
    # If all_truths DataFrame does not exist, create it
    if all_truths is None:
        truths = observations[truth_col].value_counts()
        
        all_truths = pd.DataFrame({
            truth_col : truths.index.values,
            #"class" : ["None" for i in range(len(truths))],
            "num_obs" : np.zeros(len(truths), dtype=int),
            "findable" : [np.NaN for i in range(len(truths))]})
        all_truths[truth_col] = all_truths[truth_col].astype(str)
        
        num_obs_per_truth = observations[truth_col].value_counts()
        all_truths.loc[all_truths[truth_col].isin(num_obs_per_truth.index.values), "num_obs"] =  num_obs_per_truth.values
        
        all_truths.sort_values(by=["num_obs", truth_col], ascending=[False, True], inplace=True)
        all_truths.reset_index(inplace=True, drop=True)

        findable_present = False
        
    # If it does exist, add columns
    else:
        if "findable" not in all_truths.columns:
            warn = (
                "No findable column found in all_truths. Completeness\n" \
                "statistics can not be calculated."
            )
            warnings.warn(warn)
            findable_present = False
        _checkColumnTypesEqual(all_truths, observations, ["truth"], column_mapping)
        
        
    # Check column types
    _checkColumnTypes(observations, ["truth"], column_mapping)
    _checkColumnTypes(observations, ["obs_id"], column_mapping)
    _checkColumnTypes(linkage_members, ["obs_id"], column_mapping)
    _checkColumnTypesEqual(observations, linkage_members, ["obs_id"], column_mapping)

    if len(linkage_members) > -1:
        
        # Grab only observation IDs and truth from observations
        all_linkages = observations[[obs_id_col, truth_col]].copy()

        # Merge truth from observations with linkage_members on observation IDs
        all_linkages = all_linkages.merge(
            linkage_members[[linkage_id_col,
                            obs_id_col]], 
            on=obs_id_col)
        
        # Group the data frame of truths, linkage_ids and 
        # observation IDs by truth and linkage ID
        # then count the number of occurences
        all_linkages = all_linkages.groupby(
            by=[truth_col, linkage_id_col]
        ).count().reset_index()
        all_linkages.rename(
            columns={
                obs_id_col : "num_obs"
            }, 
            inplace=True
        )

        # Calculate the total number of observations in each linkage
        num_obs_in_linkage = all_linkages.groupby(
            by=[linkage_id_col]
        )["num_obs"].sum().to_frame(name="num_obs_in_linkage")
        num_obs_in_linkage.reset_index(
            drop=False,
            inplace=True
        )

        # Merge with num_obs_in_linkage to get the total 
        # number of observations in each linkage
        all_linkages = all_linkages.merge(
            num_obs_in_linkage,
            left_on=linkage_id_col, 
            right_on=linkage_id_col, 
            suffixes=("", "_")
        )

        # Calculate the number of unique truths in each linkage
        num_truth_in_linkage = all_linkages.groupby(
            by=[linkage_id_col]
        )[truth_col].nunique().to_frame(name="num_members")
        num_obs_in_linkage.reset_index(
            drop=False,
            inplace=True
        )

        # Merge with num_obs_in_linkage to get the total 
        # number of observations in each linkage
        all_linkages = all_linkages.merge(
            num_truth_in_linkage,
            left_on=linkage_id_col, 
            right_on=linkage_id_col, 
            suffixes=("", "_")
        )

        # Merge with all_truths to get the total number of 
        # observations in the observations data frame
        all_linkages = all_linkages.merge(
            all_truths[[truth_col, "num_obs"]].rename(
                columns={ 
                    "num_obs" : "num_obs_in_observations"},
            ),
            left_on=truth_col, 
            right_on=truth_col, 
            suffixes=("", "_")
        )

        # For each truth calculate the percent of observations 
        # in a linkage that belong to that truth 
        all_linkages["percentage_in_linkage"] = 100. * all_linkages["num_obs"] / all_linkages["num_obs_in_linkage"]
        all_linkages["contamination_percentage_in_linkages"] = 100 - all_linkages["percentage_in_linkage"]

        # Sort by linkage_id and the percentage then reset the index 
        all_linkages.sort_values(
            by=[linkage_id_col, "percentage_in_linkage"],
            ascending=[True, False],
            inplace=True
        )
        all_linkages.reset_index(
            drop=True,
            inplace=True
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
        all_linkages.loc[
            (all_linkages["num_obs"] == all_linkages["num_obs_in_linkage"]), 
            "pure"] = 1

        # Complete pure linkages: any linkage where all observations of a truth are linked
        all_linkages.loc[
            (all_linkages["num_obs"] == all_linkages["num_obs_in_observations"])
            & (all_linkages["pure"] == 1), 
            "pure_complete"] = 1

        # Partial linkages: any linkage where up to a contamination percentage of observations belong to other truths
        all_linkages.loc[
            (all_linkages["pure"] == 0) 
            & (all_linkages["contamination_percentage_in_linkages"] <= contamination_percentage), 
            "partial"] = 1
        partial_linkages = all_linkages[all_linkages["partial"] == 1][linkage_id_col].unique()
        all_linkages.loc[
            (all_linkages[linkage_id_col].isin(partial_linkages) 
             & (all_linkages["contamination_percentage_in_linkages"] > contamination_percentage)), "partial_contaminant"] = 1
        
        # If the contamination percentage is high it may set linkages with no clear majority of detections belonging to one object
        # as partials.. these are actually mixed so make sure they are correctly indentified.
        contamination_counts = all_linkages[all_linkages[linkage_id_col].isin(partial_linkages)].groupby(linkage_id_col)["contamination_percentage_in_linkages"].nunique()
        no_majority_partials = contamination_counts[contamination_counts.values == 1].index.values
        all_linkages.loc[all_linkages[linkage_id_col].isin(no_majority_partials), "partial"] = 0
        all_linkages.loc[all_linkages[linkage_id_col].isin(no_majority_partials), "partial_contaminant"] = 0
        all_linkages.loc[all_linkages[linkage_id_col].isin(no_majority_partials), "mixed"] = 1

        # Mixed linkages: any linkage that is not pure or partial
        all_linkages.loc[
            (all_linkages["pure"] == 0) 
            & (all_linkages["partial"] == 0)
            & (all_linkages["partial_contaminant"] == 0), 
            "mixed"] = 1
        
        # Update found columns
        all_linkages.loc[(all_linkages["num_obs"] >= min_obs) & (all_linkages["pure"] == 1), "found_pure"] = 1
        all_linkages.loc[(all_linkages["num_obs"] >= min_obs) & (all_linkages["partial"] == 1), "found_partial"] = 1
        all_linkages.loc[(all_linkages["found_pure"] == 1) | (all_linkages["found_partial"] == 1), "found"] = 1

        
        # Calculate number of observations in pure linkages for each truth
        pure_obs = all_linkages[all_linkages["pure"] == 1].groupby(by=truth_col)["num_obs"].sum().to_frame(name="obs_in_pure")
        pure_obs.reset_index(
            inplace=True
        )
        
        # Calculate number of observations in pure complete linkages for each truth
        pure_complete_obs = all_linkages[all_linkages["pure_complete"] == 1].groupby(by=truth_col)["num_obs"].sum().to_frame(name="obs_in_pure_complete")
        pure_complete_obs.reset_index(
            inplace=True
        )
        
        # Calculate number of observations in partial linkages for each truth
        partial_obs = all_linkages[all_linkages["partial"] == 1].groupby(by=truth_col)["num_obs"].sum().to_frame(name="obs_in_partial")
        partial_obs.reset_index(
            inplace=True
        )
        
        # Calculate number of observations in partial linkages for each truth
        partial_contaminant_obs = all_linkages[(all_linkages["partial_contaminant"] == 1)].groupby(by=truth_col)["num_obs"].sum().to_frame(name="obs_in_partial_contaminant")
        partial_contaminant_obs.reset_index(
            inplace=True
        )

        # Calculate number of observations in mixed linkages for each truth
        mixed_obs = all_linkages[all_linkages["mixed"] == 1].groupby(by=truth_col)["num_obs"].sum().to_frame(name="obs_in_mixed")
        mixed_obs.reset_index(
            inplace=True
        )
        

        linkage_types = all_linkages.groupby(by=[truth_col])[["pure", "pure_complete", "partial", "partial_contaminant", "mixed", "found_pure", "found_partial", "found"]].sum()
        linkage_types.reset_index(
            inplace=True
        )

        for df in [pure_obs, pure_complete_obs, partial_obs, partial_contaminant_obs, mixed_obs]:
            linkage_types = linkage_types.merge(df, on=truth_col, how="outer")
            
        all_truths = all_truths.merge(
            linkage_types, 
            on=truth_col,  
            how="outer"
        )
        all_truths_int_cols = [
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
        all_truths[all_truths_int_cols] = all_truths[all_truths_int_cols].fillna(0)
        all_truths[all_truths_int_cols] = all_truths[all_truths_int_cols].astype(int)
        
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
            "obs_in_mixed_linkages"
        ]
        summary = {c : [] for c in summary_cols}
        
        class_list, truths_list = _classHandler(classes, observations, column_mapping)

        # Loop through the classes and summarize the results
        for c, v in zip(class_list, truths_list):

            # Create masks for the class of truths
            all_truths_class = all_truths[all_truths[truth_col].isin(v)]
            all_linkages_class = all_linkages[all_linkages[truth_col].isin(v)]
            observations_class = observations[observations[truth_col].isin(v)]

            # Add class and the number of members to the summary
            summary["class"].append(c)
            summary["num_members"].append(len(v))
            summary["num_obs"].append(len(observations_class))


            # Number of truths found
            found = len((all_truths_class[all_truths_class["found"] >= 1]))
            summary["found"].append(found)

            if findable_present:

                # Number of truths findable
                findable = len(all_truths_class[all_truths_class["findable"] == 1])
                summary["findable"].append(findable)

                # Number of findable truths found
                findable_found = len(
                    all_truths_class[
                        (all_truths_class["findable"] == 1) 
                        & (all_truths_class["found"] >= 1)
                    ]
                )
                summary["findable_found"].append(findable_found)
                
                # Calculate completeness
                if findable == 0:
                    completeness = np.NaN
                else:
                    completeness = 100. * findable_found / findable
                summary["completeness"].append(completeness)

                # Number of findable truths missed
                findable_missed = len(
                    all_truths_class[
                        (all_truths_class["findable"] == 1) 
                        & (all_truths_class["found"] == 0)
                    ]
                )
                summary["findable_missed"].append(findable_missed)

                # Number of not findable truths found
                not_findable_found = len(
                    all_truths_class[
                        (all_truths_class["findable"] == 0) 
                        & (all_truths_class["found"] >= 1)
                    ]
                )
                summary["not_findable_found"].append(not_findable_found)

                # Number of not findable truths missed
                not_findable_missed = len(
                    all_truths_class[
                        (all_truths_class["findable"] == 0) 
                        & (all_truths_class["found"] == 0)
                    ]
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
            for linkage_type in ["found_pure", "found_partial", "pure", "pure_complete", "partial", "partial_contaminant", "mixed"]:
                summary["{}_linkages".format(linkage_type)].append(
                    all_linkages_class[all_linkages_class[linkage_type] == 1][linkage_id_col].nunique()
                )
            summary["linkages"].append(all_linkages_class[linkage_id_col].nunique())

            # Calculate number of linkage types that contain observations of this class
            for linkage_type in ["pure", "pure_complete", "partial", "partial_contaminant", "mixed"]:
                summary["unique_in_{}_linkages".format(linkage_type)].append(
                    all_linkages_class[all_linkages_class[linkage_type] == 1][truth_col].nunique()
                )

            # Calculate number of observations in different linkages for each class
            for linkage_type in ["pure", "pure_complete", "partial", "partial_contaminant", "mixed"]:
                summary["obs_in_{}_linkages".format(linkage_type)].append(
                    all_truths_class["obs_in_{}".format(linkage_type)].sum()
                )
                
            summary["unique_in_pure_and_partial_linkages"].append(
                all_truths_class[
                    (all_truths_class["pure"] >= 1) 
                    & (all_truths_class["partial"] >= 1)
                ][truth_col].nunique()
            )
            
            summary["unique_in_partial_linkages_only"].append(
                all_truths_class[
                    (all_truths_class["pure"] == 0) 
                    & (all_truths_class["partial"] >= 1)
                ][truth_col].nunique()
            )
            
            summary["unique_in_pure_linkages_only"].append(
                all_truths_class[
                    (all_truths_class["pure"] >= 1) 
                    & (all_truths_class["partial"] == 0)
                ][truth_col].nunique()
            )
        
        
    
        all_linkages.loc[all_linkages["mixed"] == 1, truth_col] = np.NaN
        all_linkages.loc[all_linkages["mixed"] == 1, "contamination_percentage_in_linkages"] = np.NaN
        all_linkages[truth_col] = all_linkages[truth_col].astype(str)
       
        
        # Drop all duplicate linkage_id entries which has the effect of 
        # dropping all but one of the entries for mixed linkages and dropping 
        # the contaminant entries for partial linkages. Pure linkages are already
        # unique at this stage
        all_linkages.drop_duplicates(
            subset=[linkage_id_col],
            keep="first", 
            inplace=True
        )
        
        # Reset index after dataframe size change
        all_linkages.reset_index(
            drop=True,
            inplace=True
        )
        
        # Organize columns and rename a few
        all_linkages = all_linkages[[
            linkage_id_col,
            #"num_obs",
            "num_obs_in_linkage",
            "num_members",
            #"num_obs_in_observations",
            #"percentage_in_linkage",
            "pure",
            "pure_complete",
            "partial",
            # "partial_contaminant", # UNCOMMENT FOR DEBUG
            "mixed",
            "contamination_percentage_in_linkages",
            "found_pure",
            "found_partial",
            "found",
            truth_col]]
        all_linkages.rename(columns={
                truth_col : "linked_truth",
                "num_obs_in_linkage" : "num_obs",
                "contamination_percentage_in_linkages" : "contamination_percentage",
            },
            inplace=True
        )
        
        all_truths = all_truths[[
            "truth",
            #"class",
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
        ]]
        
        summary = pd.DataFrame(summary)
        summary.sort_values(by=["num_obs", "class"], ascending=False, inplace=True)
        summary.reset_index(inplace=True, drop=True)
        
    else:
        
        all_linkages = pd.DataFrame(
            columns=[
                linkage_id_col,
                "num_obs",
                "num_members",
                "pure",
                "pure_complete",
                "partial",
                "mixed",
                "contamination_percentage",
                "found_pure",
                "found_partial",
                "found",
                "linked_truth"
            ]
        )

        summary = pd.DataFrame(
            columns=summary_cols
        )

    return all_linkages, all_truths, summary