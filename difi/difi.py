import time
import warnings
import numpy as np
import pandas as pd

from .utils import _checkColumnTypes
from .utils import _checkColumnTypesEqual
from .utils import _classHandler
from .utils import _percentHandler

__all__ = ["analyzeLinkages"]

def analyzeLinkages(observations, 
                    linkageMembers, 
                    allLinkages=None, 
                    allTruths=None,
                    summary=None,
                    minObs=5, 
                    contaminationThreshold=20., 
                    classes=None,
                    verbose=True,
                    columnMapping={"linkage_id": "linkage_id",
                                   "obs_id": "obs_id",
                                   "truth": "truth"}):
    """
    Did I Find It? 
    
    Given a data frame of observations and a data frame defining possible linkages made from those observations
    this function identifies each linkage as one of three possible types:
    - pure: a linkage where all constituent observations belong to a single truth
    - partial: a linkage that contains observations belonging to multiple truths but 
        equal to or more than minObs observations of one truth and no more than the contamination threshold
        of observations of other truths. For example, a linkage with ten observations, eight of which belong to
        a single unique truth and two of which belong to other truths has contamination percentage 20%. If the threshold
        is set to 20% or greater, and minObs is less than or equal to eight then the truth with the eight observations
        is considered found and the linkage is considered a partial linkage.
    - mixed: all linkages that are neither pure nor partial.
    
    
    Parameters
    ----------
    observations : `~pandas.DataFrame`
        Pandas DataFrame with at least two columns: observation IDs and the truth values
        (the object to which the observation belongs to).
    linkageMembers : `~pandas.DataFrame`
        Pandas DataFrame with at least two columns: linkage IDs and the observation 
    allLinkages : {`~pandas.DataFrame`, None}, optional
        Pandas DataFrame with one row per linkage with at least one column: linkage IDs.
        If None, allLinkages will be created.
        [Default = None]
    allTruths : {`~pandas.DataFrame`, None}, optional
        Pandas DataFrame with one row per unique truth with at least one column: truths.
        If None, allTruths will be created.
        [Default = None]
    minObs : int, optional
        The minimum number of observations belonging to one object for a linkage to be pure. 
        The minimum number of observations belonging to one object in a contaminated linkage
        (number of contaminated observations allowed is set by the contaminationThreshold)
        for the linkage to be partial. For example, if minObs is 5 then any linkage with 5 or more 
        detections belonging to a unique object, with no detections belonging to any other object will be 
        considered a pure linkage and the object is found. Likewise, if minObs is 5 and contaminationThreshold is 
        0.2 then a linkage with 10 members, where 8 belong to one object and 2 belong to other objects, will 
        be considered a partial linkage, and the object with 8 detections is considered found. 
        [Default = 5]
    contaminationThreshold : float, optional 
        Number of detections expressed as a percentage [0-100] belonging to other objects in a linkage
        allowed for the object with the most detections in the linkage to be considered found. 
        [Default = 20]
    classes : {dict, str, None}
        Analyze observations for truths grouped in different classes. 
        str : Name of the column in the dataframe which identifies 
            the class of each truth.
        dict : A dictionary with class names as keys and a list of unique 
            truths belonging to each class as values.
        None : If there are no classes of truths.
    columnMapping : dict, optional
        The mapping of columns in observations and linkageMembers to internally used names. 
        Needs the following: "linkage_id" : ..., "truth": ... and "obs_id" : ... .
        
    Returns
    -------
    allLinkages : `~pandas.DataFrame`
        DataFrame with added pure, partial, false, contamination, num_obs, num_members, linked_truth 
    allTruths : `~pandas.DataFrame`
        DataFrame with added found_pure, found_partial, found columns. 
    summary : `~pandas.DataFrame`
        DataFrame with columns add to summarize the different type of linkages analyzed

    Raises
    ------
    TypeError : If the truth column in observations does not have type "Object", 
        or if the obs_id columns in observations and linkageMembers do not have the same type, 
        or if the linkage_id columns in allLinkages (if passed) and linkageMembers do not have the same type, 
        or if the truth columns in allTruths (if passed) and observations do not have the same type.
    """
    time_start = time.time()
    
    # Raise error if there are no observations
    if len(observations) == 0: 
        raise ValueError("There are no observations in the observations DataFrame!")
    
    
    # If allLinkages DataFrame does not exist, create it
    if allLinkages is None:
        linkage_ids = linkageMembers[columnMapping["linkage_id"]].unique()
        linkage_ids.sort()
        allLinkages = pd.DataFrame({
            columnMapping["linkage_id"] : linkage_ids})
        allLinkages[columnMapping["linkage_id"]] = allLinkages[columnMapping["linkage_id"]].astype(str)
    else:
        _checkColumnTypesEqual(allLinkages, linkageMembers, ["linkage_id"], columnMapping)
    
    # Prepare allLinkage columns
    allLinkages["num_members"] = np.ones(len(allLinkages)) * np.NaN
    allLinkages["num_obs"] = np.ones(len(allLinkages)) * np.NaN
    allLinkages["pure"] = np.zeros(len(allLinkages), dtype=int)
    allLinkages["partial"] = np.zeros(len(allLinkages), dtype=int)
    allLinkages["mixed"] = np.zeros(len(allLinkages), dtype=int)
    allLinkages["contamination"] = np.ones(len(allLinkages), dtype=int) * np.NaN
    allLinkages["linked_truth"] = np.ones(len(allLinkages), dtype=int) * np.NaN
    
    # Add the number of observations each linkage as 
    num_obs_per_linkage = linkageMembers[columnMapping["linkage_id"]].value_counts().sort_index()
    allLinkages.loc[allLinkages[columnMapping["linkage_id"]].isin(num_obs_per_linkage.index.values), "num_obs"] = num_obs_per_linkage.values

    # If allTruths DataFrame does not exist, create it
    if allTruths is None:
        truths = observations[columnMapping["truth"]].value_counts()
        allTruths = pd.DataFrame({
            columnMapping["truth"] : truths.index.values,
            "found_pure" : np.zeros(len(truths), dtype=int),
            "found_partial" : np.zeros(len(truths), dtype=int),
            "found" : np.zeros(len(truths), dtype=int)})
        allTruths[columnMapping["truth"]] = allTruths[columnMapping["truth"]].astype(str)
        
    # If it does exist, add columns
    else:
        allTruths["found_pure"] = np.zeros(len(allTruths), dtype=int)
        allTruths["found_partial"] = np.zeros(len(allTruths), dtype=int)
        allTruths["found"] = np.zeros(len(allTruths), dtype=int)
        _checkColumnTypesEqual(allTruths, observations, ["truth"], columnMapping)
        
    if "num_obs" not in allTruths.columns:
        num_obs_per_truth = observations[columnMapping["truth"]].value_counts()#.sort_index()
        allTruths.loc[allTruths[columnMapping["truth"]].isin(num_obs_per_truth.index.values), "num_obs"] =  num_obs_per_truth.values
        
    # Check column types
    _checkColumnTypes(observations, ["truth"], columnMapping)
    _checkColumnTypes(observations, ["obs_id"], columnMapping)
    _checkColumnTypes(linkageMembers, ["obs_id"], columnMapping)
    _checkColumnTypesEqual(observations, linkageMembers, ["obs_id"], columnMapping)

    if len(linkageMembers) > 0:
        
        ### Calculate the number of unique truth's per linkage
        
        # Grab only observation IDs and truth from observations
        linkage_truth = observations[[columnMapping["obs_id"], columnMapping["truth"]]]

        # Merge truth from observations with linkageMembers on observation IDs
        linkage_truth = linkage_truth.merge(
            linkageMembers[[columnMapping["linkage_id"],
                            columnMapping["obs_id"]]], 
            on=columnMapping["obs_id"])
        
        # Drop observation ID column
        linkage_truth.drop(columns=columnMapping["obs_id"], inplace=True)

        # Drop duplicate rows, any correct linkage will now only have one row since
        # all the truth values would have been the same, any incorrect linkage
        # will now have multiple rows for each unique truth value
        linkage_truth.drop_duplicates(inplace=True)

        # Sort by linkage IDs and reset index
        linkage_truth.sort_values(by=columnMapping["linkage_id"], inplace=True)
        linkage_truth.reset_index(inplace=True, drop=True)

        # Grab the number of unique truths per linkage and update 
        # the allLinkages DataFrame with the result
        unique_truths_per_linkage = linkage_truth[columnMapping["linkage_id"]].value_counts().sort_index()
        allLinkages.loc[allLinkages[columnMapping["linkage_id"]].isin(unique_truths_per_linkage.index.values), "num_members"] = unique_truths_per_linkage.values

        ### Find all the pure linkages and identify them as such
        
        # All the linkages where num_members = 1 are pure linkages
        single_member_linkages = linkage_truth[
            linkage_truth[columnMapping["linkage_id"]].isin(
                allLinkages[(allLinkages["num_members"] == 1) & (allLinkages["num_obs"] >= minObs)][columnMapping["linkage_id"]].values)]

        # Update the linked_truth field in allLinkages with the linked object
        pure_linkages = allLinkages[columnMapping["linkage_id"]].isin(single_member_linkages[columnMapping["linkage_id"]].values)
        allLinkages.loc[pure_linkages, "linked_truth"] = single_member_linkages[columnMapping["truth"]].values

        # Update the pure field in allLinkages to indicate which linkages are pure
        allLinkages.loc[(allLinkages["linked_truth"].notna()), "pure"] = 1

        ### Find all the partial linkages and identify them as such

        # Grab only observation IDs and truth from observations
        linkage_truth = observations[[columnMapping["obs_id"], columnMapping["truth"]]]

        # Merge truth from observations with linkageMembers on observation IDs
        linkage_truth = linkage_truth.merge(
            linkageMembers[[columnMapping["linkage_id"],
                            columnMapping["obs_id"]]], 
            on=columnMapping["obs_id"])

        # Remove non-pure linkages
        linkage_truth = linkage_truth[linkage_truth[columnMapping["linkage_id"]].isin(
            allLinkages[allLinkages["pure"] != 1][columnMapping["linkage_id"]])]

        # Drop observation ID column
        linkage_truth.drop(columns=columnMapping["obs_id"], inplace=True)

        # Group by linkage IDs and truths, creates a multi-level index with linkage ID
        # as the first index, then truth as the second index and as values is the count 
        # of the number of times the truth shows up in the linkage
        linkage_truth = linkage_truth.groupby(linkage_truth[[
            columnMapping["linkage_id"],
            columnMapping["truth"]]].columns.tolist(), as_index=False).size()

        #import pdb; pdb.set_trace()

        # Reset the index to create a DataFrame
        linkage_truth = linkage_truth.reset_index()
        linkage_truth[columnMapping["linkage_id"]] = linkage_truth[columnMapping["linkage_id"]].astype(str)

        # Rename 0 column to num_obs which counts the number of observations
        # each unique truth has in each linkage
        linkage_truth.rename(columns={0: "num_obs"}, inplace=True)

        # Sort by linkage ID and num_obs so that the truth with the most observations
        # in each linkage is last for each linkage
        linkage_truth.sort_values(by=[columnMapping["linkage_id"], "num_obs"], inplace=True)

        # Drop duplicate rows, keeping only the last row 
        linkage_truth.drop_duplicates(subset=[columnMapping["linkage_id"]], inplace=True, keep="last")

        # Grab all linkages and merge truth from observations with linkageMembers on observation IDs
        linkage_truth = linkage_truth.merge(allLinkages[[columnMapping["linkage_id"], "num_obs"]], on=columnMapping["linkage_id"])

        # Rename num_obs column in allLinkages to total_num_obs
        linkage_truth.rename(columns={"num_obs_x": "num_obs", "num_obs_y": "total_num_obs"}, inplace=True)

        # Calculate contamination 
        linkage_truth["contamination"] = 100. * (1 - linkage_truth["num_obs"] / linkage_truth["total_num_obs"])

        # Select partial linkages: have at least the minimum observations of a single truth and have no
        # more than x% contamination
        partial_linkages = linkage_truth[(linkage_truth["num_obs"] >= minObs) 
                                       & (linkage_truth["contamination"] <= contaminationThreshold)]

        # Update allLinkages to indicate partial linkages, update linked_truth field
        # Set every linkage that isn't partial or pure to false
        allLinkages.loc[allLinkages[columnMapping["linkage_id"]].isin(partial_linkages[columnMapping["linkage_id"]]), "linked_truth"] = partial_linkages[columnMapping["truth"]].values
        allLinkages.loc[allLinkages[columnMapping["linkage_id"]].isin(partial_linkages[columnMapping["linkage_id"]]), "partial"] = 1
        allLinkages.loc[allLinkages[columnMapping["linkage_id"]].isin(partial_linkages[columnMapping["linkage_id"]]), "contamination"] = partial_linkages["contamination"].values
        allLinkages.loc[(allLinkages["pure"] != 1) & (allLinkages["partial"] != 1), "mixed"] = 1
        allLinkages.loc[(allLinkages["pure"] == 1), "contamination"] = 0.0

        # Update allTruths to indicate which objects were found in pure and partial linkages, if found in either the object is considered found
        allTruths.loc[allTruths[columnMapping["truth"]].isin(allLinkages[allLinkages["pure"] == 1]["linked_truth"].values), "found_pure"] = 1
        allTruths.loc[allTruths[columnMapping["truth"]].isin(allLinkages[allLinkages["partial"] == 1]["linked_truth"].values), "found_partial"] = 1
        allTruths.loc[(allTruths["found_pure"] == 1) | (allTruths["found_partial"] == 1), "found"] = 1
        
        linkageMembers_with_truth = linkageMembers.merge(observations[[columnMapping["obs_id"], columnMapping["truth"]]], on=columnMapping["obs_id"])
        allTruths_with_linkages = allTruths.merge(allLinkages[[columnMapping["linkage_id"], "num_obs", "linked_truth", "pure", "partial"]], left_on=columnMapping["truth"], right_on="linked_truth")
        mixed_linkages = allLinkages[allLinkages["mixed"] == 1][columnMapping["linkage_id"]].unique()
        
        # Count the number of pure, partial, and mixed linkages there are
        num_linkages_all = len(allLinkages)
        num_pure_all = len(allLinkages[allLinkages["pure"] == 1])
        num_pure_unique_all = allLinkages[(~allLinkages["linked_truth"].isna())]["linked_truth"].nunique()
        num_pure_complete_all = len(allTruths_with_linkages[(
            (allTruths_with_linkages["pure"] == 1) 
            & (allTruths_with_linkages["num_obs_x"] == allTruths_with_linkages["num_obs_y"]))])
        num_partial_all = len(allLinkages[allLinkages["partial"] == 1])
        num_mixed_all = len(allLinkages[allLinkages["mixed"] == 1])

        # Number of clusters of each cluster type (pure, partial, mixed) for each class
        num_pure_list = []                              
        num_pure_complete_list = []             
        num_partial_list = []
        num_mixed_list = [] 
        num_linkages_list = []
        
        # Number of unique truths in each cluster type
        num_pure_unique_list = []  
        num_pure_complete_unique_list = []                        
        num_partial_unique_list = []                      
        num_mixed_unique_list = []
        num_pure_partial_unique_list = []
        num_pure_only_unique_list = []
        num_partial_only_unique_list = []
        
        # Completeness for each class
        completeness_list = []
        num_found_list = []
        num_findable_list = []
        num_findable_found_list = []
        num_findable_missed_list = []
        num_not_findable_found_list = []
        num_not_findable_missed_list = []
        
        class_list, truths_list = _classHandler(classes, observations, columnMapping)

        for c, v in zip(class_list, truths_list):
            
            # Find the number of unique truths in class
            num_unique = len(np.unique(v))

            # Set a mask for both the allTruths and allLinkages dataframes
            mask_allTruths = (allTruths[columnMapping["truth"]].isin(v))
            mask_allLinkages = (allLinkages["linked_truth"].isin(v))

            # Find number of pure linkages
            num_pure = len(allLinkages[mask_allLinkages & (allLinkages["pure"] == 1)])
            pure_unique = allLinkages[mask_allLinkages & (allLinkages["pure"] == 1)]["linked_truth"].unique()
            num_pure_unique = len(pure_unique)

            # Find number of pure linkages that contain all observations (complete pure linkages)
            num_pure_complete = allTruths_with_linkages[((allTruths_with_linkages["linked_truth"].isin(v))
                                        & (allTruths_with_linkages["pure"] == 1) 
                                        & (allTruths_with_linkages["num_obs_x"] == allTruths_with_linkages["num_obs_y"]))][columnMapping["linkage_id"]].nunique()
            num_pure_complete_unique = allTruths_with_linkages[((allTruths_with_linkages["linked_truth"].isin(v))
                                        & (allTruths_with_linkages["pure"] == 1) 
                                        & (allTruths_with_linkages["num_obs_x"] == allTruths_with_linkages["num_obs_y"]))]["linked_truth"].nunique()

            # Find number of partial linkages
            num_partial = len(allLinkages[mask_allLinkages & (allLinkages["partial"] == 1)])
            partial_unique = allLinkages[mask_allLinkages & (allLinkages["partial"] == 1)]["linked_truth"].unique()
            num_partial_unique = len(partial_unique)

            # Find number of class in mixed linkages
            mixed_class = linkageMembers_with_truth[linkageMembers_with_truth[columnMapping["linkage_id"]].isin(mixed_linkages) & linkageMembers_with_truth[columnMapping["truth"]].isin(v)]
            num_mixed = mixed_class[columnMapping["linkage_id"]].nunique()
            num_mixed_unique = mixed_class[columnMapping["truth"]].nunique()

            # Find the number of linkages that have a linked truth belonging to a 
            # member of this class
            num_pure_and_partial = num_pure + num_partial 
            num_linkages = num_pure_and_partial + num_mixed

            # Find number of truths in both pure and partial linkages
            num_pure_partial_unique = len(np.unique(np.concatenate([pure_unique, partial_unique])))

            # Find number of truths only in pure and not in partial linkages
            num_pure_only_unique = len(np.unique(pure_unique[np.isin(pure_unique, partial_unique, invert=True)]))

            # Find number of truths only in partial and not in pure linkages
            num_partial_only_unique = len(np.unique(partial_unique[np.isin(partial_unique, pure_unique, invert=True)]))

            num_found = len(allTruths[mask_allTruths & (allTruths["found"] == 1)])
            if "findable" in allTruths.columns:
                num_findable = len(allTruths[mask_allTruths & (allTruths["findable"] == 1)])
                num_findable_found = len(allTruths[mask_allTruths & (allTruths["findable"] == 1) & (allTruths["found"] == 1)])
                num_findable_missed = len(allTruths[mask_allTruths & (allTruths["findable"] == 1) & (allTruths["found"] == 0)])
                num_not_findable_found = len(allTruths[mask_allTruths & (allTruths["found"] == 1) & (allTruths["findable"] == 0)])
                num_not_findable_missed = len(allTruths[mask_allTruths & (allTruths["found"] == 0) & (allTruths["findable"] == 0)])
                                
                if num_findable_found == 0:
                    completeness = 0.0   
                else:
                    completeness = num_findable_found / (num_findable_found + num_findable_missed) * 100.

            else:
                # If 'findable' is not a column in allTruths then issue 
                # warning that completeness cannot be calculated
                warnings.warn("No 'findable' column was found in allTruths. Cannot compute completeness.", UserWarning)

                completeness = np.NaN
                num_findable = np.NaN
                num_findable_found = np.NaN
                num_findable_missed = np.NaN
                num_not_findable_found = np.NaN
                num_not_findable_missed = np.NaN

            # Append results to lists
            num_pure_list.append(num_pure)                     
            num_pure_unique_list.append(num_pure_unique)           
            num_pure_complete_list.append(num_pure_complete)          
            num_pure_complete_unique_list.append(num_pure_complete_unique)   
            num_partial_list.append(num_partial)                 
            num_partial_unique_list.append(num_partial_unique)                                           
            num_mixed_list.append(num_mixed)                                                        
            num_mixed_unique_list.append(num_mixed_unique)
            num_linkages_list.append(num_linkages)
            num_pure_partial_unique_list.append(num_pure_partial_unique)
            num_pure_only_unique_list.append(num_pure_only_unique)
            num_partial_only_unique_list.append(num_partial_only_unique)
            completeness_list.append(completeness)
            num_findable_list.append(num_findable)
            num_found_list.append(num_found)
            num_findable_found_list.append(num_findable_found)
            num_findable_missed_list.append(num_findable_missed)
            num_not_findable_found_list.append(num_not_findable_found)
            num_not_findable_missed_list.append(num_not_findable_missed)

            print("{}".format(c))
            print("----------------------------------------------------------------")
            print("                                   Number  (% class)  (% total)")
            print("All linkages:                    {:>8d}  ({:6.2f}%)  ({:6.2f}%)".format(
                num_linkages,
                _percentHandler(num_linkages, num_linkages),
                _percentHandler(num_linkages, num_linkages_all)
                )
            )
            print("Pure linkages:                   {:>8d}  ({:6.2f}%)  ({:6.2f}%)".format(
                num_pure,
                _percentHandler(num_pure, num_linkages),
                _percentHandler(num_pure, num_linkages_all)
                )
            )
            print("Complete pure linkages:          {:>8d}  ({:6.2f}%)  ({:6.2f}%)".format(
                num_pure_complete,
                _percentHandler(num_pure_complete, num_linkages),
                _percentHandler(num_pure_complete, num_linkages_all)
                )
            )
            print("Partial linkages:                {:>8d}  ({:6.2f}%)  ({:6.2f}%)".format(
                num_partial,
                _percentHandler(num_partial, num_linkages),
                _percentHandler(num_partial, num_linkages_all)
                )
            )
            print("Pure and partial linkages:       {:>8d}  ({:6.2f}%)  ({:6.2f}%)".format(
                num_pure_and_partial,
                _percentHandler(num_pure_and_partial, num_linkages),
                _percentHandler(num_pure_and_partial, num_linkages_all)
                )
            )
            print("Mixed linkages:                  {:>8d}  ({:6.2f}%)  ({:6.2f}%)".format(
                num_mixed,
                _percentHandler(num_mixed, num_linkages),
                _percentHandler(num_mixed, num_linkages_all)
                )
            )
            print("")

            print("                                   Number  (% class) (% findable)")
            print("Unique {}:".format(c))
            print(" ..in pure linkages:             {:>8d}  ({:6.2f}%)  ({:6.2f}%)".format(
                num_pure_unique,
                _percentHandler(num_pure_unique, num_unique),
                _percentHandler(num_pure_unique, num_findable),
                )
            )
            print(" ..in complete pure linkages:    {:>8d}  ({:6.2f}%)  ({:6.2f}%)".format(
                num_pure_complete_unique,
                _percentHandler(num_pure_complete_unique, num_unique),
                _percentHandler(num_pure_complete_unique, num_findable)
                )
            )
            print(" ..in partial linkages:          {:>8d}  ({:6.2f}%)  ({:6.2f}%)".format(
                num_partial_unique,
                _percentHandler(num_partial_unique, num_unique),
                _percentHandler(num_partial_unique, num_findable)
                )
            )
            print(" ..in pure and partial linkages: {:>8d}  ({:6.2f}%)  ({:6.2f}%)".format(
                num_pure_partial_unique,
                _percentHandler(num_pure_partial_unique, num_unique),
                _percentHandler(num_pure_partial_unique, num_findable)
                )
            )
            print(" ..in mixed linkages:            {:>8d}  ({:6.2f}%)  ({:6.2f}%)".format(
                num_mixed_unique,
                _percentHandler(num_mixed_unique, num_unique),
                _percentHandler(num_mixed_unique, num_findable)
                )
            )
            print(" ..only in pure linkages:        {:>8d}  ({:6.2f}%)  ({:6.2f}%)".format(
                num_pure_only_unique,
                _percentHandler(num_pure_only_unique, num_unique),
                _percentHandler(num_pure_only_unique, num_findable)
                )
            )
            print(" ..only in partial linkages:     {:>8d}  ({:6.2f}%)  ({:6.2f}%)".format(
                num_partial_only_unique,
                _percentHandler(num_partial_only_unique, num_unique),
                _percentHandler(num_partial_only_unique, num_findable)
                )
            )
            print("")
            
            
            print("Findable:                        {}".format(
                num_findable)
            )
            print("Found:                           {}".format(
                num_found)
            )
            print("Findable found:                  {}".format(
                num_findable_found)
            )
            print("Findable missed:                 {}".format(
                num_findable_missed)
            )
            print("Not findable found:              {}".format(
                num_not_findable_found)
            )
            print("Not findable missed:             {}".format(
                num_not_findable_missed)
            )
            print()
            print("Completeness (findable found / findable):            {:6.2f}%".format(
                completeness)
            )
            print()
                
        summary = pd.DataFrame({
            "class" : class_list,
            "completeness" : completeness_list,
            "findable" : num_findable_list,
            "found" : num_found_list,
            "findable_found" : num_findable_found_list,
            "findable_missed" : num_findable_missed_list,
            "not_findable_found" : num_not_findable_found_list,
            "not_findable_missed" : num_not_findable_missed_list,
            "linkages" : num_linkages_list,
            "pure_linkages" : num_pure_list,                    
            "pure_complete_linkages" : num_pure_complete_list,   
            "partial_linkages" : num_partial_list,    
            "mixed_linkages" : num_mixed_list,
            "unique_in_pure" : num_pure_unique_list,
            "unique_in_pure_complete" : num_pure_complete_unique_list,
            "unique_in_partial" : num_partial_unique_list,
            "unique_in_pure_and_partial" : num_pure_partial_unique_list,
            "unique_in_pure_only" : num_pure_only_unique_list,
            "unique_in_partial_only" : num_partial_only_unique_list,
            "unique_in_mixed" : num_mixed_unique_list,     
        })
        summary.sort_values(by="class", inplace=True)
        summary.reset_index(inplace=True, drop=True)

    return allLinkages, allTruths, summary
    
    

