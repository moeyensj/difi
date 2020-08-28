import numpy as np
import pandas as pd


def createPureLinkage(truth, observations, all_linkages, all_truths, summary, min_obs=5, min_linkage_length=5):
    """
    Create a pure linkage: a linkage containing only the observations of 
    one unique truth. 
    
    Parameters
    ----------
    truth : str
        The name / object ID of the truth
    observations : `~pandas.DataFrame`
        A dataframe containing observations. Must have at least the following
        two columns: 1) an observation ID column and 2) a truth column.
    all_linkages : dict
        Dictionary of various linkage properties that is updated by this function.
        This dictionary is converted into a `pandas.DataFrame` by the 
        createTestData function.
    all_truths : `~pandas.DataFrame`
        

    summary : `~pandas.DataFrame`

    
    min_obs : int, optional
        The minimum number of observations to be considered found.
    min_linkage_length : int, optional
        The minimum length of a linkage. If the truth does not have equal to or 
        more than this many observations no linkage will be created. 

    Returns
    -------
    linkage : `~numpy.ndarray` (N) or None
        An array of observation IDs if a linkage can be created
        with more than min_linkage_length observations, if not returns
        None. 
    """
    # Establish what the maximum number of observations 
    # can be for this given truth
    max_obs = observations[observations["truth"] == truth]["obs_id"].nunique()
    
    # If the maximum number of observations is less than the
    # minimum allowed linkage length, we can't create a linkage
    if max_obs < min_linkage_length:
        return None
    
    # Randomly pick a linkage length and check to see if it is complete
    num_obs = np.random.choice(np.arange(min_linkage_length, max_obs + 1))
    is_complete = (num_obs == max_obs)
    
    # Select observations to form the linkage
    linkage = np.random.choice(observations[observations["truth"] == truth]["obs_id"].values, num_obs, replace=False)
    linkage.sort()
    is_found = (len(linkage) >= min_obs)
    
    # Update attributes in the all_linkages dictionary accordingly
    all_linkages["num_obs"].append(len(linkage))
    all_linkages["num_members"].append(1)
    all_linkages["pure"].append(1)
    if is_complete:
        all_linkages["pure_complete"].append(1)
    else:
        all_linkages["pure_complete"].append(0)
    all_linkages["partial"].append(0)
    all_linkages["mixed"].append(0)
    all_linkages["contamination_percentage"].append(0.0)
    if len(linkage) >= min_obs:
        all_linkages["found_pure"].append(1)
        all_linkages["found"].append(1)
    else:
        all_linkages["found_pure"].append(0)
        all_linkages["found"].append(0)
    all_linkages["found_partial"].append(0)
    all_linkages["linked_truth"].append(truth)
    
    # Create some useful masks
    truth_class = observations[observations["truth"] == truth]["class"].values[0]
    summary_mask = (summary["class"] == truth_class)
    all_truths_mask = (all_truths["truth"] == truth)
    
    # Update the summary dataframe
    summary.loc[summary_mask, "linkages"] += 1
    summary.loc[summary_mask, "pure_linkages"] += 1
    summary.loc[summary_mask, "obs_in_pure_linkages"] += len(linkage)
    summary.loc[summary["class"] == "All", "linkages"] += 1
    summary.loc[summary["class"] == "All", "pure_linkages"] += 1
    summary.loc[summary["class"] == "All", "obs_in_pure_linkages"] += len(linkage)
    if is_complete:
        summary.loc[summary_mask, "pure_complete_linkages"] += 1
        summary.loc[summary_mask, "obs_in_pure_complete_linkages"] += len(linkage)
        summary.loc[summary["class"] == "All", "pure_complete_linkages"] += 1
        summary.loc[summary["class"] == "All", "obs_in_pure_complete_linkages"] += len(linkage)
    if is_found:
        summary.loc[summary_mask, "found_pure_linkages"] += 1
        summary.loc[summary["class"] == "All", "found_pure_linkages"] += 1
        
    # Update the all_truths dataframe
    all_truths.loc[all_truths_mask, "pure"] += 1
    all_truths.loc[all_truths_mask, "obs_in_pure"] += len(linkage)
    if is_complete:
        all_truths.loc[all_truths_mask, "pure_complete"] += 1
        all_truths.loc[all_truths_mask, "obs_in_pure_complete"] += len(linkage)
    if is_found:
        all_truths.loc[all_truths_mask, "found"] += 1
        all_truths.loc[all_truths_mask, "found_pure"] += 1

    return linkage

def createPartialLinkage(truth, 
                         observations, 
                         all_linkages, 
                         all_truths,
                         summary, 
                         contamination_percentage=20, 
                         min_obs=5, 
                         min_linkage_length=5):
    """
    Create a partial linkage: a linkage containing observations of multiple truths but 
    enough observations of the given truth for it to potentially be retrievable. 
    
    Parameters
    ----------
    truth : str
        The name / object ID of the truth
    num_correct_obs : int
        Number of observations the linkage should contain of the given
        truth.
    num_contaminated_obs : int
        Number of observations the linkage should contain of different 
        truths. 
    observations : `~pandas.DataFrame`
        A dataframe containing observations. Must have at least the following
        two columns: 1) an observation ID column and 2) a truth column.
    all_linkages : dict
        Dictionary of various linkage properties that is updated by this function.
        This dictionary is converted into a `pandas.DataFrame` by the 
        createTestData function.

    Returns
    -------
    linkage : `~numpy.ndarray` (N)
        An array of observation IDs.
    """
    # Get the maximum number of observations possible for the desired truth
    max_obs = observations[observations["truth"] == truth]["obs_id"].nunique()
    
    # Randomly select how many contaminated observations should be included
    num_contaminated_obs = np.random.choice(np.arange(1, 2, 3))
    
    # If the maximum number of observations is less than the minimum linkage
    # length, don't create a linkage
    if max_obs < min_linkage_length:
        return None
    
    # Randomly select a number of correct observations 
    num_correct_obs = np.random.choice(np.arange(min_linkage_length, max_obs + 1))
    
    # Grab correct observation IDs
    linkage_correct = np.random.choice(observations[observations["truth"] == truth]["obs_id"].values, num_correct_obs, replace=False)
    
    # Grab incorrect observation IDs
    linkage_contaminated = np.random.choice(observations[observations["truth"] != truth]["obs_id"].values, num_contaminated_obs, replace=False)
    
    # Calculate the total number of members in the linkage
    members = observations[observations["obs_id"].isin(linkage_contaminated)]["truth"].nunique() + 1
    
    # Combine the observation IDs and sort them
    linkage = np.concatenate([linkage_correct, linkage_contaminated])
    linkage.sort()
    
    # Calculate the contamination percentage, if it is higher than the maximum allowed
    # stop and do not create a linkage
    cp = 100. * num_contaminated_obs / (num_contaminated_obs + num_correct_obs)
    if cp > contamination_percentage:
        return None
    
    # If the number of correct observations is equal to or 
    # greater than min_obs we consider the object found
    found = (num_correct_obs >= min_obs)
    
    # Update the all_linkage dictionary
    all_linkages["num_obs"].append(len(linkage))
    all_linkages["num_members"].append(members)
    all_linkages["pure"].append(0)
    all_linkages["pure_complete"].append(0)
    all_linkages["partial"].append(1)
    all_linkages["mixed"].append(0)
    all_linkages["contamination_percentage"].append(cp)
    all_linkages["found_pure"].append(0)
    if found:
        all_linkages["found_partial"].append(1)
        all_linkages["found"].append(1)
    else:
        all_linkages["found_partial"].append(0)
        all_linkages["found"].append(0)
    all_linkages["linked_truth"].append(truth)
    
    # Create masks
    truth_class = observations[observations["truth"] == truth]["class"].values[0]
    summary_mask = (summary["class"] == truth_class)
    all_truths_mask = (all_truths["truth"] == truth)
    observations_mask_contaminated = observations["obs_id"].isin(linkage_contaminated)
                 
    # Update the summary dataframe
    summary.loc[summary_mask, "linkages"] += 1
    summary.loc[summary_mask, "partial_linkages"] += 1
    summary.loc[summary_mask, "obs_in_partial_linkages"] += num_correct_obs
    summary.loc[summary["class"] == "All", "linkages"] += 1
    summary.loc[summary["class"] == "All", "partial_linkages"] += 1
    summary.loc[summary["class"] == "All", "obs_in_partial_linkages"] += num_correct_obs
    summary.loc[summary["class"] == "All", "partial_contaminant_linkages"] += 1
    
    # For each truth class of the contaminant observations, update the summay dataframe
    for c in observations[observations_mask_contaminated]["class"].unique():        
        summary.loc[summary["class"] == c, "partial_contaminant_linkages"] += 1
        if c != truth_class:
            summary.loc[summary["class"] == c, "linkages"] += 1
        
        obs_contaminated = observations[observations_mask_contaminated & observations["class"].isin([c])]["obs_id"].nunique()
        summary.loc[summary["class"] == c, "obs_in_partial_contaminant_linkages"] += obs_contaminated
        summary.loc[summary["class"] == "All", "obs_in_partial_contaminant_linkages"] += obs_contaminated
    
    # Update if the object is found or not
    if found:
        summary.loc[summary_mask, "found_partial_linkages"] += 1
        summary.loc[summary["class"] == "All", "found_partial_linkages"] += 1
        all_truths.loc[all_truths_mask, "found"] += 1
        all_truths.loc[all_truths_mask, "found_partial"] += 1
    all_truths.loc[all_truths_mask, "partial"] += 1                  
                                       
    truths_occurences = observations[observations["obs_id"].isin(linkage)]["truth"].value_counts()
    for t_i, obs_i in zip(truths_occurences.index.values, truths_occurences.values):
        if t_i == truth:
            all_truths.loc[all_truths_mask, "obs_in_partial"] += obs_i
        else:
            all_truths.loc[all_truths["truth"] == t_i, "obs_in_partial_contaminant"] += obs_i
            all_truths.loc[all_truths["truth"] == t_i, "partial_contaminant"] += 1
                                       
    return linkage
    
def createMixedLinkage(observations, all_linkages, all_truths, summary, min_linkage_length=5):
    """
    Create a mixed linkage: a linkage containing observations of many truths.
    These are typically noise or spurrious linkages. 
    
    Parameters
    ----------
    num_obs : int
        Number of observations the linkage should contain.
    observations : `~pandas.DataFrame`
        A dataframe containing observations. Must have at least the following
        two columns: 1) an observation ID column and 2) a truth column.
    all_linkages : dict
        Dictionary of various linkage properties that is updated by this function.
        This dictionary is converted into a `pandas.DataFrame` by the 
        createTestData function.

    Returns
    -------
    linkage : `~numpy.ndarray` (N)
        An array of observation IDs.
    """
    max_obs = all_truths["num_obs"].max()
    num_obs = np.random.choice(np.arange(min_linkage_length, max_obs + 1))      
                                       
    truths = np.random.choice(observations["truth"].unique(), num_obs, replace=False)
    linkage = []
    for i in truths:
        obs_ids = observations[observations["truth"] == i]["obs_id"].values
        linkage.append(np.random.choice(obs_ids))
    linkage = np.array(linkage)
    linkage.sort()
    
    all_linkages["num_obs"].append(len(linkage))
    all_linkages["num_members"].append(len(np.unique(truths)))
    all_linkages["pure"].append(0)
    all_linkages["pure_complete"].append(0)
    all_linkages["partial"].append(0)
    all_linkages["mixed"].append(1)
    all_linkages["contamination_percentage"].append(np.NaN)
    all_linkages["found_pure"].append(0)
    all_linkages["found_partial"].append(0)
    all_linkages["found"].append(0)
    all_linkages["linked_truth"].append(np.NaN)
    
    truth_class = observations[observations["truth"].isin(truths)]["class"].unique()
    observations_mask = observations["obs_id"].isin(linkage)
    for tc in truth_class:
        summary_mask = (summary["class"] == tc)
        summary.loc[summary_mask, "linkages"] += 1
        summary.loc[summary_mask, "mixed_linkages"] += 1
        
        obs_class = observations[observations_mask & observations["class"].isin([tc])]["obs_id"].nunique()
        summary.loc[summary_mask, "obs_in_mixed_linkages"] += obs_class
        summary.loc[summary["class"] == "All",  "obs_in_mixed_linkages"] += obs_class
        
    summary.loc[summary["class"] == "All", "linkages"] += 1
    summary.loc[summary["class"] == "All", "mixed_linkages"] += 1
                                       
    truths_occurences = observations[observations["obs_id"].isin(linkage)]["truth"].value_counts()
    for tti, obs_i in zip(truths_occurences.index.values, truths_occurences.values):
        all_truths.loc[all_truths["truth"] == tti, "obs_in_mixed"] += obs_i
        all_truths.loc[all_truths["truth"] == tti, "mixed"] += 1
                                       
    return linkage


def createTruthClass(name, num_truths, num_obs):
    """
    Create a simple class of truths with support data products. 
    
    Parameters
    ----------
    name : str
        Class name.
    num_truths : int
        Number of unique clas members to make. 
    num_obs : list (num_truths)
        List with the number of observations to make 
        for each truth.
    
    """
    
    observations = {
        "obs_id" : [],
        "truth" : [],
        "class" : []
    }
    all_truths = {
        "truth" : [],
        "num_obs" : [],
    }
    summary = {
        "class" : [name],
        "num_members" : [num_truths],
        "num_obs" : [sum(num_obs)],
        "completeness" : [np.NaN],
        "findable" : [0],
        "found" : [0],
        "findable_found" : [0],
        "findable_missed" : [0],
        "not_findable_found" : [0],
        "not_findable_missed" : [0],
        "linkages" : [0], 
        "found_pure_linkages" : [0],
        "found_partial_linkages" : [0],
        "pure_linkages" : [0],
        "pure_complete_linkages" : [0],
        "partial_linkages" : [0],
        "partial_contaminant_linkages" : [0],
        "mixed_linkages" : [0],
        "unique_in_pure_linkages" : [0],
        "unique_in_pure_complete_linkages" : [0],
        "unique_in_pure_linkages_only" : [0],
        "unique_in_partial_linkages_only" : [0],
        "unique_in_pure_and_partial_linkages" : [0],
        "unique_in_partial_linkages" : [0],
        "unique_in_partial_contaminant_linkages" : [0],
        "unique_in_mixed_linkages" : [0],
        "obs_in_pure_linkages" : [0],
        "obs_in_pure_complete_linkages" : [0],
        "obs_in_partial_linkages" : [0],
        "obs_in_partial_contaminant_linkages" : [0],
        "obs_in_mixed_linkages" : [0]
    }

    truths = ["{}{:02d}".format(name, i) for i in range(num_truths)]
    all_truths["truth"] = truths
    for i, j in enumerate(num_obs):
        observations["truth"] += [truths[i] for _ in range(j)]
        all_truths["num_obs"].append(j)    
    observations["obs_id"] = ["obs{:05d}".format(i) for i in range(len(observations["truth"]))]
    observations["class"] = [name for _ in range(len(observations["truth"]))]
    
    observations = pd.DataFrame(observations)
    all_truths = pd.DataFrame(all_truths)
    summary = pd.DataFrame(summary)
    return observations, all_truths, summary

def createTestDataSet(min_obs, min_linkage_length, max_contamination_percentage):
    
    np.random.seed(42)

    # Create three classes of truths:
    # red: 6 truths with ranging between 5-10 observations each
    # blue: 6 truths with ranging between 5-10 observations each
    # green: 20 truths with 1 observation each
    observations_reds, all_truths_reds, summary_reds = createTruthClass("red", 6, [5, 6, 7, 8, 9, 10])
    observations_blues, all_truths_blues, summary_blues = createTruthClass("blue", 6, [5, 6, 7, 8, 9, 10])
    observations_greens, all_truths_greens, summary_greens =  createTruthClass("green", 30, [1 for i in range(30)])

    # Concatenate their dataframes into a single dataset
    observations = pd.concat([observations_reds, observations_blues, observations_greens])
    observations = observations.sample(frac=1)
    observations.reset_index(inplace=True, drop=True)
    observations["obs_id"] = ["obs{:05}".format(i) for i in range(len(observations))]

    all_truths = pd.concat([all_truths_reds, all_truths_blues, all_truths_greens])
    all_truths.sort_values(by=["num_obs", "truth"], ascending=[False, True], inplace=True)
    all_truths.reset_index(inplace=True, drop=True)

    # Add the "All" class to the summary dataframe
    summary = pd.concat([summary_reds, summary_blues, summary_greens])
    summary_all = pd.DataFrame({
        "class" : ["All"],
        "num_members" : [int(summary["num_members"].sum())],
        "num_obs" : [int(summary["num_obs"].sum())],
        "completeness" : [np.NaN],
        "findable" : [0],
        "found" : [0],
        "findable_found" : [0],
        "findable_missed" : [0],
        "not_findable_found" : [0],
        "not_findable_missed" : [0],
        "linkages" : [0], 
        "found_pure_linkages" : [0],
        "found_partial_linkages" : [0],
        "pure_linkages" : [0],
        "pure_complete_linkages" : [0],
        "partial_linkages" : [0],
        "partial_contaminant_linkages" : [0],
        "mixed_linkages" : [0],
        "unique_in_pure_linkages" : [0],
        "unique_in_pure_complete_linkages" : [0],
        "unique_in_pure_linkages_only" : [0],
        "unique_in_partial_linkages_only" : [0],
        "unique_in_pure_and_partial_linkages" : [0],
        "unique_in_partial_linkages" : [0],
        "unique_in_partial_contaminant_linkages" : [0],
        "unique_in_mixed_linkages" : [0],
        "obs_in_pure_linkages" : [0],
        "obs_in_pure_complete_linkages" : [0],
        "obs_in_partial_linkages" : [0],
        "obs_in_partial_contaminant_linkages" : [0],
        "obs_in_mixed_linkages" : [0]
    })
    
    summary = summary.append(summary_all)
    summary.sort_values(by=["num_obs", "num_members"], ascending=[False, True], inplace=True)
    summary.reset_index(inplace=True, drop=True)
    
    all_truths.loc[:, "findable"] = 0
    all_truths.loc[all_truths["num_obs"] >= min_obs, "findable"] = 1
    all_truths.loc[:, "found_pure"] = 0
    all_truths.loc[:, "found_partial"] = 0
    all_truths.loc[:, "found"] = 0
    all_truths.loc[:, "pure"] = 0
    all_truths.loc[:, "pure_complete"] = 0
    all_truths.loc[:, "partial"] = 0
    all_truths.loc[:, "partial_contaminant"] = 0
    all_truths.loc[:, "mixed"] = 0
    all_truths.loc[:, "pure"] = 0
    all_truths.loc[:, "obs_in_pure"] = 0
    all_truths.loc[:, "obs_in_pure_complete"] = 0
    all_truths.loc[:, "obs_in_partial"] = 0
    all_truths.loc[:, "obs_in_partial_contaminant"] = 0
    all_truths.loc[:, "obs_in_mixed"] = 0
    
    
    classes = {}
    for c in observations["class"].unique():
        classes[c] = observations[observations["class"] == c]["truth"].unique()
     
    
    linkage_members = {
        "linkage_id" : [],
        "obs_id" : []
    }
    
    all_linkages = {
        "linkage_id" : [],
        "num_obs" : [],
        "num_members" : [],
        "pure" : [],
        "pure_complete" : [],
        "partial" : [],
        "mixed" : [],
        "contamination_percentage" : [],
        "found_pure" : [],
        "found_partial" : [],
        "found" : [],
        "linked_truth" : [],
    }

    linkage_id_iter = 0
    for c, t in classes.items():
        
        # Add pure linkages
        for ti in np.random.choice(t, len(t)):

            linkage = createPureLinkage(
                ti, 
                observations, 
                all_linkages, 
                all_truths, 
                summary, 
                min_obs=min_obs,
                min_linkage_length=min_linkage_length)
            
            if linkage is not None:
                                       
                linkage_id = "linkage{:05d}".format(linkage_id_iter)
                linkage_members["linkage_id"] += [linkage_id for _ in range(len(linkage))]
                linkage_members["obs_id"] += list(linkage)
                all_linkages["linkage_id"].append(linkage_id)

                linkage_id_iter += 1
            

        # Add partial linkages
        for ti in np.random.choice(t, len(t) - 1, replace=False):
            
            linkage = createPartialLinkage(
                ti, 
                observations, 
                all_linkages, 
                all_truths,
                summary, 
                contamination_percentage=max_contamination_percentage, 
                min_obs=min_obs, 
                min_linkage_length=min_linkage_length)
                                       
            if linkage is not None:
                                       
                linkage_id = "linkage{:05d}".format(linkage_id_iter)
                linkage_members["linkage_id"] += [linkage_id for _ in range(len(linkage))]
                linkage_members["obs_id"] += list(linkage)
                all_linkages["linkage_id"].append(linkage_id)
                                       
                linkage_id_iter += 1
                
    
    # Create 10 mixed linkages           
    for i in range(10):

        linkage = createMixedLinkage(
            observations, 
            all_linkages, 
            all_truths, 
            summary, 
            min_linkage_length=min_linkage_length
        )
        
        linkage_id = "linkage{:05d}".format(linkage_id_iter)
        linkage_members["linkage_id"] += [linkage_id for _ in range(len(linkage))]
        linkage_members["obs_id"] += list(linkage)
        all_linkages["linkage_id"].append(linkage_id)

        linkage_id_iter += 1
    
    
    linkage_members = pd.DataFrame(linkage_members)
    all_linkages = pd.DataFrame(all_linkages)
    all_linkages["linked_truth"] = all_linkages["linked_truth"].astype(str)
    
    
    for c, t in classes.items():
        all_truths_mask = (all_truths["truth"].isin(t))
        summary_mask = (summary["class"] == c)

        # Number of unique truths found
        found = all_truths[all_truths_mask & (all_truths["found"] >= 1)]["truth"].nunique()

        # Number of unique truths findable
        findable = all_truths[all_truths_mask]["findable"].sum()

        # Number of findable truths found
        findable_found = len(
            all_truths[
                all_truths_mask 
                & (all_truths["findable"] == 1) 
                & (all_truths["found"] >= 1)
            ]
        )

        # Number of not findable truths found
        not_findable_found = len(
            all_truths[
                all_truths_mask 
                & (all_truths["findable"] == 0) 
                & (all_truths["found"] >= 1)
            ]
        )

        # Number of findable truths not found (missed)
        findable_missed = len(
            all_truths[
                all_truths_mask 
                & (all_truths["findable"] == 1) 
                & (all_truths["found"] == 0)
            ]
        )

        # Number of not findable truths found 
        not_findable_found = len(
            all_truths[
                all_truths_mask 
                & (all_truths["findable"] == 0) 
                & (all_truths["found"] >= 1)
            ]
        )

        # Number of not findable truths not found 
        not_findable_missed = len(
            all_truths[
                all_truths_mask 
                & (all_truths["findable"] == 0) 
                & (all_truths["found"] == 0)
            ]
        )

        summary.loc[summary_mask, "found"] = found
        summary.loc[summary_mask, "findable"] = findable
        summary.loc[summary_mask, "findable_found"] = findable_found
        summary.loc[summary_mask, "findable_missed"] = findable_missed
        summary.loc[summary_mask, "not_findable_found"] = not_findable_found
        summary.loc[summary_mask, "not_findable_missed"] = not_findable_missed
    
    # Count which objects appear in what type of linkage
    pure = []
    pure_complete = []
    partial_contaminant = []
    partial = []
    mixed = []

    for linkage_id in all_linkages["linkage_id"].unique():
        all_linkages_mask = (all_linkages["linkage_id"] == linkage_id)
        linkage_members_mask = (linkage_members["linkage_id"] == linkage_id)
        
        if all_linkages[all_linkages_mask]["pure"].values[0] == 1:
            pure.append(all_linkages[all_linkages_mask]["linked_truth"].values[0])

        if all_linkages[all_linkages_mask]["pure_complete"].values[0] == 1:
            pure_complete.append(all_linkages[all_linkages_mask]["linked_truth"].values[0])

        if all_linkages[all_linkages_mask]["partial"].values[0] == 1:
            linked_truth = all_linkages[all_linkages_mask]["linked_truth"].values[0]
            partial.append(linked_truth)
            
            obs_ids = linkage_members[linkage_members_mask]["obs_id"].values
            obs_contaminant_mask = (observations["obs_id"].isin(obs_ids)) & (~observations["truth"].isin([linked_truth]))
            partial_contaminant += list(observations[obs_contaminant_mask]["truth"].values)

        if all_linkages[all_linkages_mask]["mixed"].values[0] == 1:
            obs_ids = linkage_members[linkage_members["linkage_id"] == linkage_id]["obs_id"].values
            truths = observations[observations["obs_id"].isin(obs_ids)]["truth"].unique()
            mixed += list(truths)

    pure = np.unique(pure)
    pure_complete = np.unique(pure_complete)
    partial = np.unique(partial)
    partial_contaminant = np.unique(partial_contaminant)
    mixed = np.unique(mixed)

    # Update summary with linkage - truth break down
    for c, t in classes.items():
        summary_mask = (summary["class"] == c)
        pure_set = set(pure).intersection(set(t))
        pure_complete_set = set(pure_complete).intersection(set(t))
        partial_set = set(partial).intersection(set(t))
        partial_contaminant_set = set(partial_contaminant).intersection(set(t))
        mixed_set = set(mixed).intersection(set(t))

        unique_in_pure = pure_set
        unique_in_pure_complete = pure_complete_set
        unique_in_partial = partial_set
        unique_in_partial_contaminant = partial_contaminant_set
        unique_in_pure_and_partial = list(pure_set.intersection(partial_set))
        unique_in_pure_only = unique_in_pure - unique_in_partial
        unique_in_partial_only = unique_in_partial - unique_in_pure
        unique_in_mixed = mixed_set

        summary.loc[summary_mask, "unique_in_pure_linkages"] = len(unique_in_pure)
        summary.loc[summary_mask, "unique_in_pure_complete_linkages"] = len(unique_in_pure_complete)
        summary.loc[summary_mask, "unique_in_partial_linkages"] = len(unique_in_partial)
        summary.loc[summary_mask, "unique_in_partial_contaminant_linkages"] = len(unique_in_partial_contaminant)
        summary.loc[summary_mask, "unique_in_pure_and_partial_linkages"] = len(unique_in_pure_and_partial)
        summary.loc[summary_mask, "unique_in_pure_linkages_only"] = len(unique_in_pure_only)
        summary.loc[summary_mask, "unique_in_partial_linkages_only"] = len(unique_in_partial_only)
        summary.loc[summary_mask, "unique_in_mixed_linkages"] = len(unique_in_mixed)

    # Update total number of truths found, findable, missed, etc.. 
    summary_mask = (summary["class"] == "All")
    cols = [
        "findable", 
        "found", 
        "findable_found", 
        "findable_missed", 
        "not_findable_found", 
        "not_findable_missed",
        "unique_in_pure_linkages",
        "unique_in_pure_complete_linkages",
        "unique_in_partial_linkages",
        "unique_in_partial_contaminant_linkages",
        "unique_in_pure_and_partial_linkages",
        "unique_in_pure_linkages_only",
        "unique_in_partial_linkages_only",
        "unique_in_mixed_linkages",
    ]
    for col in cols:
        summary.loc[summary_mask, col] = summary.loc[~summary_mask, col].sum()
    
    summary["completeness"] = 100 * summary["findable_found"] / summary["findable"]
    
    summary.sort_values(
        by=["num_obs"], 
        ascending=False,
        inplace=True
    )

    return observations, all_truths, linkage_members, all_linkages, summary

