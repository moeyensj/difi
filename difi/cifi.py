import time
import numpy as np
import pandas as pd

__all__ = ["analyzeObservations"]

def analyzeObservations(observations,
                        minObs=5, 
                        unknownIDs=[],
                        falsePositiveIDs=[],
                        verbose=True,
                        columnMapping={"linkage_id": "linkage_id",
                                       "obs_id": "obs_id",
                                       "truth": "truth"}):
    """
    Can I Find It?
    
    Parameters
    ----------
    observations : `~pandas.DataFrame`
        Pandas DataFrame with at least two columns: observation IDs and the truth values
        (the object to which the observation belongs to).
    minObs : int, optional
        The minimum number of observations required for a truth to be considered
        findable. 
        [Default = 5]
    unknownIDs : list, optional
        Values in the name column for unknown observations.
        [Default = []]
    falsePositiveIDs : list, optional
        Names of false positive IDs.
        [Default = []]
    verbose : bool, optional
        Print progress statements? 
        [Default = True]
    columnMapping : dict, optional
        Column name mapping of observations to internally used column names. 
    
    Returns
    -------
    allTruths : `~pandas.DataFrame`
        Object summary DataFrame.
    summary : `~pandas.DataFrame`
        Overall summary DataFrame. 
    """
    time_start = time.time()
    if verbose == True:
        print("Analyzing observations...")

    # Raise error if there are no observations
    if len(observations) == 0: 
        raise ValueError("There are no observations in the observations DataFrame!")
    
    # Count number of false positive observations, real observations, unknown observations, the number of unique truths, and those 
    # that should be findable
    num_fp_obs = len(observations[observations[columnMapping["truth"]].isin(falsePositiveIDs)])
    num_unknown_obs = len(observations[observations[columnMapping["truth"]].isin(unknownIDs)])
    num_truth_obs = len(observations[~observations[columnMapping["truth"]].isin(unknownIDs + falsePositiveIDs)])
    num_truths = observations[columnMapping["truth"]].nunique()
    unique_truths = observations[~observations[columnMapping["truth"]].isin(unknownIDs + falsePositiveIDs)][columnMapping["truth"]].unique()
    num_unique_truths = len(unique_truths)
    num_obs_per_object = observations[~observations[columnMapping["truth"]].isin(unknownIDs + falsePositiveIDs)][columnMapping["truth"]].value_counts().values
    truths_num_obs_descending = observations[~observations[columnMapping["truth"]].isin(unknownIDs + falsePositiveIDs)][columnMapping["truth"]].value_counts().index.values
    findable = truths_num_obs_descending[np.where(num_obs_per_object >= minObs)[0]]
    
    # Populate allTruths DataFrame
    allTruths = pd.DataFrame(columns=[
        columnMapping["truth"], 
        "num_obs", 
        "findable"])
    
    allTruths[columnMapping["truth"]] = truths_num_obs_descending
    allTruths["num_obs"] = num_obs_per_object
    allTruths.loc[allTruths[columnMapping["truth"]].isin(findable), "findable"] = 1
    allTruths.loc[allTruths["findable"] != 1, ["findable"]] = 0
    num_findable = len(allTruths[allTruths["findable"] == 1])
    percent_known = num_truth_obs / len(observations) * 100
    percent_unknown = num_unknown_obs / len(observations) * 100
    percent_false_positive = num_fp_obs / len(observations) * 100
    
    # Prepare summary DataFrame
    summary = pd.DataFrame({
        "num_unique_truths" : num_truths, 
        "num_unique_known_truths" : num_unique_truths,
        "num_unique_known_truths_findable" : num_findable,
        "num_known_truth_observations": num_truth_obs,
        "num_unknown_truth_observations": num_unknown_obs,
        "num_false_positive_observations": num_fp_obs,
        "percent_known_truth_observations": percent_known,
        "percent_unknown_truth_observations": percent_unknown,
        "percent_false_positive_observations": percent_false_positive}, index=[0]) 
    
    time_end = time.time()
    if verbose == True:
        print("Known truth observations: {}".format(num_truth_obs))
        print("Unknown truth observations: {}".format(num_unknown_obs))
        print("False positive observations: {}".format(num_fp_obs))
        print("Percent known truth observations (%): {:1.3f}".format(percent_known))
        print("Percent unknown truth observations (%): {:1.3f}".format(percent_unknown))
        print("Percent false positive observations (%): {:1.3f}".format(percent_false_positive))
        print("Unique truths: {}".format(num_truths))
        print("Unique known truths : {}".format(num_unique_truths))
        print("Unique known truths with at least {} detections: {}".format(minObs, num_findable))
        print("") 
        print("Total time in seconds: {}".format(time_end - time_start))
        print("")
        
    return allTruths, summary
