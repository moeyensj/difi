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

    Analyzes a DataFrame containing observations. These observations need at least two columns:
    i) the observation ID column
    ii) the truth column

    We assume there to be three kinds of truths:
    i) known truth: an observation that has a known source
    ii) unknown truth: an observation that has an unknown source (this could be several unknown sources)
    iii) false positive truth: a false positive observation 

    By definiton, these three kinds of labels or truths can be separated into what is assumed known
    and assumed unknown. Known truths are sources that can be used to test algorithms. Unknown truths is the 
    default source for observations, it is up to the linking algorithm to identify and associate. False positive truths
    are useful when testing how linking algorithms work in the presence of noise. 

    This function assumes that only known truths can be findable, a findable known truth is a truth 
    that has at least the minObs number of observations.
    
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
    num_obs_per_object = observations[columnMapping["truth"]].value_counts().values
    num_obs_descending = observations[columnMapping["truth"]].value_counts().index.values
    findable = num_obs_descending[np.where(num_obs_per_object >= minObs)[0]]
    
    # Populate allTruths DataFrame
    allTruths = pd.DataFrame(columns=[
        columnMapping["truth"], 
        "num_obs", 
        "findable"])
    
    allTruths[columnMapping["truth"]] = num_obs_descending
    allTruths["num_obs"] = num_obs_per_object
    allTruths.loc[(allTruths[columnMapping["truth"]].isin(findable)) & (~allTruths[columnMapping["truth"]].isin(unknownIDs + falsePositiveIDs)), "findable"] = 1
    allTruths.loc[allTruths["findable"] != 1, ["findable"]] = 0
    num_findable = len(allTruths[allTruths["findable"] == 1])
    percent_known = num_truth_obs / len(observations) * 100.0
    percent_unknown = num_unknown_obs / len(observations) * 100.0
    percent_false_positive = num_fp_obs / len(observations) * 100.0

    # Prepare summary DataFrame
    summary = pd.DataFrame(index=[0]) 
    summary["num_unique_truths"] = num_truths, 
    summary["num_unique_known_truths"] = num_unique_truths,
    summary["num_unique_known_truths_findable"] = num_findable,
    summary["num_known_truth_observations"] = num_truth_obs
    summary["num_unknown_truth_observations"] = num_unknown_obs
    summary["num_false_positive_observations"] = num_fp_obs
    summary["percent_known_truth_observations"] = percent_known
    summary["percent_unknown_truth_observations"] = percent_unknown
    summary["percent_false_positive_observations"] = percent_false_positive
    
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
