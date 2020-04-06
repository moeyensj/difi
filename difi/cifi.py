import time
import numpy as np
import pandas as pd

from .utils import _checkColumnTypes
from .utils import _classHandler

__all__ = ["analyzeObservations"]

def analyzeObservations(observations,
                        minObs=5, 
                        classes=None,
                        verbose=True,
                        columnMapping={"linkage_id": "linkage_id",
                                       "obs_id": "obs_id",
                                       "truth": "truth"}):
    """
    Can I Find It?

    Analyzes a DataFrame containing observations. These observations need at least two columns:
    i) the observation ID column
    ii) the truth column
    
    Parameters
    ----------
    observations : `~pandas.DataFrame`
        Pandas DataFrame with at least two columns: observation IDs and the truth values
        (the object to which the observation belongs to).
    minObs : int, optional
        The minimum number of observations required for a truth to be considered
        findable. 
        [Default = 5]
    classes : {dict, str, None}
        Analyze observations for truths grouped in different classes. 
        str : Name of the column in the dataframe which identifies 
            the class of each truth.
        dict : A dictionary with class names as keys and a list of unique 
            truths belonging to each class as values.
        None : If there are no classes of truths.
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
        
    Raises
    ------
    TypeError : If the truth column in observations does not have type "Object"
    """
    time_start = time.time()
    if verbose == True:
        print("Analyzing observations...")

    # Raise error if there are no observations
    if len(observations) == 0: 
        raise ValueError("There are no observations in the observations DataFrame!")
        
    # Check column types
    _checkColumnTypes(observations, ["truth"], columnMapping)
    
    num_observations_list = []
    num_truths_list = []
    num_findable_list = []
        
    # Populate allTruths DataFrame
    allTruths = pd.DataFrame(columns=[
        columnMapping["truth"], 
        "num_obs", 
        "findable"])
    
    num_obs_per_object = observations[columnMapping["truth"]].value_counts().values
    num_obs_descending = observations[columnMapping["truth"]].value_counts().index.values
    findable = num_obs_descending[np.where(num_obs_per_object >= minObs)[0]]
    allTruths[columnMapping["truth"]] = num_obs_descending
    allTruths["num_obs"] = num_obs_per_object
    allTruths.loc[(allTruths[columnMapping["truth"]].isin(findable)), "findable"] = 1
    allTruths.loc[allTruths["findable"] != 1, ["findable"]] = 0
    
    class_list, truths_list = _classHandler(classes, observations, columnMapping)

    for c, v in zip(class_list, truths_list):
        
        num_obs = len(observations[observations[columnMapping["truth"]].isin(v)])
        unique_truths = observations[observations[columnMapping["truth"]].isin(v)][columnMapping["truth"]].unique()
        num_unique_truths = len(unique_truths)
        findable = allTruths[allTruths[columnMapping["truth"]].isin(v)]["findable"].sum()
        
        num_observations_list.append(num_obs)
        num_truths_list.append(num_unique_truths)
        num_findable_list.append(findable)

    # Prepare summary DataFrame
    summary = pd.DataFrame({
        "class" : class_list,
        "num_obs" : num_observations_list,
        "num_truths" : num_truths_list,
        "findable" : num_findable_list
    })
    summary.sort_values(by="class", inplace=True)
    summary.reset_index(inplace=True, drop=True)
    
    time_end = time.time()
    return allTruths, summary
