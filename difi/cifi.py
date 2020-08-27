import time
import numpy as np
import pandas as pd

from .utils import _checkColumnTypes
from .utils import _classHandler

__all__ = ["analyzeObservations"]

def analyzeObservations(observations,
                        min_obs=5, 
                        classes=None,
                        column_mapping={"linkage_id": "linkage_id",
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
    min_obs : int, optional
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
    column_mapping : dict, optional
        Column name mapping of observations to internally used column names. 
    
    Returns
    -------
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
            "findable" : int
                Number of truths deemed findable (all_truths must be passed to this 
                function with a findable column)
        
    Raises
    ------
    TypeError : If the truth column in observations does not have type "Object"
    """
    truth_col = column_mapping["truth"]

    # Raise error if there are no observations
    if len(observations) == 0: 
        raise ValueError("There are no observations in the observations DataFrame!")
        
    # Check column types
    _checkColumnTypes(observations, ["truth"], column_mapping)
    
    num_observations_list = []
    num_truths_list = []
    num_findable_list = []
        
    # Populate all_truths DataFrame
    dtypes = np.dtype([
        (truth_col, str),
        ("num_obs", int),
        ("findable", int)])
    data = np.empty(0, dtype=dtypes)
    all_truths = pd.DataFrame(data)
    
    num_obs_per_object = observations[truth_col].value_counts().values
    num_obs_descending = observations[truth_col].value_counts().index.values
    findable = num_obs_descending[np.where(num_obs_per_object >= min_obs)[0]]
    all_truths[truth_col] = num_obs_descending
    all_truths["num_obs"] = num_obs_per_object
    all_truths.loc[:, "findable"] = 0
    all_truths.loc[(all_truths[truth_col].isin(findable)), "findable"] = 1
    
    all_truths["findable"] = all_truths["findable"].astype(int)
    all_truths.sort_values(
        by=["num_obs", truth_col], 
        ascending=[False, True], 
        inplace=True
    )
    all_truths.reset_index(
        inplace=True, 
        drop=True
    )
    
    class_list, truths_list = _classHandler(classes, observations, column_mapping)

    for c, v in zip(class_list, truths_list):
        
        num_obs = len(observations[observations[truth_col].isin(v)])
        unique_truths = observations[observations[truth_col].isin(v)][truth_col].unique()
        num_unique_truths = len(unique_truths)
        findable = int(all_truths[all_truths[truth_col].isin(v)]["findable"].sum())
        
        num_observations_list.append(num_obs)
        num_truths_list.append(num_unique_truths)
        num_findable_list.append(findable)

    # Prepare summary DataFrame
    summary = pd.DataFrame({
        "class" : class_list,
        "num_members" : num_truths_list,
        "num_obs" : num_observations_list,
        "findable" : num_findable_list
    })
    summary.sort_values(by=["num_obs", "class"], ascending=False, inplace=True)
    summary.reset_index(inplace=True, drop=True)
    
    return all_truths, summary
