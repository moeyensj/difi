import numpy as np
import pandas as pd

__all__ = ["readLinkagesByLineFile"]

def readLinkagesByLineFile(linkagesFile, 
                           readCSVKwargs={"header": None}, 
                           linkageIDStart=1, 
                           columnMapping={"obs_id": "obs_id", 
                                          "linkage_id" : "linkage_id"}):    
    """
    Reads a file that contains linkages where each linkage is written in terms of its
    observations line by line. 
    
    Example:
    137541512 137543165 137615070 137620728 138216303 138216866 138221227 
    137541512 137543165 137615070 137620728 138216303 138216866 138221227 144513728 144533645 
    137541512 137543165 137615070 137620728 138216303 138216866 138221227 144513728 144533645 146991832 147084549 
    137541512 137543165 137615070 137620728 138216303 138216866 138221227 144514371 144534274 
    137541512 137543165 137615070 137620728 142747928 142763154 
    137541512 137543165 137615070 137620728 142748009 142763229 
    137541512 137543165 137615070 137620728 142748009 142763229 144513839 144533746 
    137541512 137543165 137615070 137620728 142748120 142763338 
    137541512 137543165 137615070 137620728 142748305 142763529 
    137541512 137543165 137615070 137620728 142748337 142763570 
    
    Parameters
    ----------
    linkagesFile : str
        Path the linkages file that needs to be converted. 
    readCSVKwargs : dict, optional
        The kwargs used by `pandas.read_csv` to read the linkages file. 
        [Default = {'header' : None}]
    linkageIDStart = 1
        Number at which to start the linkage ID count. 
        [Default = 1]
    columnMapping : dict, optional
        The mapping of columns in linkagesFile to internally used names. 
        Needs the following: "linkage_id" : ..., "obs_id" : ... .
        [Default = {'obs_id' : 'obs_id',
                    'linkage_id' : 'linkage_id'}]
    Returns
    -------
    linkageMembers : `~pandas.DataFrame`
        DataFrame with two columns: the linkage ID and a second column with one row
        per observation ID.
    """                                                                                                   
    # Read initial file
    linkages = pd.read_csv(linkagesFile, names=[columnMapping["obs_id"]], **readCSVKwargs)
    
    # Make array of linkage IDs 
    linkage_ids = np.arange(linkageIDStart, linkageIDStart + len(linkages), dtype=int)

    # Split each linkage into its different observation IDs
    linkage_list = linkages[columnMapping["obs_id"]].str.split(" ").tolist()

    # Build initial DataFrame
    linkageMembers = pd.DataFrame(pd.DataFrame(linkage_list, index=linkage_ids).stack(), columns=[columnMapping["obs_id"]])

    # Reset index 
    linkageMembers.reset_index(1, drop=True, inplace=True)

    # Make linkage_id its own column
    linkageMembers[columnMapping["linkage_id"]] = linkageMembers.index

    # Re-arrange column order 
    linkageMembers = linkageMembers[[columnMapping["linkage_id"], columnMapping["obs_id"]]]

    # Not all linkages have the same number of detections, empty detections needs to be dropped
    linkageMembers[columnMapping["obs_id"]].replace("", np.nan, inplace=True)
    linkageMembers.dropna(inplace=True)
    linkageMembers.reset_index(drop=True, inplace=True)

    return linkageMembers