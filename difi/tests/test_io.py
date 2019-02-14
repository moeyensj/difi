import os
import numpy as np
import pandas as pd 
from pandas.util.testing import assert_frame_equal

from difi import readLinkagesByLineFile

def test_readLinkagesByLineFile_fromFile():
    # Set sample input
    linkagesFile = os.path.join(os.path.dirname(__file__), "linkagesByLine.txt")
    
    # Load solution
    linkageMembers_solution = pd.read_csv(os.path.join(os.path.dirname(__file__), "linkageMembers_solution.txt"), sep=" ", index_col=False, dtype={"obs_id" : str, "linkage_id": np.int64})
    
    linkageMembers_test = readLinkagesByLineFile(linkagesFile,
                                                 readCSVKwargs={"header": None},
                                                 linkageIDStart=1,
                                                 columnMapping={"obs_id": "obs_id",
                                                                "linkage_id" : "linkage_id"})
    
    # Re-arange columns in case order is changed (python 3.5 and earlier)
    linkageMembers_test = linkageMembers_test[["linkage_id", 
                                               "obs_id"]]
    
    # Assert dataframes are equal
    assert_frame_equal(linkageMembers_test, linkageMembers_solution)
