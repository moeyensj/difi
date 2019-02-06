import os
import pandas as pd 
from pandas.util.testing import assert_frame_equal

from difi import analyzeLinkages

def test_analyzeLinkages():
    # Load sample input
    
    linkageMembers = pd.read_csv(os.path.join(os.path.dirname(__file__), "linkageMembers.txt"), sep=" ", index_col=False)
    observations = pd.read_csv(os.path.join(os.path.dirname(__file__), "observations.txt"), sep=" ", index_col=False)
    
    # Load solution
    allLinkages_solution = pd.read_csv(os.path.join(os.path.dirname(__file__), "allLinkages_solution.txt"), sep=" ", index_col=False)
    allTruths_solution = pd.read_csv(os.path.join(os.path.dirname(__file__), "allTruths_solution.txt"), sep=" ", index_col=False)
    
    allLinkages_test, allTruths_test = analyzeLinkages(observations, 
                                                       linkageMembers,
                                                       minObs=5, 
                                                       contaminationThreshold=0.2)
    
    # Re-arange columns in-case order is changed (python 3.5 and earlier)
    allLinkages_test = allLinkages_test[["linkage_id", 
                                         "num_members", 
                                         "num_obs", 
                                         "pure", 
                                         "partial", 
                                         "false", 
                                         "contamination", 
                                         "linked_truth"]]
    allTruths_test = allTruths_test[["truth", 
                                     "found_pure", 
                                     "found_partial", 
                                     "found"]] 
    
    # Assert dataframes are equal
    assert_frame_equal(allLinkages_test, allLinkages_solution)
    assert_frame_equal(allTruths_test, allTruths_solution)

    