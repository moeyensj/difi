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
    
    # Assert dataframes are equal
    assert_frame_equal(allLinkages_test, allLinkages_solution)
    assert_frame_equal(allTruths_test, allTruths_solution)

    