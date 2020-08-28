import os
import numpy as np
import pandas as pd 
from pandas.testing import assert_frame_equal

from ..io import readLinkagesByLineFile
from .create_test_data import createTestDataSet

def test_readLinkagesByLineFile():
    ### Test readLinkagesByLineFile using test data set
    observations_test, all_truths_test, linkage_members_test, all_linkages_test, summary_test = createTestDataSet(5, 5, 30)

    linkages_by_line = linkage_members_test.groupby("linkage_id")["obs_id"].apply(np.array).to_frame()
    linkages_by_line_file = os.path.join(os.path.dirname(__file__), "linkagesByLine.txt")
    with open(linkages_by_line_file, "w") as f: 
        for linkage in linkages_by_line["obs_id"].values:
            f.write(" ".join(linkage) + "\n")

   
    linkage_members = readLinkagesByLineFile(linkages_by_line_file)

    assert_frame_equal(linkage_members[["obs_id"]], linkage_members_test[["obs_id"]])

    return