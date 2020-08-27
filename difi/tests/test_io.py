import os
import numpy as np
import pandas as pd 
from pandas.testing import assert_frame_equal

from ..io import readLinkagesByLineFile
from .create_test_data import createTestDataSet

def test_readLinkagesByLineFile():
    observations_test, all_truths_test, linkage_members_test, all_linkages_test, summary_test = createTestDataSet(5, 5, 30)

    linkages_by_line = linkage_members_test.groupby("linkage_id")["obs_id"].apply(np.array).to_frame()
    with open("linkagesByLine.txt", "w") as f: 
        for linkage in linkages_by_line["obs_id"].values:
            f.write(" ".join(linkage) + "\n")

   
    linkage_members = readLinkagesByLineFile("linkagesByLine.txt")

    assert_frame_equal(linkage_members[["obs_id"]], linkage_members_test[["obs_id"]])

    return