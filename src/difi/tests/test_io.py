import os

import pandas as pd
from pandas.testing import assert_frame_equal

from ..io import readLinkagesByLineFile


def test_readLinkagesByLineFile():
    # Test readLinkagesByLineFile using test data set

    linkages_by_line_file = os.path.join(os.path.dirname(__file__), "linkagesByLine.txt")
    linkage_members = readLinkagesByLineFile(linkages_by_line_file)

    # Create expected linkage_members DataFrame
    expected_linkage_members = {
        "linkage_id": [],
        "obs_id": [],
    }
    linkages = {
        "1": [
            "obs00002",
            "obs00017",
            "obs00068",
            "obs00099",
            "obs00102",
        ],
        "2": ["obs00000", "obs00008", "obs00013", "obs00024", "obs00049", "obs00051"],
        "3": [
            "obs00007",
            "obs00039",
            "obs00046",
            "obs00056",
        ],
    }
    for linkage_id, obs_ids in linkages.items():
        for obs_id in obs_ids:
            expected_linkage_members["linkage_id"].append(linkage_id)
            expected_linkage_members["obs_id"].append(obs_id)
    expected_linkage_members = pd.DataFrame(expected_linkage_members)

    assert_frame_equal(linkage_members, expected_linkage_members)
    return
