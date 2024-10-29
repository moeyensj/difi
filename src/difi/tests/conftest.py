import os

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def test_observations():
    """
    Create a test data set of observations. These observations were taken
    a subset of the data used for the tutorial. The observations are:

    obs_id,time,ra,dec,night_id,truth                  # object, tracklet?
    ------------------------------------------------
    obs_000000,58366.402,348.24238,5.44284,612,23636   # 1, singleton
    obs_000001,58368.294,347.76604,5.28888,614,23636   # 1, tracklet 01
    obs_000002,58368.360,347.74873,5.28336,614,23636   # 1, tracklet 01
    obs_000003,58371.312,347.00285,5.02589,617,23636   # 1, singleton
    obs_000004,58374.276,346.25985,4.74993,620,23636   # 1, tracklet 02
    obs_000005,58374.318,346.24901,4.74593,620,23636   # 1, tracklet 02
    obs_000006,58366.338,353.88551,1.57526,612,58177   # 2, tracklet 01
    obs_000007,58366.402,353.86922,1.57478,612,58177   # 2, tracklet 01
    obs_000008,58369.324,353.14697,1.54406,615,58177   # 2, tracklet 02
    obs_000009,58369.361,353.13724,1.54357,615,58177   # 2, tracklet 02
    obs_000010,58372.288,352.39237,1.49640,618,58177   # 2, tracklet 03
    obs_000011,58372.368,352.37078,1.49490,618,58177   # 2, tracklet 03
    obs_000012,58375.310,351.61052,1.43343,621,58177   # 2, tracklet 04
    obs_000013,58375.337,351.60326,1.43283,621,58177   # 2, tracklet 04
    obs_000014,58378.319,350.83215,1.35940,624,58177   # 2, tracklet 05
    obs_000015,58378.389,350.81329,1.35757,624,58177   # 2, tracklet 05
    obs_000016,58366.370,358.60072,8.22694,612,82134   # 3, tracklet 01
    obs_000017,58366.371,358.60060,8.22693,612,82134   # 3, tracklet 01
    obs_000018,58366.400,358.59361,8.22688,612,82134   # 3, tracklet 01
    obs_000019,58366.401,358.59347,8.22688,612,82134   # 3, tracklet 01
    obs_000020,58368.351,358.14871,8.21534,614,82134   # 3, singleton
    obs_000021,58369.326,357.92042,8.20361,615,82134   # 3, tracklet 02
    obs_000022,58369.359,357.91196,8.20314,615,82134   # 3, tracklet 02
    obs_000023,58369.360,357.91173,8.20313,615,82134   # 3, tracklet 02
    obs_000024,58372.287,357.20594,8.14448,618,82134   # 3, tracklet 04
    obs_000025,58372.369,357.18470,8.14241,618,82134   # 3, tracklet 04
    obs_000026,58375.310,356.45433,8.05016,621,82134   # 3, tracklet 05
    obs_000027,58375.358,356.44137,8.04837,621,82134   # 3, tracklet 05
    obs_000028,58378.318,355.69765,7.92566,624,82134   # 3, tracklet 06
    obs_000029,58378.387,355.67936,7.92248,624,82134   # 3, tracklet 06

    Object:
    23636: 7 observations
    58177: 15 observations
    82134: 15 observations
    """
    observations = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "test_observations.csv"),
        index_col=False,
        dtype={
            "obs_id": str,
            "object_id": str,
        },
    )
    return observations


@pytest.fixture
def test_linkages(test_observations):
    """
    Create a test data set of linkages.

    linkage_id,obs_id,object_id
    pure_complete_23636,obs_000000,23636
    pure_complete_23636,obs_000001,23636
    pure_complete_23636,obs_000002,23636
    pure_complete_23636,obs_000003,23636
    pure_complete_23636,obs_000004,23636
    pure_complete_23636,obs_000005,23636
    pure_complete_58177,obs_000006,58177
    pure_complete_58177,obs_000007,58177
    pure_complete_58177,obs_000008,58177
    pure_complete_58177,obs_000009,58177
    pure_complete_58177,obs_000010,58177
    pure_complete_58177,obs_000011,58177
    pure_complete_58177,obs_000012,58177
    pure_complete_58177,obs_000013,58177
    pure_complete_58177,obs_000014,58177
    pure_complete_58177,obs_000015,58177
    pure_complete_82134,obs_000016,82134
    pure_complete_82134,obs_000017,82134
    pure_complete_82134,obs_000018,82134
    pure_complete_82134,obs_000019,82134
    pure_complete_82134,obs_000020,82134
    pure_complete_82134,obs_000021,82134
    pure_complete_82134,obs_000022,82134
    pure_complete_82134,obs_000023,82134
    pure_complete_82134,obs_000024,82134
    pure_complete_82134,obs_000025,82134
    pure_complete_82134,obs_000026,82134
    pure_complete_82134,obs_000027,82134
    pure_complete_82134,obs_000028,82134
    pure_complete_82134,obs_000029,82134
    pure_23636,obs_000002,23636
    pure_23636,obs_000003,23636
    pure_23636,obs_000004,23636
    pure_23636,obs_000005,23636
    pure_58177,obs_000008,58177
    pure_58177,obs_000009,58177
    pure_58177,obs_000010,58177
    pure_58177,obs_000011,58177
    pure_58177,obs_000012,58177
    pure_58177,obs_000013,58177
    pure_58177,obs_000014,58177
    pure_58177,obs_000015,58177
    pure_82134,obs_000018,82134
    pure_82134,obs_000019,82134
    pure_82134,obs_000020,82134
    pure_82134,obs_000021,82134
    pure_82134,obs_000022,82134
    pure_82134,obs_000023,82134
    pure_82134,obs_000024,82134
    pure_82134,obs_000025,82134
    pure_82134,obs_000026,82134
    pure_82134,obs_000027,82134
    pure_82134,obs_000028,82134
    pure_82134,obs_000029,82134
    partial_23636,obs_000000,23636
    partial_23636,obs_000002,23636
    partial_23636,obs_000003,23636
    partial_23636,obs_000004,23636
    partial_23636,obs_000005,23636
    partial_23636,obs_000008,58177
    partial_23636,obs_000026,82134
    partial_58177,obs_000006,58177
    partial_58177,obs_000010,58177
    partial_58177,obs_000011,58177
    partial_58177,obs_000012,58177
    partial_58177,obs_000015,58177
    partial_58177,obs_000027,82134
    partial_58177,obs_000028,82134
    partial_82134,obs_000004,23636
    partial_82134,obs_000012,58177
    partial_82134,obs_000017,82134
    partial_82134,obs_000020,82134
    partial_82134,obs_000022,82134
    partial_82134,obs_000025,82134
    partial_82134,obs_000026,82134
    mixed_0,obs_000000,23636
    mixed_0,obs_000005,23636
    mixed_0,obs_000009,58177
    mixed_0,obs_000015,58177
    mixed_0,obs_000021,82134
    mixed_0,obs_000024,82134
    mixed_0,obs_000027,82134
    mixed_1,obs_000003,23636
    mixed_1,obs_000006,58177
    mixed_1,obs_000008,58177
    mixed_1,obs_000020,82134
    mixed_1,obs_000023,82134
    mixed_1,obs_000027,82134
    mixed_1,obs_000028,82134
    mixed_2,obs_000002,23636
    mixed_2,obs_000005,23636
    mixed_2,obs_000012,58177
    mixed_2,obs_000013,58177
    mixed_2,obs_000015,58177
    mixed_2,obs_000016,82134
    mixed_2,obs_000025,82134

    Returns
    -------
    linkage_members : `~pandas.DataFrame`
        DataFrame containing the observation IDs and linkage IDs for
        each linkage in (linkage_id, obs_id) pairs.
    all_linkages_expected: `~pandas.DataFrame`
        A per-linkage summary.
    """
    linkage_members = {
        "linkage_id": [],
        "obs_id": [],
    }
    all_linkages_expected = {
        "linkage_id": [],
        "num_members": [],
        "num_obs": [],
        "pure": [],
        "pure_complete": [],
        "partial": [],
        "mixed": [],
        "contamination_percentage": [],
        "found_pure": [],
        "found_partial": [],
        "found": [],
        "linked_object_id": [],
    }

    rng = np.random.default_rng(20230428)

    # Create pure complete linkages for each object
    for object_id in ["23636", "58177", "82134"]:
        obs_ids = test_observations[test_observations["object_id"] == object_id]["obs_id"].values
        linkage_id = f"pure_complete_{object_id}"
        for obs_id in obs_ids:
            linkage_members["linkage_id"].append(linkage_id)
            linkage_members["obs_id"].append(obs_id)

        all_linkages_expected["linkage_id"].append(linkage_id)
        all_linkages_expected["num_members"].append(1)
        all_linkages_expected["num_obs"].append(len(obs_ids))
        all_linkages_expected["pure"].append(True)
        all_linkages_expected["pure_complete"].append(True)
        all_linkages_expected["partial"].append(False)
        all_linkages_expected["mixed"].append(False)
        all_linkages_expected["contamination_percentage"].append(0.0)
        all_linkages_expected["found_pure"].append(True)
        all_linkages_expected["found_partial"].append(False)
        all_linkages_expected["found"].append(True)
        all_linkages_expected["linked_object_id"].append(object_id)

    # Create pure linkages but not complete for each object
    for object_id in ["23636", "58177", "82134"]:
        obs_ids = test_observations[test_observations["object_id"] == object_id]["obs_id"].values[2:]
        linkage_id = f"pure_{object_id}"
        for obs_id in obs_ids:
            linkage_members["linkage_id"].append(linkage_id)
            linkage_members["obs_id"].append(obs_id)

        all_linkages_expected["linkage_id"].append(linkage_id)
        all_linkages_expected["num_members"].append(1)
        all_linkages_expected["num_obs"].append(len(obs_ids))
        all_linkages_expected["pure"].append(True)
        all_linkages_expected["pure_complete"].append(False)
        all_linkages_expected["partial"].append(False)
        all_linkages_expected["mixed"].append(False)
        all_linkages_expected["contamination_percentage"].append(0.0)
        all_linkages_expected["found_pure"].append(True)
        all_linkages_expected["found_partial"].append(False)
        all_linkages_expected["found"].append(True)
        all_linkages_expected["linked_object_id"].append(object_id)

    # Create partial linkages for each object (7 observations each)
    for object_id in ["23636", "58177", "82134"]:
        obs_ids = test_observations[test_observations["object_id"] == object_id]["obs_id"].values
        other_obs_ids = test_observations[test_observations["object_id"] != object_id]["obs_id"].values
        obs_ids = rng.choice(obs_ids, size=5, replace=False)
        obs_ids_linkage = rng.choice(other_obs_ids, size=2, replace=False)
        obs_ids = np.concatenate([obs_ids, obs_ids_linkage])
        obs_ids.sort()

        num_members = test_observations[test_observations["obs_id"].isin(obs_ids)]["object_id"].nunique()
        contamination_percentage = 100.0 * len(obs_ids_linkage) / len(obs_ids)

        linkage_id = f"partial_{object_id}"
        for obs_id in obs_ids:
            linkage_members["linkage_id"].append(linkage_id)
            linkage_members["obs_id"].append(obs_id)

        all_linkages_expected["linkage_id"].append(linkage_id)
        all_linkages_expected["num_members"].append(num_members)
        all_linkages_expected["num_obs"].append(len(obs_ids))
        all_linkages_expected["pure"].append(False)
        all_linkages_expected["pure_complete"].append(False)
        all_linkages_expected["partial"].append(True)
        all_linkages_expected["mixed"].append(False)
        all_linkages_expected["contamination_percentage"].append(contamination_percentage)
        all_linkages_expected["found_pure"].append(False)
        all_linkages_expected["found_partial"].append(True)
        all_linkages_expected["found"].append(True)
        all_linkages_expected["linked_object_id"].append(object_id)

    # Create mixed linkages (7 observations each)
    for i, object_id in enumerate(["23636", "58177", "82134"]):
        obs_ids = test_observations[test_observations["object_id"] == object_id]["obs_id"].values
        other_obs_ids = test_observations[test_observations["object_id"] != object_id]["obs_id"].values
        obs_ids = rng.choice(obs_ids, size=2, replace=False)
        obs_ids_linkage = rng.choice(other_obs_ids, size=5, replace=False)
        obs_ids = np.concatenate([obs_ids, obs_ids_linkage])
        obs_ids.sort()

        num_members = test_observations[test_observations["obs_id"].isin(obs_ids)]["object_id"].nunique()

        linkage_id = f"mixed_{i}"
        for obs_id in obs_ids:
            linkage_members["linkage_id"].append(linkage_id)
            linkage_members["obs_id"].append(obs_id)

        all_linkages_expected["linkage_id"].append(linkage_id)
        all_linkages_expected["num_members"].append(num_members)
        all_linkages_expected["num_obs"].append(len(obs_ids))
        all_linkages_expected["pure"].append(False)
        all_linkages_expected["pure_complete"].append(False)
        all_linkages_expected["partial"].append(False)
        all_linkages_expected["mixed"].append(True)
        all_linkages_expected["contamination_percentage"].append(np.NaN)
        all_linkages_expected["found_pure"].append(False)
        all_linkages_expected["found_partial"].append(False)
        all_linkages_expected["found"].append(False)
        all_linkages_expected["linked_object_id"].append("nan")

    linkage_members = pd.DataFrame(linkage_members)
    all_linkages_expected = pd.DataFrame(all_linkages_expected)

    return linkage_members, all_linkages_expected


@pytest.fixture
def test_all_objects(test_observations):
    """
    Creates a dataframe with the truth for each observation.
    """
    all_truths_dict = {
        "object_id": [],
        "num_obs": [],
    }
    for truth in test_observations["object_id"].unique():
        all_truths_dict["object_id"].append(truth)
        all_truths_dict["num_obs"].append(len(test_observations[test_observations["object_id"] == truth]))

    all_truths = pd.DataFrame(all_truths_dict)
    all_truths.sort_values(by="object_id", inplace=True, ignore_index=True)
    return all_truths
