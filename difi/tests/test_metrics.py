import pytest
import numpy as np
import pandas as pd 
from pandas.testing import assert_frame_equal

from ..metrics import calcFindableMinObs
from ..metrics import calcFindableNightlyLinkages
from .create_test_data import createTestDataSet

MIN_OBS = range(5, 10)

def test_calcFindableMinObs():
    ### Test calcFindableMinObs against the test data set

    column_mapping = {
        "truth" : "truth",
        "obs_id" : "obs_id",
    }
    
    for min_obs in MIN_OBS:
        # Generate test data set
        observations_test, all_truths_test, linkage_members_test, all_linkages_test, summary_test = createTestDataSet(
            min_obs, 
            5, 
            20)
    
        findable_observations = calcFindableMinObs(observations_test, min_obs=min_obs, column_mapping=column_mapping)

        for truth in findable_observations.index:

            # Make sure all observations are correctly identified as findable
            obs_ids = findable_observations[findable_observations.index == truth]["obs_ids"].values[0]
            np.testing.assert_array_equal(obs_ids, observations_test[observations_test["truth"] == truth]["obs_id"].values)

            # Make sure all objects with not findable are not included in the findable_observations dataframe
            not_findable_truths_test = all_truths_test[all_truths_test["findable"] == 0]["truth"].values
            assert len(findable_observations[findable_observations.index.isin(not_findable_truths_test)]) == 0

    return

def test_calcFindableNightlyLinkages():
    ### Test calcFindableNightlyLinkages against the test data set
    column_mapping = {
        "truth" : "truth",
        "obs_id" : "obs_id",
        "time" : "time",
        "night" : "night",
    }
    # Generate test data set
    observations_test, all_truths_test, linkage_members_test, all_linkages_test, summary_test = createTestDataSet(
        5, 
        5, 
        20)

    # For every single truth in blue, their observations are seperated by a half day
    for truth in observations_test[observations_test["class"] == "blue"]["truth"].unique():
        mask = (observations_test["truth"] == truth)
        observations_test.loc[mask, "time"] = np.arange(0, len(observations_test[mask])/2, 0.5)

    # For every single truth in red, their observations are seperated by a quarter day
    for truth in observations_test[observations_test["class"] == "red"]["truth"].unique():
        mask = (observations_test["truth"] == truth)
        observations_test.loc[mask, "time"] = np.arange(0, len(observations_test[mask])/4, 0.25)
        
    # Observation times for greens are selected at random from the available ones in blues and greens
    observations_test.loc[observations_test["class"] == "green", "time"] = np.random.choice(
        observations_test[~observations_test["time"].isna()]["time"].values, 
        len(observations_test[observations_test["class"] == "green"]),
        replace=True)

    # Lets add a night column which is simply the floor of the observation time
    observations_test["night"] = np.floor(observations_test["time"]).astype(int)

    # With a maximum separation of 0.25 only reds should be findable
    findable_observations = calcFindableNightlyLinkages(
        observations_test,
        linkage_min_obs=2,
        max_obs_separation=0.25,
        min_linkage_nights=1,
        column_mapping=column_mapping
    )

    for truth in findable_observations.index:
        # Make sure all observations are correctly identified as findable
        obs_ids = findable_observations[findable_observations.index == truth]["obs_ids"].values[0]
        np.testing.assert_array_equal(obs_ids, observations_test[observations_test["truth"] == truth]["obs_id"].values)

    # Make sure that only reds were found
    classes_found = observations_test[observations_test["truth"].isin(findable_observations.index.values)]["class"].unique()
    np.testing.assert_array_equal(classes_found, np.array(["red"]))

    # With a maximum separation of 0.5 reds and blues should be findable
    findable_observations = calcFindableNightlyLinkages(
        observations_test,
        linkage_min_obs=2,
        max_obs_separation=0.5,
        min_linkage_nights=1,
        column_mapping=column_mapping
    )

    for truth in findable_observations.index:
        # Make sure all observations are correctly identified as findable
        obs_ids = findable_observations[findable_observations.index == truth]["obs_ids"].values[0]
        np.testing.assert_array_equal(obs_ids, observations_test[observations_test["truth"] == truth]["obs_id"].values)
        
    # Make sure that only reds and blues were found
    classes_found = observations_test[observations_test["truth"].isin(findable_observations.index.values)]["class"].unique()
    np.testing.assert_array_equal(classes_found, np.array(["red", "blue"]))

    # With a minimum linkage length of 1, everything should be findable
    findable_observations = calcFindableNightlyLinkages(
        observations_test,
        linkage_min_obs=1,
        max_obs_separation=0.5,
        min_linkage_nights=1,
        column_mapping=column_mapping
    )

    for truth in findable_observations.index:
        # Make sure all observations are correctly identified as findable
        obs_ids = findable_observations[findable_observations.index == truth]["obs_ids"].values[0]
        np.testing.assert_array_equal(obs_ids, observations_test[observations_test["truth"] == truth]["obs_id"].values)
        
    # Make sure that only reds and blues were found
    classes_found = observations_test[observations_test["truth"].isin(findable_observations.index.values)]["class"].unique()
    np.testing.assert_array_equal(classes_found, np.array(["red", "blue", "green"]))

    return