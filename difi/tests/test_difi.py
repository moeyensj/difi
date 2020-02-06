import os
import math
import string
import random
import pytest
import numpy as np
import pandas as pd 
from pandas.testing import assert_frame_equal

from ..difi import analyzeLinkages

def test_analyzeLinkages_fromFile():
    columnMapping = {
        "obs_id" : "obs_id",
        "linkage_id" : "linkage_id",
        "truth" : "truth"
    }

    # Load sample input
    linkageMembers = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "linkageMembers.txt"), 
        sep=" ", 
        index_col=False,
        dtype={
            "linkage_id" : int,
            "obs_id" : int,
        }
    )
    observations = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "observations.txt"), 
        sep=" ", 
        index_col=False,
        dtype={
            "truth" : str,
            "obs_id" : int,
        }
    )
    
    # Load solution
    allLinkages_solution = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "allLinkages_solution.txt"), 
        sep=" ",
        index_col=False,
        dtype={
            "linkage_id" : int,
            "linked_truth" : str,
        }
    )
    allTruths_solution = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "allTruths_solution.txt"),
        sep=" ", 
        index_col=False,
        dtype={
            "truth" : str,
        }
    )
    
    allLinkages_test, allTruths_test, summary_test = analyzeLinkages(observations, 
                                                                     linkageMembers,
                                                                     allLinkages=allLinkages_solution[["linkage_id"]],
                                                                     minObs=5, 
                                                                     contaminationThreshold=0.2,
                                                                     columnMapping=columnMapping)
    
    # Re-arange columns in case order is changed (python 3.5 and earlier)
    allLinkages_test = allLinkages_test[["linkage_id", 
                                         "num_members", 
                                         "num_obs", 
                                         "pure", 
                                         "partial", 
                                         "mixed", 
                                         "contamination", 
                                         "linked_truth"]]
    allTruths_test = allTruths_test[["truth", 
                                     "found_pure", 
                                     "found_partial", 
                                     "found"]] 
    
    # Assert dataframes are equal
    assert_frame_equal(allLinkages_test, allLinkages_solution)
    assert_frame_equal(allTruths_test, allTruths_solution)
    
def test_analyzeLinkages_singleObject_found():
    # Create randomly sized pure linkage
    num_obs = np.random.randint(2, 100000)
    name = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(np.random.randint(64))])
    truth = [name for i in range(num_obs)]
    obs_ids = np.arange(1, num_obs + 1)
    linkage_id = np.random.randint(10000000000)
    linkage_ids = [linkage_id for i in range(num_obs)]
    columnMapping = {
        "linkage_id" : ''.join([random.choice(string.ascii_letters + string.digits) for n in range(np.random.randint(64))]),
        "obs_id" : ''.join([random.choice(string.ascii_letters + string.digits) for n in range(np.random.randint(64))]),
        "truth" : ''.join([random.choice(string.ascii_letters + string.digits) for n in range(np.random.randint(64))])
    }

    # Create the linkageMembers dataframe
    linkageMembers = pd.DataFrame(
        {
            columnMapping["linkage_id"] : linkage_ids,
            columnMapping["obs_id"] : obs_ids
        },
    )

    # Create the observations dataframe
    observations = pd.DataFrame(
        {
            columnMapping["obs_id"] : obs_ids,
            columnMapping["truth"] : truth,
        },
    )
    observations[columnMapping["obs_id"]] = observations[columnMapping["obs_id"]].astype(int)

    # Run analysis for case when it should found
    allLinkages_test, allTruths_test, summary_test = analyzeLinkages(observations, 
                                                                     linkageMembers,
                                                                     minObs=num_obs - 1, 
                                                                     columnMapping=columnMapping, 
                                                                     verbose=True)

    ### Test allLinkages

    # Test the the length of all the dataframes is one
    for df in [allLinkages_test, allTruths_test, summary_test]:
        assert len(df) == 1

    # Test that the number of observations were correctly counted
    assert allLinkages_test["num_obs"].values[0] == num_obs

    # Test that the number of members is 1
    assert allLinkages_test["num_members"].values[0] == 1

    # Test that the number of linkages is as expected
    assert allLinkages_test["pure"].values[0] == 1
    assert allLinkages_test["partial"].values[0] == 0
    assert allLinkages_test["mixed"].values[0] == 0
    assert math.isnan(allLinkages_test["contamination"].values[0])

    # Test the linked truth is the name
    assert allLinkages_test["linked_truth"].values[0] == name

    ### Test allTruths

    # Test number of linkages is consistent
    assert allTruths_test["found_pure"].values[0] == 1
    assert allTruths_test["found_partial"].values[0] == 0
    assert allTruths_test["found"].values[0] == 1

    # Test that the truth is the name
    assert allTruths_test[columnMapping["truth"]].values[0] == name

    ### Test summary
    summary = pd.DataFrame({
        'num_unique_known_truths_found' : [1],
        'num_unique_known_truths_missed' : [np.NaN],
        'percent_completeness' : [np.NaN],
        'num_known_truths_pure_linkages' : [1],
        'num_known_truths_partial_linkages' : [0],
        'num_unknown_truths_pure_linkages' : [0],
        'num_unknown_truths_partial_linkages' : [0],
        'num_false_positive_pure_linkages' : [0],
        'num_false_positive_partial_linkages' : [0],
        'num_mixed_linkages' : [0],
        'num_total_linkages' : [1]
    })
    
    # Re-arange columns in case order is changed (python 3.5 and earlier)
    summary = summary[[
        'num_unique_known_truths_found', 
        'num_unique_known_truths_missed',
        'percent_completeness',
        'num_known_truths_pure_linkages',
        'num_known_truths_partial_linkages', 
        'num_unknown_truths_pure_linkages',
        'num_unknown_truths_partial_linkages',
        'num_false_positive_pure_linkages',
        'num_false_positive_partial_linkages',
        'num_mixed_linkages',
        'num_total_linkages']]

    assert_frame_equal(summary, summary_test)
    
    # Add findable column
    allTruths_test["findable"] = [1]
    
    # Run analysis for case when it should found with a findable column 
    # suggesting it is findable (pass allTruths)
    allLinkages_test, allTruths_test, summary_test = analyzeLinkages(observations, 
                                                                     linkageMembers,
                                                                     allTruths=allTruths_test,
                                                                     minObs=num_obs - 1, 
                                                                     columnMapping=columnMapping, 
                                                                     verbose=True)

    ### Test allLinkages

    # Test the the length of all the dataframes is one
    for df in [allLinkages_test, allTruths_test, summary_test]:
        assert len(df) == 1

    # Test that the number of observations were correctly counted
    assert allLinkages_test["num_obs"].values[0] == num_obs

    # Test that the number of members is 1
    assert allLinkages_test["num_members"].values[0] == 1

    # Test that the number of linkages is as expected
    assert allLinkages_test["pure"].values[0] == 1
    assert allLinkages_test["partial"].values[0] == 0
    assert allLinkages_test["mixed"].values[0] == 0
    assert math.isnan(allLinkages_test["contamination"].values[0])

    # Test the linked truth is the name
    assert allLinkages_test["linked_truth"].values[0] == name

    ### Test allTruths

    # Test number of linkages is consistent
    assert allTruths_test["found_pure"].values[0] == 1
    assert allTruths_test["found_partial"].values[0] == 0
    assert allTruths_test["found"].values[0] == 1

    # Test that the truth is the name
    assert allTruths_test[columnMapping["truth"]].values[0] == name

    ### Test summary
    summary = pd.DataFrame({
        'num_unique_known_truths_found' : [1],
        'num_unique_known_truths_missed' : [0],
        'percent_completeness' : [100.0],
        'num_known_truths_pure_linkages' : [1],
        'num_known_truths_partial_linkages' : [0],
        'num_unknown_truths_pure_linkages' : [0],
        'num_unknown_truths_partial_linkages' : [0],
        'num_false_positive_pure_linkages' : [0],
        'num_false_positive_partial_linkages' : [0],
        'num_mixed_linkages' : [0],
        'num_total_linkages' : [1]
    })
    
    # Re-arange columns in case order is changed (python 3.5 and earlier)
    summary = summary[[
        'num_unique_known_truths_found', 
        'num_unique_known_truths_missed',
        'percent_completeness',
        'num_known_truths_pure_linkages',
        'num_known_truths_partial_linkages', 
        'num_unknown_truths_pure_linkages',
        'num_unknown_truths_partial_linkages',
        'num_false_positive_pure_linkages',
        'num_false_positive_partial_linkages',
        'num_mixed_linkages',
        'num_total_linkages']]

    assert_frame_equal(summary, summary_test)
    
    # Add findable column
    allTruths_test["findable"] = [0]
    
    # Run analysis for case when it should found with a findable column 
    # suggesting it is not findable (pass allTruths)
    allLinkages_test, allTruths_test, summary_test = analyzeLinkages(observations, 
                                                                     linkageMembers,
                                                                     allTruths=allTruths_test,
                                                                     minObs=num_obs - 1, 
                                                                     columnMapping=columnMapping, 
                                                                     verbose=True)

    ### Test allLinkages

    # Test the the length of all the dataframes is one
    for df in [allLinkages_test, allTruths_test, summary_test]:
        assert len(df) == 1

    # Test that the number of observations were correctly counted
    assert allLinkages_test["num_obs"].values[0] == num_obs

    # Test that the number of members is 1
    assert allLinkages_test["num_members"].values[0] == 1

    # Test that the number of linkages is as expected
    assert allLinkages_test["pure"].values[0] == 1
    assert allLinkages_test["partial"].values[0] == 0
    assert allLinkages_test["mixed"].values[0] == 0
    assert math.isnan(allLinkages_test["contamination"].values[0])
    
    # Test the linked truth is the name
    assert allLinkages_test["linked_truth"].values[0] == name

    ### Test allTruths

    # Test number of linkages is consistent
    assert allTruths_test["found_pure"].values[0] == 1
    assert allTruths_test["found_partial"].values[0] == 0
    assert allTruths_test["found"].values[0] == 1

    # Test that the truth is the name
    assert allTruths_test[columnMapping["truth"]].values[0] == name

    ### Test summary
    summary = pd.DataFrame({
        'num_unique_known_truths_found' : [1],
        'num_unique_known_truths_missed' : [0],
        'percent_completeness' : [100.0],
        'num_known_truths_pure_linkages' : [1],
        'num_known_truths_partial_linkages' : [0],
        'num_unknown_truths_pure_linkages' : [0],
        'num_unknown_truths_partial_linkages' : [0],
        'num_false_positive_pure_linkages' : [0],
        'num_false_positive_partial_linkages' : [0],
        'num_mixed_linkages' : [0],
        'num_total_linkages' : [1]
    })
    
    # Re-arange columns in case order is changed (python 3.5 and earlier)
    summary = summary[[
        'num_unique_known_truths_found', 
        'num_unique_known_truths_missed',
        'percent_completeness',
        'num_known_truths_pure_linkages',
        'num_known_truths_partial_linkages', 
        'num_unknown_truths_pure_linkages',
        'num_unknown_truths_partial_linkages',
        'num_false_positive_pure_linkages',
        'num_false_positive_partial_linkages',
        'num_mixed_linkages',
        'num_total_linkages']]

    assert_frame_equal(summary, summary_test)

def test_analyzeLinkages_singleObject_missed():    
    # Create randomly sized pure linkage
    num_obs = np.random.randint(2, 100000)
    name = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(np.random.randint(64))])
    truth = [name for i in range(num_obs)]
    obs_ids = np.arange(1, num_obs + 1)
    linkage_id = np.random.randint(10000000000)
    linkage_ids = [linkage_id for i in range(num_obs)]
    columnMapping = {
        "linkage_id" : ''.join([random.choice(string.ascii_letters + string.digits) for n in range(np.random.randint(64))]),
        "obs_id" : ''.join([random.choice(string.ascii_letters + string.digits) for n in range(np.random.randint(64))]),
        "truth" : ''.join([random.choice(string.ascii_letters + string.digits) for n in range(np.random.randint(64))])
    }

    # Create the linkageMembers dataframe
    linkageMembers = pd.DataFrame(
        {
            columnMapping["linkage_id"] : linkage_ids,
            columnMapping["obs_id"] : obs_ids
        },
    )

    # Create the observations dataframe
    observations = pd.DataFrame(
        {
            columnMapping["obs_id"] : obs_ids,
            columnMapping["truth"] : truth,
        },
    )
    observations[columnMapping["obs_id"]] = observations[columnMapping["obs_id"]].astype(int)
    
    # Run analysis for case when it should not be found
    allLinkages_test, allTruths_test, summary_test = analyzeLinkages(observations, 
                                                                     linkageMembers, 
                                                                     minObs=num_obs + 1, 
                                                                     columnMapping=columnMapping, 
                                                                     verbose=True)

    ### Test allLinkages

    # Test the the length of all the dataframes is one
    for df in [allLinkages_test, allTruths_test, summary_test]:
        assert len(df) == 1

    # Test that the number of observations were correctly counted
    assert allLinkages_test["num_obs"].values[0] == num_obs

    # Test that the number of members is 1
    assert allLinkages_test["num_members"].values[0] == 1

    # Test that the number of linkages is as expected
    assert allLinkages_test["pure"].values[0] == 0
    assert allLinkages_test["partial"].values[0] == 0
    assert allLinkages_test["mixed"].values[0] == 1
    assert math.isnan(allLinkages_test["contamination"].values[0])

    # Test the linked truth is nan
    assert math.isnan(allLinkages_test["linked_truth"].values[0])
    
    ### Test allTruths

    # Test number of linkages is consistent
    assert allTruths_test["found_pure"].values[0] == 0
    assert allTruths_test["found_partial"].values[0] == 0
    assert allTruths_test["found"].values[0] == 0

    # Test that the truth is the name
    assert allTruths_test[columnMapping["truth"]].values[0] == name

    ### Test summary
    summary = pd.DataFrame({
        'num_unique_known_truths_found' : [0],
        'num_unique_known_truths_missed' : [np.NaN],
        'percent_completeness' : [np.NaN],
        'num_known_truths_pure_linkages' : [0],
        'num_known_truths_partial_linkages' : [0],
        'num_unknown_truths_pure_linkages' : [0],
        'num_unknown_truths_partial_linkages' : [0],
        'num_false_positive_pure_linkages' : [0],
        'num_false_positive_partial_linkages' : [0],
        'num_mixed_linkages' : [1],
        'num_total_linkages' : [1]
    })
    
    # Re-arange columns in case order is changed (python 3.5 and earlier)
    summary = summary[[
        'num_unique_known_truths_found', 
        'num_unique_known_truths_missed',
        'percent_completeness',
        'num_known_truths_pure_linkages',
        'num_known_truths_partial_linkages', 
        'num_unknown_truths_pure_linkages',
        'num_unknown_truths_partial_linkages',
        'num_false_positive_pure_linkages',
        'num_false_positive_partial_linkages',
        'num_mixed_linkages',
        'num_total_linkages']]

    assert_frame_equal(summary, summary_test)
    
    # Add findable column
    allTruths_test["findable"] = [0]
    
    # Run analysis for case when it should not be found (with allTruths passed)
    allLinkages_test, allTruths_test, summary_test = analyzeLinkages(observations, 
                                                                     linkageMembers, 
                                                                     allTruths=allTruths_test,
                                                                     minObs=num_obs + 1, 
                                                                     columnMapping=columnMapping, 
                                                                     verbose=True)

    ### Test allLinkages

    # Test the the length of all the dataframes is one
    for df in [allLinkages_test, allTruths_test, summary_test]:
        assert len(df) == 1

    # Test that the number of observations were correctly counted
    assert allLinkages_test["num_obs"].values[0] == num_obs

    # Test that the number of members is 1
    assert allLinkages_test["num_members"].values[0] == 1

    # Test that the number of linkages is as expected
    assert allLinkages_test["pure"].values[0] == 0
    assert allLinkages_test["partial"].values[0] == 0
    assert allLinkages_test["mixed"].values[0] == 1
    assert math.isnan(allLinkages_test["contamination"].values[0])

    # Test the linked truth is nan
    assert math.isnan(allLinkages_test["linked_truth"].values[0])
    
    ### Test allTruths

    # Test number of linkages is consistent
    assert allTruths_test["found_pure"].values[0] == 0
    assert allTruths_test["found_partial"].values[0] == 0
    assert allTruths_test["found"].values[0] == 0

    # Test that the truth is the name
    assert allTruths_test[columnMapping["truth"]].values[0] == name

    ### Test summary
    summary = pd.DataFrame({
        'num_unique_known_truths_found' : [0],
        'num_unique_known_truths_missed' : [0],
        'percent_completeness' : [0.0],
        'num_known_truths_pure_linkages' : [0],
        'num_known_truths_partial_linkages' : [0],
        'num_unknown_truths_pure_linkages' : [0],
        'num_unknown_truths_partial_linkages' : [0],
        'num_false_positive_pure_linkages' : [0],
        'num_false_positive_partial_linkages' : [0],
        'num_mixed_linkages' : [1],
        'num_total_linkages' : [1]
    })
    
    # Re-arange columns in case order is changed (python 3.5 and earlier)
    summary = summary[[
        'num_unique_known_truths_found', 
        'num_unique_known_truths_missed',
        'percent_completeness',
        'num_known_truths_pure_linkages',
        'num_known_truths_partial_linkages', 
        'num_unknown_truths_pure_linkages',
        'num_unknown_truths_partial_linkages',
        'num_false_positive_pure_linkages',
        'num_false_positive_partial_linkages',
        'num_mixed_linkages',
        'num_total_linkages']]

    assert_frame_equal(summary, summary_test)
    
def test_analyzeLinkages_multiObject():   
    # Randomly assign names to the columns
    columnMapping = {
        "linkage_id" : ''.join([random.choice(string.ascii_letters + string.digits) for n in range(np.random.randint(64))]),
        "obs_id" : ''.join([random.choice(string.ascii_letters + string.digits) for n in range(np.random.randint(64))]),
        "truth" : ''.join([random.choice(string.ascii_letters + string.digits) for n in range(np.random.randint(64))])
    }

    linkageMembers_list = []
    observations_list = []

    # Create three randomly sized pure linkages
    prev_obs = 0
    min_obs = []
    names = []
    for i in range(0, 3):
        num_obs = np.random.randint(prev_obs + 2, prev_obs + 50)
        name = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(np.random.randint(64))])
        truth = [name for i in range(num_obs)]
        obs_ids = np.arange(prev_obs + 1, prev_obs + num_obs + 1)

        linkage_id = i + 1
        linkage_ids = [linkage_id for i in range(num_obs)]

        # Create the linkageMembers dataframe
        linkageMembers = pd.DataFrame(
            {
                columnMapping["linkage_id"] : linkage_ids,
                columnMapping["obs_id"] : obs_ids
            },

        )

        # Create the observations dataframe
        observations = pd.DataFrame(
            {
                columnMapping["obs_id"] : obs_ids,
                columnMapping["truth"] : truth,
            },

        )

        linkageMembers_list.append(linkageMembers)
        observations_list.append(observations)

        prev_obs += num_obs
        min_obs.append(num_obs)
        names.append(name)

    # Concatenate dataframes and grab number of observations for each
    # unique object
    linkageMembers = pd.concat(linkageMembers_list, sort=False)
    observations = pd.concat(observations_list, sort=False)
    minObs = np.array(min_obs)
    names = np.array(names)
    
    #### CASE 1: Testing pure and mixed linkages
    
    # Case 1a: Run analysis for case when all three should not be found

    allLinkages_test, allTruths_test, summary_test = analyzeLinkages(observations, 
                                                                     linkageMembers, 
                                                                     minObs=minObs.max() + 1, 
                                                                     columnMapping=columnMapping, 
                                                                     verbose=True)
    ### Test allLinkages

    # Test that the length of allLinkages and allTruths dataframes is one
    for df in [allLinkages_test, allTruths_test]:
        assert len(df) == 3

    # Test that the length of the summary dataframe is one
    assert len(summary_test) == 1

    # Test that the number of observations were correctly counted
    np.testing.assert_equal(allLinkages_test["num_obs"].values, minObs)

    # Test that the number of members is 1
    np.testing.assert_equal(allLinkages_test["num_members"].values, np.array([1, 1, 1]))

    # Test that the number of linkages is as expected
    np.testing.assert_equal(allLinkages_test["pure"].values, np.array([0, 0, 0]))
    np.testing.assert_equal(allLinkages_test["partial"].values, np.array([0, 0, 0]))
    np.testing.assert_equal(allLinkages_test["mixed"].values, np.array([1, 1, 1]))
    for i in allLinkages_test["contamination"].values:
        assert math.isnan(i)

    # Test the linked truth is nan
    for i in allLinkages_test["linked_truth"].values:
        assert math.isnan(i)

    ### Test allTruths

    # Test number of linkages is consistent
    np.testing.assert_equal(allTruths_test["found_pure"].values, np.array([0, 0, 0]))
    np.testing.assert_equal(allTruths_test["found_partial"].values, np.array([0, 0, 0]))
    np.testing.assert_equal(allTruths_test["found"].values, np.array([0, 0, 0]))

    # Test that the truth is the name
    np.testing.assert_equal(allTruths_test[columnMapping["truth"]].values, names)

    ### Test summary
    summary = pd.DataFrame({
        'num_unique_known_truths_found' : [0],
        'num_unique_known_truths_missed' : [np.NaN],
        'percent_completeness' : [np.NaN],
        'num_known_truths_pure_linkages' : [0],
        'num_known_truths_partial_linkages' : [0],
        'num_unknown_truths_pure_linkages' : [0],
        'num_unknown_truths_partial_linkages' : [0],
        'num_false_positive_pure_linkages' : [0],
        'num_false_positive_partial_linkages' : [0],
        'num_mixed_linkages' : [3],
        'num_total_linkages' : [3]
    })

    # Re-arange columns in case order is changed (python 3.5 and earlier)
    summary = summary[[
        'num_unique_known_truths_found', 
        'num_unique_known_truths_missed',
        'percent_completeness',
        'num_known_truths_pure_linkages',
        'num_known_truths_partial_linkages', 
        'num_unknown_truths_pure_linkages',
        'num_unknown_truths_partial_linkages',
        'num_false_positive_pure_linkages',
        'num_false_positive_partial_linkages',
        'num_mixed_linkages',
        'num_total_linkages']]

    assert_frame_equal(summary, summary_test)

    # Case 1b: Run analysis for case when one should be found

    allLinkages_test, allTruths_test, summary_test = analyzeLinkages(observations, 
                                                                     linkageMembers, 
                                                                     minObs=minObs.max() - 1, 
                                                                     columnMapping=columnMapping, 
                                                                     verbose=True)
    ### Test allLinkages

    # Test that the length of allLinkages and allTruths dataframes is one
    for df in [allLinkages_test, allTruths_test]:
        assert len(df) == 3

    # Test that the length of the summary dataframe is one
    assert len(summary_test) == 1

    # Test that the number of observations were correctly counted
    np.testing.assert_equal(allLinkages_test["num_obs"].values, minObs)

    # Test that the number of members is 1
    np.testing.assert_equal(allLinkages_test["num_members"].values, np.array([1, 1, 1]))

    # Test that the number of linkages is as expected
    np.testing.assert_equal(allLinkages_test["pure"].values, np.array([0, 0, 1]))
    np.testing.assert_equal(allLinkages_test["partial"].values, np.array([0, 0, 0]))
    np.testing.assert_equal(allLinkages_test["mixed"].values, np.array([1, 1, 0]))
    for i in allLinkages_test["contamination"].values:
        assert math.isnan(i)

    # Test the linked truth is nan
    for i in allLinkages_test["linked_truth"].values[0:2]:
        assert math.isnan(i)
    # Test the linked truth for other two linkages is correct
    np.testing.assert_equal(allLinkages_test["linked_truth"].values[-1], names[-1])

    ### Test allTruths

    # Test number of linkages is consistent
    np.testing.assert_equal(allTruths_test["found_pure"].values, np.array([0, 0, 1]))
    np.testing.assert_equal(allTruths_test["found_partial"].values, np.array([0, 0, 0]))
    np.testing.assert_equal(allTruths_test["found"].values, np.array([0, 0, 1]))

    # Test that the truth is the name
    np.testing.assert_equal(allTruths_test[columnMapping["truth"]].values, names)

    ### Test summary
    summary = pd.DataFrame({
        'num_unique_known_truths_found' : [1],
        'num_unique_known_truths_missed' : [np.NaN],
        'percent_completeness' : [np.NaN],
        'num_known_truths_pure_linkages' : [1],
        'num_known_truths_partial_linkages' : [0],
        'num_unknown_truths_pure_linkages' : [0],
        'num_unknown_truths_partial_linkages' : [0],
        'num_false_positive_pure_linkages' : [0],
        'num_false_positive_partial_linkages' : [0],
        'num_mixed_linkages' : [2],
        'num_total_linkages' : [3]
    })

    # Re-arange columns in case order is changed (python 3.5 and earlier)
    summary = summary[[
        'num_unique_known_truths_found', 
        'num_unique_known_truths_missed',
        'percent_completeness',
        'num_known_truths_pure_linkages',
        'num_known_truths_partial_linkages', 
        'num_unknown_truths_pure_linkages',
        'num_unknown_truths_partial_linkages',
        'num_false_positive_pure_linkages',
        'num_false_positive_partial_linkages',
        'num_mixed_linkages',
        'num_total_linkages']]

    assert_frame_equal(summary, summary_test)

    # Case 1c: Run analysis for case when all should be found

    allLinkages_test, allTruths_test, summary_test = analyzeLinkages(observations, 
                                                                     linkageMembers, 
                                                                     minObs=minObs.min() - 1, 
                                                                     columnMapping=columnMapping, 
                                                                     verbose=True)
    ### Test allLinkages

    # Test that the length of allLinkages and allTruths dataframes is one
    for df in [allLinkages_test, allTruths_test]:
        assert len(df) == 3

    # Test that the length of the summary dataframe is one
    assert len(summary_test) == 1

    # Test that the number of observations were correctly counted
    np.testing.assert_equal(allLinkages_test["num_obs"].values, minObs)

    # Test that the number of members is 1
    np.testing.assert_equal(allLinkages_test["num_members"].values, np.array([1, 1, 1]))

    # Test that the number of linkages is as expected
    np.testing.assert_equal(allLinkages_test["pure"].values, np.array([1, 1, 1]))
    np.testing.assert_equal(allLinkages_test["partial"].values, np.array([0, 0, 0]))
    np.testing.assert_equal(allLinkages_test["mixed"].values, np.array([0, 0, 0]))
    for i in allLinkages_test["contamination"].values:
        assert math.isnan(i)

    # Test the linked truth is equal to names
    np.testing.assert_equal(allLinkages_test["linked_truth"].values, names)

    ### Test allTruths

    # Test number of linkages is consistent
    np.testing.assert_equal(allTruths_test["found_pure"].values, np.array([1, 1, 1]))
    np.testing.assert_equal(allTruths_test["found_partial"].values, np.array([0, 0, 0]))
    np.testing.assert_equal(allTruths_test["found"].values, np.array([1, 1, 1]))

    # Test that the truth is the name
    np.testing.assert_equal(allTruths_test[columnMapping["truth"]].values, names)

    ### Test summary
    summary = pd.DataFrame({
        'num_unique_known_truths_found' : [3],
        'num_unique_known_truths_missed' : [np.NaN],
        'percent_completeness' : [np.NaN],
        'num_known_truths_pure_linkages' : [3],
        'num_known_truths_partial_linkages' : [0],
        'num_unknown_truths_pure_linkages' : [0],
        'num_unknown_truths_partial_linkages' : [0],
        'num_false_positive_pure_linkages' : [0],
        'num_false_positive_partial_linkages' : [0],
        'num_mixed_linkages' : [0],
        'num_total_linkages' : [3]
    })

    # Re-arange columns in case order is changed (python 3.5 and earlier)
    summary = summary[[
        'num_unique_known_truths_found', 
        'num_unique_known_truths_missed',
        'percent_completeness',
        'num_known_truths_pure_linkages',
        'num_known_truths_partial_linkages', 
        'num_unknown_truths_pure_linkages',
        'num_unknown_truths_partial_linkages',
        'num_false_positive_pure_linkages',
        'num_false_positive_partial_linkages',
        'num_mixed_linkages',
        'num_total_linkages']]

    assert_frame_equal(summary, summary_test)

    # Case 1d: Run analysis for case when all should be found, make second object unknown 
    # make third object a false positive

    allLinkages_test, allTruths_test, summary_test = analyzeLinkages(observations, 
                                                                     linkageMembers, 
                                                                     unknownIDs=[names[1]],
                                                                     falsePositiveIDs=[names[2]],
                                                                     minObs=minObs.min() - 1, 
                                                                     columnMapping=columnMapping, 
                                                                     verbose=True)
    ### Test allLinkages

    # Test that the length of allLinkages and allTruths dataframes is one
    for df in [allLinkages_test, allTruths_test]:
        assert len(df) == 3

    # Test that the length of the summary dataframe is one
    assert len(summary_test) == 1

    # Test that the number of observations were correctly counted
    np.testing.assert_equal(allLinkages_test["num_obs"].values, minObs)

    # Test that the number of members is 1
    np.testing.assert_equal(allLinkages_test["num_members"].values, np.array([1, 1, 1]))

    # Test that the number of linkages is as expected
    np.testing.assert_equal(allLinkages_test["pure"].values, np.array([1, 1, 1]))
    np.testing.assert_equal(allLinkages_test["partial"].values, np.array([0, 0, 0]))
    np.testing.assert_equal(allLinkages_test["mixed"].values, np.array([0, 0, 0]))
    for i in allLinkages_test["contamination"].values:
        assert math.isnan(i)

    # Test the linked truth is equal to names
    np.testing.assert_equal(allLinkages_test["linked_truth"].values, names)

    ### Test allTruths

    # Test number of linkages is consistent
    np.testing.assert_equal(allTruths_test["found_pure"].values, np.array([1, 1, 1]))
    np.testing.assert_equal(allTruths_test["found_partial"].values, np.array([0, 0, 0]))
    np.testing.assert_equal(allTruths_test["found"].values, np.array([1, 1, 1]))

    # Test that the truth is the name
    np.testing.assert_equal(allTruths_test[columnMapping["truth"]].values, names)

    ### Test summary
    summary = pd.DataFrame({
        'num_unique_known_truths_found' : [1],
        'num_unique_known_truths_missed' : [np.NaN],
        'percent_completeness' : [np.NaN],
        'num_known_truths_pure_linkages' : [1],
        'num_known_truths_partial_linkages' : [0],
        'num_unknown_truths_pure_linkages' : [1],
        'num_unknown_truths_partial_linkages' : [0],
        'num_false_positive_pure_linkages' : [1],
        'num_false_positive_partial_linkages' : [0],
        'num_mixed_linkages' : [0],
        'num_total_linkages' : [3]
    })

    # Re-arange columns in case order is changed (python 3.5 and earlier)
    summary = summary[[
        'num_unique_known_truths_found', 
        'num_unique_known_truths_missed',
        'percent_completeness',
        'num_known_truths_pure_linkages',
        'num_known_truths_partial_linkages', 
        'num_unknown_truths_pure_linkages',
        'num_unknown_truths_partial_linkages',
        'num_false_positive_pure_linkages',
        'num_false_positive_partial_linkages',
        'num_mixed_linkages',
        'num_total_linkages']]

    assert_frame_equal(summary, summary_test)

    # Case 1e: Run analysis for case when all should be found, make second object unknown 
    # make third object a false positive, make them all findable
    allTruths_test["findable"] = np.ones(3, dtype=int)

    allLinkages_test, allTruths_test, summary_test = analyzeLinkages(observations, 
                                                                     linkageMembers, 
                                                                     allTruths=allTruths_test,
                                                                     unknownIDs=[names[1]],
                                                                     falsePositiveIDs=[names[2]],
                                                                     minObs=minObs.min() - 1, 
                                                                     columnMapping=columnMapping, 
                                                                     verbose=True)
    ### Test allLinkages

    # Test that the length of allLinkages and allTruths dataframes is one
    for df in [allLinkages_test, allTruths_test]:
        assert len(df) == 3

    # Test that the length of the summary dataframe is one
    assert len(summary_test) == 1

    # Test that the number of observations were correctly counted
    np.testing.assert_equal(allLinkages_test["num_obs"].values, minObs)

    # Test that the number of members is 1
    np.testing.assert_equal(allLinkages_test["num_members"].values, np.array([1, 1, 1]))

    # Test that the number of linkages is as expected
    np.testing.assert_equal(allLinkages_test["pure"].values, np.array([1, 1, 1]))
    np.testing.assert_equal(allLinkages_test["partial"].values, np.array([0, 0, 0]))
    np.testing.assert_equal(allLinkages_test["mixed"].values, np.array([0, 0, 0]))
    for i in allLinkages_test["contamination"].values:
        assert math.isnan(i)

    # Test the linked truth is equal to names
    np.testing.assert_equal(allLinkages_test["linked_truth"].values, names)

    ### Test allTruths

    # Test number of linkages is consistent
    np.testing.assert_equal(allTruths_test["found_pure"].values, np.array([1, 1, 1]))
    np.testing.assert_equal(allTruths_test["found_partial"].values, np.array([0, 0, 0]))
    np.testing.assert_equal(allTruths_test["found"].values, np.array([1, 1, 1]))

    # Test that the truth is the name
    np.testing.assert_equal(allTruths_test[columnMapping["truth"]].values, names)

    ### Test summary
    summary = pd.DataFrame({
        'num_unique_known_truths_found' : [1],
        'num_unique_known_truths_missed' : [0],
        'percent_completeness' : [100.0],
        'num_known_truths_pure_linkages' : [1],
        'num_known_truths_partial_linkages' : [0],
        'num_unknown_truths_pure_linkages' : [1],
        'num_unknown_truths_partial_linkages' : [0],
        'num_false_positive_pure_linkages' : [1],
        'num_false_positive_partial_linkages' : [0],
        'num_mixed_linkages' : [0],
        'num_total_linkages' : [3]
    })

    # Re-arange columns in case order is changed (python 3.5 and earlier)
    summary = summary[[
        'num_unique_known_truths_found', 
        'num_unique_known_truths_missed',
        'percent_completeness',
        'num_known_truths_pure_linkages',
        'num_known_truths_partial_linkages', 
        'num_unknown_truths_pure_linkages',
        'num_unknown_truths_partial_linkages',
        'num_false_positive_pure_linkages',
        'num_false_positive_partial_linkages',
        'num_mixed_linkages',
        'num_total_linkages']]

    assert_frame_equal(summary, summary_test)

    # Case 1f: Run analysis for case when all should be found, make second object unknown 
    # make third object a false positive, make only the latter two findable
    allTruths_test["findable"] = np.array([0, 1, 1])

    allLinkages_test, allTruths_test, summary_test = analyzeLinkages(observations, 
                                                                     linkageMembers, 
                                                                     allTruths=allTruths_test,
                                                                     unknownIDs=[names[1]],
                                                                     falsePositiveIDs=[names[2]],
                                                                     minObs=minObs.min() - 1, 
                                                                     columnMapping=columnMapping, 
                                                                     verbose=True)
    ### Test allLinkages

    # Test that the length of allLinkages and allTruths dataframes is one
    for df in [allLinkages_test, allTruths_test]:
        assert len(df) == 3

    # Test that the length of the summary dataframe is one
    assert len(summary_test) == 1

    # Test that the number of observations were correctly counted
    np.testing.assert_equal(allLinkages_test["num_obs"].values, minObs)

    # Test that the number of members is 1
    np.testing.assert_equal(allLinkages_test["num_members"].values, np.array([1, 1, 1]))

    # Test that the number of linkages is as expected
    np.testing.assert_equal(allLinkages_test["pure"].values, np.array([1, 1, 1]))
    np.testing.assert_equal(allLinkages_test["partial"].values, np.array([0, 0, 0]))
    np.testing.assert_equal(allLinkages_test["mixed"].values, np.array([0, 0, 0]))
    for i in allLinkages_test["contamination"].values:
        assert math.isnan(i)

    # Test the linked truth is equal to names
    np.testing.assert_equal(allLinkages_test["linked_truth"].values, names)

    ### Test allTruths

    # Test number of linkages is consistent
    np.testing.assert_equal(allTruths_test["found_pure"].values, np.array([1, 1, 1]))
    np.testing.assert_equal(allTruths_test["found_partial"].values, np.array([0, 0, 0]))
    np.testing.assert_equal(allTruths_test["found"].values, np.array([1, 1, 1]))

    # Test that the truth is the name
    np.testing.assert_equal(allTruths_test[columnMapping["truth"]].values, names)

    ### Test summary
    summary = pd.DataFrame({
        'num_unique_known_truths_found' : [1],
        'num_unique_known_truths_missed' : [0],
        'percent_completeness' : [100.0],
        'num_known_truths_pure_linkages' : [1],
        'num_known_truths_partial_linkages' : [0],
        'num_unknown_truths_pure_linkages' : [1],
        'num_unknown_truths_partial_linkages' : [0],
        'num_false_positive_pure_linkages' : [1],
        'num_false_positive_partial_linkages' : [0],
        'num_mixed_linkages' : [0],
        'num_total_linkages' : [3]
    })

    # Re-arange columns in case order is changed (python 3.5 and earlier)
    summary = summary[[
        'num_unique_known_truths_found', 
        'num_unique_known_truths_missed',
        'percent_completeness',
        'num_known_truths_pure_linkages',
        'num_known_truths_partial_linkages', 
        'num_unknown_truths_pure_linkages',
        'num_unknown_truths_partial_linkages',
        'num_false_positive_pure_linkages',
        'num_false_positive_partial_linkages',
        'num_mixed_linkages',
        'num_total_linkages']]

    assert_frame_equal(summary, summary_test)

    # Case 1f: Run analysis for case when all should be found
    # make third object a false positive, make only the latter two findable
    allTruths_test["findable"] = np.array([0, 1, 1])

    allLinkages_test, allTruths_test, summary_test = analyzeLinkages(observations, 
                                                                     linkageMembers, 
                                                                     allTruths=allTruths_test,
                                                                     unknownIDs=[],
                                                                     falsePositiveIDs=[names[2]],
                                                                     minObs=minObs.min() - 1, 
                                                                     columnMapping=columnMapping, 
                                                                     verbose=True)
    ### Test allLinkages

    # Test that the length of allLinkages and allTruths dataframes is one
    for df in [allLinkages_test, allTruths_test]:
        assert len(df) == 3

    # Test that the length of the summary dataframe is one
    assert len(summary_test) == 1

    # Test that the number of observations were correctly counted
    np.testing.assert_equal(allLinkages_test["num_obs"].values, minObs)

    # Test that the number of members is 1
    np.testing.assert_equal(allLinkages_test["num_members"].values, np.array([1, 1, 1]))

    # Test that the number of linkages is as expected
    np.testing.assert_equal(allLinkages_test["pure"].values, np.array([1, 1, 1]))
    np.testing.assert_equal(allLinkages_test["partial"].values, np.array([0, 0, 0]))
    np.testing.assert_equal(allLinkages_test["mixed"].values, np.array([0, 0, 0]))
    for i in allLinkages_test["contamination"].values:
        assert math.isnan(i)

    # Test the linked truth is equal to names
    np.testing.assert_equal(allLinkages_test["linked_truth"].values, names)

    ### Test allTruths

    # Test number of linkages is consistent
    np.testing.assert_equal(allTruths_test["found_pure"].values, np.array([1, 1, 1]))
    np.testing.assert_equal(allTruths_test["found_partial"].values, np.array([0, 0, 0]))
    np.testing.assert_equal(allTruths_test["found"].values, np.array([1, 1, 1]))

    # Test that the truth is the name
    np.testing.assert_equal(allTruths_test[columnMapping["truth"]].values, names)

    ### Test summary
    summary = pd.DataFrame({
        'num_unique_known_truths_found' : [2],
        'num_unique_known_truths_missed' : [0],
        'percent_completeness' : [200.0],
        'num_known_truths_pure_linkages' : [2],
        'num_known_truths_partial_linkages' : [0],
        'num_unknown_truths_pure_linkages' : [0],
        'num_unknown_truths_partial_linkages' : [0],
        'num_false_positive_pure_linkages' : [1],
        'num_false_positive_partial_linkages' : [0],
        'num_mixed_linkages' : [0],
        'num_total_linkages' : [3]
    })

    # Re-arange columns in case order is changed (python 3.5 and earlier)
    summary = summary[[
        'num_unique_known_truths_found', 
        'num_unique_known_truths_missed',
        'percent_completeness',
        'num_known_truths_pure_linkages',
        'num_known_truths_partial_linkages', 
        'num_unknown_truths_pure_linkages',
        'num_unknown_truths_partial_linkages',
        'num_false_positive_pure_linkages',
        'num_false_positive_partial_linkages',
        'num_mixed_linkages',
        'num_total_linkages']]

    assert_frame_equal(summary, summary_test)
    
    # Case 1g: Run analysis for case when only last two will be found
    # make third object a false positive, make all of them findable
    allTruths_test["findable"] = np.array([1, 1, 1])

    allLinkages_test, allTruths_test, summary_test = analyzeLinkages(observations, 
                                                                     linkageMembers, 
                                                                     allTruths=allTruths_test,
                                                                     unknownIDs=[],
                                                                     falsePositiveIDs=[names[2]],
                                                                     minObs=minObs.min() + 1, 
                                                                     columnMapping=columnMapping, 
                                                                     verbose=True)
    ### Test allLinkages

    # Test that the length of allLinkages and allTruths dataframes is one
    for df in [allLinkages_test, allTruths_test]:
        assert len(df) == 3

    # Test that the length of the summary dataframe is one
    assert len(summary_test) == 1

    # Test that the number of observations were correctly counted
    np.testing.assert_equal(allLinkages_test["num_obs"].values, minObs)

    # Test that the number of members is 1
    np.testing.assert_equal(allLinkages_test["num_members"].values, np.array([1, 1, 1]))

    # Test that the number of linkages is as expected
    np.testing.assert_equal(allLinkages_test["pure"].values, np.array([0, 1, 1]))
    np.testing.assert_equal(allLinkages_test["partial"].values, np.array([0, 0, 0]))
    np.testing.assert_equal(allLinkages_test["mixed"].values, np.array([1, 0, 0]))
    for i in allLinkages_test["contamination"].values:
        assert math.isnan(i)

    # Test the linked truth is nan
    assert math.isnan(allLinkages_test["linked_truth"].values[0])
       
    # Test the linked truth for other two linkages is correct
    np.testing.assert_equal(allLinkages_test["linked_truth"].values[1:], names[1:])

    ### Test allTruths

    # Test number of linkages is consistent
    np.testing.assert_equal(allTruths_test["found_pure"].values, np.array([0, 1, 1]))
    np.testing.assert_equal(allTruths_test["found_partial"].values, np.array([0, 0, 0]))
    np.testing.assert_equal(allTruths_test["found"].values, np.array([0, 1, 1]))

    # Test that the truth is the name
    np.testing.assert_equal(allTruths_test[columnMapping["truth"]].values, names)

    ### Test summary
    summary = pd.DataFrame({
        'num_unique_known_truths_found' : [1],
        'num_unique_known_truths_missed' : [1],
        'percent_completeness' : [50.0],
        'num_known_truths_pure_linkages' : [1],
        'num_known_truths_partial_linkages' : [0],
        'num_unknown_truths_pure_linkages' : [0],
        'num_unknown_truths_partial_linkages' : [0],
        'num_false_positive_pure_linkages' : [1],
        'num_false_positive_partial_linkages' : [0],
        'num_mixed_linkages' : [1],
        'num_total_linkages' : [3]
    })

    # Re-arange columns in case order is changed (python 3.5 and earlier)
    summary = summary[[
        'num_unique_known_truths_found', 
        'num_unique_known_truths_missed',
        'percent_completeness',
        'num_known_truths_pure_linkages',
        'num_known_truths_partial_linkages', 
        'num_unknown_truths_pure_linkages',
        'num_unknown_truths_partial_linkages',
        'num_false_positive_pure_linkages',
        'num_false_positive_partial_linkages',
        'num_mixed_linkages',
        'num_total_linkages']]

    assert_frame_equal(summary, summary_test)

    #### CASE 2: Testing pure, partial and mixed linkages

    # Case 2a: Run analysis such that the contaminationThreshold doesn't allow
    # any objects to be found

    # Grab a percentage of the detections in linkage 2 and linkage 3 and swap them
    obs_ids_2to3 = linkageMembers[linkageMembers[columnMapping["linkage_id"]] == 2][columnMapping["obs_id"]].values[int(0.85*minObs[1]):]
    obs_ids_3to2 = linkageMembers[linkageMembers[columnMapping["linkage_id"]] == 3][columnMapping["obs_id"]].values[0:int(0.2*minObs[2])]
    linkageMembers.loc[linkageMembers[columnMapping["obs_id"]].isin(obs_ids_2to3), columnMapping["linkage_id"]] = 3
    linkageMembers.loc[linkageMembers[columnMapping["obs_id"]].isin(obs_ids_3to2), columnMapping["linkage_id"]] = 2

    # Re-adjust minObs
    minObs = np.array([minObs[0], 
                       minObs[1] - len(obs_ids_2to3) + len(obs_ids_3to2),  
                       minObs[2] + len(obs_ids_2to3) - len(obs_ids_3to2)])

    allLinkages_test, allTruths_test, summary_test = analyzeLinkages(observations, 
                                                                     linkageMembers, 
                                                                     #allTruths=allTruths_test,
                                                                     unknownIDs=[],
                                                                     falsePositiveIDs=[],
                                                                     contaminationThreshold=0.0,
                                                                     minObs=minObs[1], 
                                                                     columnMapping=columnMapping, 
                                                                     verbose=True)
    ### Test allLinkages

    # Test that the length of allLinkages and allTruths dataframes is one
    for df in [allLinkages_test, allTruths_test]:
        assert len(df) == 3

    # Test that the length of the summary dataframe is one
    assert len(summary_test) == 1

    # Test that the number of observations were correctly counted
    np.testing.assert_equal(allLinkages_test["num_obs"].values, minObs)

    # Test that the number of members is 1
    np.testing.assert_equal(allLinkages_test["num_members"].values, np.array([1, 2, 2]))

    # Test that the number of linkages is as expected
    np.testing.assert_equal(allLinkages_test["pure"].values, np.array([0, 0, 0]))
    np.testing.assert_equal(allLinkages_test["partial"].values, np.array([0, 0, 0]))
    np.testing.assert_equal(allLinkages_test["mixed"].values, np.array([1, 1, 1]))
    for i in allLinkages_test["contamination"].values:
        assert math.isnan(i)

    # Test the linked truth is nan
    for i in allLinkages_test["linked_truth"].values:
        assert math.isnan(i)

    ### Test allTruths

    # Test number of linkages is consistent
    np.testing.assert_equal(allTruths_test["found_pure"].values, np.array([0, 0, 0]))
    np.testing.assert_equal(allTruths_test["found_partial"].values, np.array([0, 0, 0]))
    np.testing.assert_equal(allTruths_test["found"].values, np.array([0, 0, 0]))

    # Test that the truth is the name
    np.testing.assert_equal(allTruths_test[columnMapping["truth"]].values, names)

    ### Test summary
    summary = pd.DataFrame({
        'num_unique_known_truths_found' : [0],
        'num_unique_known_truths_missed' : [np.NaN],
        'percent_completeness' : [np.NaN],
        'num_known_truths_pure_linkages' : [0],
        'num_known_truths_partial_linkages' : [0],
        'num_unknown_truths_pure_linkages' : [0],
        'num_unknown_truths_partial_linkages' : [0],
        'num_false_positive_pure_linkages' : [0],
        'num_false_positive_partial_linkages' : [0],
        'num_mixed_linkages' : [3],
        'num_total_linkages' : [3]
    })

    # Re-arange columns in case order is changed (python 3.5 and earlier)
    summary = summary[[
        'num_unique_known_truths_found', 
        'num_unique_known_truths_missed',
        'percent_completeness',
        'num_known_truths_pure_linkages',
        'num_known_truths_partial_linkages', 
        'num_unknown_truths_pure_linkages',
        'num_unknown_truths_partial_linkages',
        'num_false_positive_pure_linkages',
        'num_false_positive_partial_linkages',
        'num_mixed_linkages',
        'num_total_linkages']]

    print(summary['num_unique_known_truths_missed'])
    print(summary_test['num_unique_known_truths_missed'])
    assert_frame_equal(summary, summary_test)

    # Case 2b: Run analysis such that the contaminationThreshold doesn't allow
    # only the last object to be found

    allLinkages_test, allTruths_test, summary_test = analyzeLinkages(observations, 
                                                                     linkageMembers, 
                                                                     #allTruths=allTruths_test,
                                                                     unknownIDs=[],
                                                                     falsePositiveIDs=[],
                                                                     contaminationThreshold=0.2,
                                                                     minObs=minObs[1], 
                                                                     columnMapping=columnMapping, 
                                                                     verbose=True)
    ### Test allLinkages

    # Test that the length of allLinkages and allTruths dataframes is one
    for df in [allLinkages_test, allTruths_test]:
        assert len(df) == 3

    # Test that the length of the summary dataframe is one
    assert len(summary_test) == 1

    # Test that the number of observations were correctly counted
    np.testing.assert_equal(allLinkages_test["num_obs"].values, minObs)

    # Test that the number of members is 1
    np.testing.assert_equal(allLinkages_test["num_members"].values, np.array([1, 2, 2]))

    # Test that the number of linkages is as expected
    np.testing.assert_equal(allLinkages_test["pure"].values, np.array([0, 0, 0]))
    np.testing.assert_equal(allLinkages_test["partial"].values, np.array([0, 0, 1]))
    np.testing.assert_equal(allLinkages_test["mixed"].values, np.array([1, 1, 0]))
    for i in allLinkages_test["contamination"].values[:-1]:
        assert math.isnan(i)
    np.testing.assert_allclose(allLinkages_test["contamination"].values[-1], len(obs_ids_2to3) / minObs[-1])

    # Test the linked truth is nan for the first two linkages
    for i in allLinkages_test["linked_truth"].values[:-1]:
        assert math.isnan(i)

    # Test that the linked truth for the third linkage is as expected
    assert allLinkages_test["linked_truth"].values[-1] == names[-1]

    ### Test allTruths

    # Test number of linkages is consistent
    np.testing.assert_equal(allTruths_test["found_pure"].values, np.array([0, 0, 0]))
    np.testing.assert_equal(allTruths_test["found_partial"].values, np.array([0, 0, 1]))
    np.testing.assert_equal(allTruths_test["found"].values, np.array([0, 0, 1]))

    # Test that the truth is the name
    np.testing.assert_equal(allTruths_test[columnMapping["truth"]].values, names)

    ### Test summary
    summary = pd.DataFrame({
        'num_unique_known_truths_found' : [1],
        'num_unique_known_truths_missed' : [np.NaN],
        'percent_completeness' : [np.NaN],
        'num_known_truths_pure_linkages' : [0],
        'num_known_truths_partial_linkages' : [1],
        'num_unknown_truths_pure_linkages' : [0],
        'num_unknown_truths_partial_linkages' : [0],
        'num_false_positive_pure_linkages' : [0],
        'num_false_positive_partial_linkages' : [0],
        'num_mixed_linkages' : [2],
        'num_total_linkages' : [3]
    })

    # Re-arange columns in case order is changed (python 3.5 and earlier)
    summary = summary[[
        'num_unique_known_truths_found', 
        'num_unique_known_truths_missed',
        'percent_completeness',
        'num_known_truths_pure_linkages',
        'num_known_truths_partial_linkages', 
        'num_unknown_truths_pure_linkages',
        'num_unknown_truths_partial_linkages',
        'num_false_positive_pure_linkages',
        'num_false_positive_partial_linkages',
        'num_mixed_linkages',
        'num_total_linkages']]

    assert_frame_equal(summary, summary_test)

def test_analyzeLinkages_emptyDataFrames():
    # Case 3a: Pass empty observations and linkageMembers DataFrames

    # Randomly assign names to the columns
    columnMapping = {
        "linkage_id" : ''.join([random.choice(string.ascii_letters + string.digits) for n in range(np.random.randint(64))]),
        "obs_id" : ''.join([random.choice(string.ascii_letters + string.digits) for n in range(np.random.randint(64))]),
        "truth" : ''.join([random.choice(string.ascii_letters + string.digits) for n in range(np.random.randint(64))])
    }


    # Create the linkageMembers dataframe
    linkageMembers_empty = pd.DataFrame(
        columns=[
            columnMapping["linkage_id"],
            columnMapping["obs_id"]
        ],
        dtype=int
    )

    # Create the observations dataframe
    observations_empty = pd.DataFrame(columns=[
        columnMapping["obs_id"],
        columnMapping["truth"]
    ])
    observations_empty[columnMapping["obs_id"]] = observations_empty[columnMapping["obs_id"]].astype(int)

    # Test an error is raised when the observations dataframe is empy
    with pytest.raises(ValueError):
        allLinkages_test, allTruths_test, summary_test = analyzeLinkages(observations_empty, 
                                                                         linkageMembers_empty, 
                                                                         unknownIDs=[],
                                                                         falsePositiveIDs=[],
                                                                         contaminationThreshold=0.2,
                                                                         minObs=5, 
                                                                         columnMapping=columnMapping, 
                                                                         verbose=True)

    # Case 3b: Pass non-empty observations and empty linkageMembers DataFrames
        
    # Create randomly sized pure linkage
    num_obs = np.random.randint(2, 100000)
    name = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(np.random.randint(64))])
    truth = [name for i in range(num_obs)]
    obs_ids = np.arange(1, num_obs + 1)
    columnMapping = {
        "linkage_id" : ''.join([random.choice(string.ascii_letters + string.digits) for n in range(np.random.randint(64))]),
        "obs_id" : ''.join([random.choice(string.ascii_letters + string.digits) for n in range(np.random.randint(64))]),
        "truth" : ''.join([random.choice(string.ascii_letters + string.digits) for n in range(np.random.randint(64))])
    }

    # Create the linkageMembers dataframe
    linkageMembers_empty = pd.DataFrame(
        columns=[
            columnMapping["linkage_id"],
            columnMapping["obs_id"]
        ],
        dtype=int
    )

    # Create the observations dataframe
    observations = pd.DataFrame(
        {
            columnMapping["obs_id"] : obs_ids,
            columnMapping["truth"] : truth,
        },
    )
    observations[columnMapping["obs_id"]] = observations[columnMapping["obs_id"]].astype(int)

    # Run analysis for case when it should found
    allLinkages_test, allTruths_test, summary_test = analyzeLinkages(observations, 
                                                                     linkageMembers_empty,
                                                                     minObs=num_obs - 1, 
                                                                     columnMapping=columnMapping, 
                                                                     verbose=True)

    # Test that the length of allLinkages is zero
    assert len(allLinkages_test) == 0

    # Test that the length of allTruths is one 
    assert len(allTruths_test) == 1

    # Test that the length of the summary dataframe is one
    assert len(summary_test) == 1

    ### Test allTruths

    # Test number of linkages is consistent
    np.testing.assert_equal(allTruths_test["found_pure"].values, np.array([0]))
    np.testing.assert_equal(allTruths_test["found_partial"].values, np.array([0]))
    np.testing.assert_equal(allTruths_test["found"].values, np.array([0]))

    # Test that the truth is the name
    np.testing.assert_equal(allTruths_test[columnMapping["truth"]].values, name)

    ### Test summary
    summary = pd.DataFrame({
        'num_unique_known_truths_found' : [0],
        'num_unique_known_truths_missed' : [np.NaN],
        'percent_completeness' : [np.NaN],
        'num_known_truths_pure_linkages' : [0],
        'num_known_truths_partial_linkages' : [0],
        'num_unknown_truths_pure_linkages' : [0],
        'num_unknown_truths_partial_linkages' : [0],
        'num_false_positive_pure_linkages' : [0],
        'num_false_positive_partial_linkages' : [0],
        'num_mixed_linkages' : [0],
        'num_total_linkages' : [0]
    })

    # Re-arange columns in case order is changed (python 3.5 and earlier)
    summary = summary[[
        'num_unique_known_truths_found', 
        'num_unique_known_truths_missed',
        'percent_completeness',
        'num_known_truths_pure_linkages',
        'num_known_truths_partial_linkages', 
        'num_unknown_truths_pure_linkages',
        'num_unknown_truths_partial_linkages',
        'num_false_positive_pure_linkages',
        'num_false_positive_partial_linkages',
        'num_mixed_linkages',
        'num_total_linkages']]

    assert_frame_equal(summary, summary_test)
