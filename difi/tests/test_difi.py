import os
import math
import string
import random
import numpy as np
import pandas as pd 
from pandas.util.testing import assert_frame_equal

from difi import analyzeLinkages

def test_analyzeLinkages_fromFile():
    # Load sample input
    linkageMembers = pd.read_csv(os.path.join(os.path.dirname(__file__), "linkageMembers.txt"), sep=" ", index_col=False)
    observations = pd.read_csv(os.path.join(os.path.dirname(__file__), "observations.txt"), sep=" ", index_col=False)
    
    # Load solution
    allLinkages_solution = pd.read_csv(os.path.join(os.path.dirname(__file__), "allLinkages_solution.txt"), sep=" ", index_col=False)
    allTruths_solution = pd.read_csv(os.path.join(os.path.dirname(__file__), "allTruths_solution.txt"), sep=" ", index_col=False)
    
    allLinkages_test, allTruths_test, summary_test = analyzeLinkages(observations, 
                                                                     linkageMembers,
                                                                     minObs=5, 
                                                                     contaminationThreshold=0.2)
    
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
    # Create randomly sized pure cluster
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
    linkageMembers = pd.DataFrame({
        columnMapping["linkage_id"] : linkage_ids,
        columnMapping["obs_id"] : obs_ids
                                  })

    # Create the observations dataframe
    observations = pd.DataFrame({
        columnMapping["obs_id"] : obs_ids,
        columnMapping["truth"] : truth,
    })

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

    # Test that the number of clusters is as expected
    assert allLinkages_test["pure"].values[0] == 1
    assert allLinkages_test["partial"].values[0] == 0
    assert allLinkages_test["mixed"].values[0] == 0
    assert math.isnan(allLinkages_test["contamination"].values[0])

    # Test the linked truth is the name
    assert allLinkages_test["linked_truth"].values[0] == name

    ### Test allTruths

    # Test number of clusters is consistent
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
        'num_mixed_clusters' : [0],
        'num_total_clusters' : [1]
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
        'num_mixed_clusters',
        'num_total_clusters']]

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

    # Test that the number of clusters is as expected
    assert allLinkages_test["pure"].values[0] == 1
    assert allLinkages_test["partial"].values[0] == 0
    assert allLinkages_test["mixed"].values[0] == 0
    assert math.isnan(allLinkages_test["contamination"].values[0])

    # Test the linked truth is the name
    assert allLinkages_test["linked_truth"].values[0] == name

    ### Test allTruths

    # Test number of clusters is consistent
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
        'num_mixed_clusters' : [0],
        'num_total_clusters' : [1]
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
        'num_mixed_clusters',
        'num_total_clusters']]

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

    # Test that the number of clusters is as expected
    assert allLinkages_test["pure"].values[0] == 1
    assert allLinkages_test["partial"].values[0] == 0
    assert allLinkages_test["mixed"].values[0] == 0
    assert math.isnan(allLinkages_test["contamination"].values[0])
    
    # Test the linked truth is the name
    assert allLinkages_test["linked_truth"].values[0] == name

    ### Test allTruths

    # Test number of clusters is consistent
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
        'num_mixed_clusters' : [0],
        'num_total_clusters' : [1]
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
        'num_mixed_clusters',
        'num_total_clusters']]

    assert_frame_equal(summary, summary_test)

def test_analyzeLinkages_singleObject_missed():    
    # Create randomly sized pure cluster
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
    linkageMembers = pd.DataFrame({
        columnMapping["linkage_id"] : linkage_ids,
        columnMapping["obs_id"] : obs_ids
                                  })

    # Create the observations dataframe
    observations = pd.DataFrame({
        columnMapping["obs_id"] : obs_ids,
        columnMapping["truth"] : truth,
    })
    
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

    # Test that the number of clusters is as expected
    assert allLinkages_test["pure"].values[0] == 0
    assert allLinkages_test["partial"].values[0] == 0
    assert allLinkages_test["mixed"].values[0] == 1
    assert math.isnan(allLinkages_test["contamination"].values[0])

    # Test the linked truth is nan
    assert math.isnan(allLinkages_test["linked_truth"].values[0])
    
    ### Test allTruths

    # Test number of clusters is consistent
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
        'num_mixed_clusters' : [1],
        'num_total_clusters' : [1]
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
        'num_mixed_clusters',
        'num_total_clusters']]

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

    # Test that the number of clusters is as expected
    assert allLinkages_test["pure"].values[0] == 0
    assert allLinkages_test["partial"].values[0] == 0
    assert allLinkages_test["mixed"].values[0] == 1
    assert math.isnan(allLinkages_test["contamination"].values[0])

    # Test the linked truth is nan
    assert math.isnan(allLinkages_test["linked_truth"].values[0])
    
    ### Test allTruths

    # Test number of clusters is consistent
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
        'num_mixed_clusters' : [1],
        'num_total_clusters' : [1]
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
        'num_mixed_clusters',
        'num_total_clusters']]

    assert_frame_equal(summary, summary_test)

    