import math
import string
import random
import pytest
import numpy as np
import pandas as pd 
from pandas.testing import assert_frame_equal

from ..cifi import analyzeObservations

def test_analyzeObservations_singleObject():
    # Case 1a: Single object that should be findable

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

    # Create the observations dataframe
    observations = pd.DataFrame(
        {
            columnMapping["obs_id"] : obs_ids,
            columnMapping["truth"] : truth,
        },
    )

    # Run analysis for case when it should found
    allTruths_test, summary_test = analyzeObservations(observations,    
                                                       minObs=num_obs - 1, 
                                                       columnMapping=columnMapping, 
                                                       verbose=True)

    ### Test allTruths
    # Test the length of allTruths is one
    assert len(allTruths_test) == 1

    # Test number of linkages is consistent
    assert allTruths_test["findable"].values[0] == 1

    # Test that the truth is the name
    assert allTruths_test[columnMapping["truth"]].values[0] == name

    ### Test summary
    summary = pd.DataFrame({
        "num_unique_truths" : [1], 
        "num_unique_known_truths" : [1],
        "num_unique_known_truths_findable" : [1],
        "num_known_truth_observations": [num_obs],
        "num_unknown_truth_observations": [0],
        "num_false_positive_observations": [0],
        "percent_known_truth_observations": [100.0],
        "percent_unknown_truth_observations": [0.0],
        "percent_false_positive_observations": [0.0]
    })
    
    # Re-arange columns in case order is changed (python 3.5 and earlier)
    summary = summary[[
        "num_unique_truths",
        "num_unique_known_truths",
        "num_unique_known_truths_findable",
        "num_known_truth_observations",
        "num_unknown_truth_observations",
        "num_false_positive_observations",
        "percent_known_truth_observations",
        "percent_unknown_truth_observations",
        "percent_false_positive_observations"]]

    assert_frame_equal(summary, summary_test)

    # Case 1b: Single object that should not be findable

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

    # Create the observations dataframe
    observations = pd.DataFrame(
        {
            columnMapping["obs_id"] : obs_ids,
            columnMapping["truth"] : truth,
        },
    )

    # Run analysis for case when it should found
    allTruths_test, summary_test = analyzeObservations(observations,    
                                                       minObs=num_obs + 1, 
                                                       columnMapping=columnMapping, 
                                                       verbose=True)

    ### Test allTruths
    # Test the length of allTruths is one
    assert len(allTruths_test) == 1

    # Test number of linkages is consistent
    assert allTruths_test["findable"].values[0] == 0

    # Test that the truth is the name
    assert allTruths_test[columnMapping["truth"]].values[0] == name

    ### Test summary
    summary = pd.DataFrame({
        "num_unique_truths" : [1], 
        "num_unique_known_truths" : [1],
        "num_unique_known_truths_findable" : [0],
        "num_known_truth_observations": [num_obs],
        "num_unknown_truth_observations": [0],
        "num_false_positive_observations": [0],
        "percent_known_truth_observations": [100.0],
        "percent_unknown_truth_observations": [0.0],
        "percent_false_positive_observations": [0.0]
    })
    
    # Re-arange columns in case order is changed (python 3.5 and earlier)
    summary = summary[[
        "num_unique_truths",
        "num_unique_known_truths",
        "num_unique_known_truths_findable",
        "num_known_truth_observations",
        "num_unknown_truth_observations",
        "num_false_positive_observations",
        "percent_known_truth_observations",
        "percent_unknown_truth_observations",
        "percent_false_positive_observations"]]

    assert_frame_equal(summary, summary_test)

    # Case 1c: Single object that should be findable, marked as unknown

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

    # Create the observations dataframe
    observations = pd.DataFrame(
        {
            columnMapping["obs_id"] : obs_ids,
            columnMapping["truth"] : truth,
        },
    )

    # Run analysis for case when it should found
    allTruths_test, summary_test = analyzeObservations(observations,    
                                                       minObs=num_obs - 1, 
                                                       unknownIDs=[name],
                                                       columnMapping=columnMapping, 
                                                       verbose=True)

    ### Test allTruths
    # Test the length of allTruths is one
    assert len(allTruths_test) == 1

    # Test number of linkages is consistent
    assert allTruths_test["findable"].values[0] == 0

    # Test that the truth is the name
    assert allTruths_test[columnMapping["truth"]].values[0] == name

    ### Test summary
    summary = pd.DataFrame({
        "num_unique_truths" : [1], 
        "num_unique_known_truths" : [0],
        "num_unique_known_truths_findable" : [0],
        "num_known_truth_observations": [0],
        "num_unknown_truth_observations": [num_obs],
        "num_false_positive_observations": [0],
        "percent_known_truth_observations": [0.0],
        "percent_unknown_truth_observations": [100.0],
        "percent_false_positive_observations": [0.0]
    })
    
    # Re-arange columns in case order is changed (python 3.5 and earlier)
    summary = summary[[
        "num_unique_truths",
        "num_unique_known_truths",
        "num_unique_known_truths_findable",
        "num_known_truth_observations",
        "num_unknown_truth_observations",
        "num_false_positive_observations",
        "percent_known_truth_observations",
        "percent_unknown_truth_observations",
        "percent_false_positive_observations"]]

    assert_frame_equal(summary, summary_test)

    # Case 1d: Single object that should not be findable, marked as unknown

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

    # Create the observations dataframe
    observations = pd.DataFrame(
        {
            columnMapping["obs_id"] : obs_ids,
            columnMapping["truth"] : truth,
        },
    )

    # Run analysis for case when it should found
    allTruths_test, summary_test = analyzeObservations(observations,    
                                                       minObs=num_obs + 1, 
                                                       unknownIDs=[name],
                                                       columnMapping=columnMapping, 
                                                       verbose=True)

    ### Test allTruths
    # Test the length of allTruths is one
    assert len(allTruths_test) == 1

    # Test number of linkages is consistent
    assert allTruths_test["findable"].values[0] == 0

    # Test that the truth is the name
    assert allTruths_test[columnMapping["truth"]].values[0] == name

    ### Test summary
    summary = pd.DataFrame({
        "num_unique_truths" : [1], 
        "num_unique_known_truths" : [0],
        "num_unique_known_truths_findable" : [0],
        "num_known_truth_observations": [0],
        "num_unknown_truth_observations": [num_obs],
        "num_false_positive_observations": [0],
        "percent_known_truth_observations": [0.0],
        "percent_unknown_truth_observations": [100.0],
        "percent_false_positive_observations": [0.0]
    })
    
    # Re-arange columns in case order is changed (python 3.5 and earlier)
    summary = summary[[
        "num_unique_truths",
        "num_unique_known_truths",
        "num_unique_known_truths_findable",
        "num_known_truth_observations",
        "num_unknown_truth_observations",
        "num_false_positive_observations",
        "percent_known_truth_observations",
        "percent_unknown_truth_observations",
        "percent_false_positive_observations"]]

    assert_frame_equal(summary, summary_test)

    # Case 1e: Single object that should be findable, marked as false positive

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

    # Create the observations dataframe
    observations = pd.DataFrame(
        {
            columnMapping["obs_id"] : obs_ids,
            columnMapping["truth"] : truth,
        },
    )

    # Run analysis for case when it should found
    allTruths_test, summary_test = analyzeObservations(observations,    
                                                       minObs=num_obs - 1, 
                                                       falsePositiveIDs=[name],
                                                       columnMapping=columnMapping, 
                                                       verbose=True)

    ### Test allTruths
    # Test the length of allTruths is one
    assert len(allTruths_test) == 1

    # Test number of linkages is consistent
    assert allTruths_test["findable"].values[0] == 0

    # Test that the truth is the name
    assert allTruths_test[columnMapping["truth"]].values[0] == name

    ### Test summary
    summary = pd.DataFrame({
        "num_unique_truths" : [1], 
        "num_unique_known_truths" : [0],
        "num_unique_known_truths_findable" : [0],
        "num_known_truth_observations": [0],
        "num_unknown_truth_observations": [0],
        "num_false_positive_observations": [num_obs],
        "percent_known_truth_observations": [0.0],
        "percent_unknown_truth_observations": [0.0],
        "percent_false_positive_observations": [100.0]
    })
    
    # Re-arange columns in case order is changed (python 3.5 and earlier)
    summary = summary[[
        "num_unique_truths",
        "num_unique_known_truths",
        "num_unique_known_truths_findable",
        "num_known_truth_observations",
        "num_unknown_truth_observations",
        "num_false_positive_observations",
        "percent_known_truth_observations",
        "percent_unknown_truth_observations",
        "percent_false_positive_observations"]]

    assert_frame_equal(summary, summary_test)


def test_analyzeObservations_multiObject():   
    # Randomly assign names to the columns
    columnMapping = {
        "obs_id" : ''.join([random.choice(string.ascii_letters + string.digits) for n in range(np.random.randint(64))]),
        "truth" : ''.join([random.choice(string.ascii_letters + string.digits) for n in range(np.random.randint(64))])
    }

    observations_list = []

    # Create three fake sources and their observations
    prev_obs = 0
    min_obs = []
    names = []
    for i in range(0, 3):
        num_obs = np.random.randint(prev_obs + 2, prev_obs + 50)
        name = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(np.random.randint(64))])
        truth = [name for i in range(num_obs)]
        obs_ids = np.arange(prev_obs + 1, prev_obs + num_obs + 1)

        # Create the observations dataframe
        observations = pd.DataFrame(
            {
                columnMapping["obs_id"] : obs_ids,
                columnMapping["truth"] : truth,
            },
            dtype=str
        )

        observations_list.append(observations)

        prev_obs += num_obs
        min_obs.append(num_obs)
        names.append(name)

    # Concatenate dataframes and grab number of observations for each
    # unique object
    observations = pd.concat(observations_list)
    minObs = np.array(min_obs)
    names = np.array(names)

    # Case 2a: All objects are findable

    # Run analysis for case when it should found
    allTruths_test, summary_test = analyzeObservations(observations,    
                                                       minObs=minObs.min() - 1,
                                                       columnMapping=columnMapping, 
                                                       verbose=True)

    ### Test allTruths
    # Test the length of allTruths is three
    assert len(allTruths_test) == 3

    # Test number of linkages is consistent
    np.testing.assert_equal(allTruths_test["findable"].values, np.array([1, 1, 1]))

    # Test that the truth is the name
    np.testing.assert_equal(set(allTruths_test[columnMapping["truth"]].values), set(names))

    ### Test summary
    summary = pd.DataFrame({
        "num_unique_truths" : [3], 
        "num_unique_known_truths" : [3],
        "num_unique_known_truths_findable" : [3],
        "num_known_truth_observations": [np.sum(minObs)],
        "num_unknown_truth_observations": [0],
        "num_false_positive_observations": [0],
        "percent_known_truth_observations": [100.0],
        "percent_unknown_truth_observations": [0.0],
        "percent_false_positive_observations": [0.0]
    })
    
    # Re-arange columns in case order is changed (python 3.5 and earlier)
    summary = summary[[
        "num_unique_truths",
        "num_unique_known_truths",
        "num_unique_known_truths_findable",
        "num_known_truth_observations",
        "num_unknown_truth_observations",
        "num_false_positive_observations",
        "percent_known_truth_observations",
        "percent_unknown_truth_observations",
        "percent_false_positive_observations"]]

    assert_frame_equal(summary, summary_test)

    # Case 2b: Last two objects are findable

    # Run analysis for case when it should found
    allTruths_test, summary_test = analyzeObservations(observations,    
                                                       minObs=minObs[1] - 1,
                                                       columnMapping=columnMapping, 
                                                       verbose=True)

    ### Test allTruths
    # Test the length of allTruths is three
    assert len(allTruths_test) == 3

    # Test number of linkages is consistent
    np.testing.assert_equal(allTruths_test["findable"].values, np.array([1, 1, 0]))

    # Test that the truth is the name
    np.testing.assert_equal(set(allTruths_test[columnMapping["truth"]].values), set(names))

    ### Test summary
    summary = pd.DataFrame({
        "num_unique_truths" : [3], 
        "num_unique_known_truths" : [3],
        "num_unique_known_truths_findable" : [2],
        "num_known_truth_observations": [np.sum(minObs)],
        "num_unknown_truth_observations": [0],
        "num_false_positive_observations": [0],
        "percent_known_truth_observations": [100.0],
        "percent_unknown_truth_observations": [0.0],
        "percent_false_positive_observations": [0.0]
    })
    
    # Re-arange columns in case order is changed (python 3.5 and earlier)
    summary = summary[[
        "num_unique_truths",
        "num_unique_known_truths",
        "num_unique_known_truths_findable",
        "num_known_truth_observations",
        "num_unknown_truth_observations",
        "num_false_positive_observations",
        "percent_known_truth_observations",
        "percent_unknown_truth_observations",
        "percent_false_positive_observations"]]

    assert_frame_equal(summary, summary_test)

    # Case 2c: No objects are findable

    # Run analysis for case when it should found
    allTruths_test, summary_test = analyzeObservations(observations,    
                                                       minObs=minObs[-1] + 1,
                                                       columnMapping=columnMapping, 
                                                       verbose=True)

    ### Test allTruths
    # Test the length of allTruths is three
    assert len(allTruths_test) == 3

    # Test number of linkages is consistent
    np.testing.assert_equal(allTruths_test["findable"].values, np.array([0, 0, 0]))

    # Test that the truth is the name
    np.testing.assert_equal(set(allTruths_test[columnMapping["truth"]].values), set(names))

    ### Test summary
    summary = pd.DataFrame({
        "num_unique_truths" : [3], 
        "num_unique_known_truths" : [3],
        "num_unique_known_truths_findable" : [0],
        "num_known_truth_observations": [np.sum(minObs)],
        "num_unknown_truth_observations": [0],
        "num_false_positive_observations": [0],
        "percent_known_truth_observations": [100.0],
        "percent_unknown_truth_observations": [0.0],
        "percent_false_positive_observations": [0.0]
    })
    
    # Re-arange columns in case order is changed (python 3.5 and earlier)
    summary = summary[[
        "num_unique_truths",
        "num_unique_known_truths",
        "num_unique_known_truths_findable",
        "num_known_truth_observations",
        "num_unknown_truth_observations",
        "num_false_positive_observations",
        "percent_known_truth_observations",
        "percent_unknown_truth_observations",
        "percent_false_positive_observations"]]

    assert_frame_equal(summary, summary_test)

    # Case 2d: Two known truths are findable, one object is an unknown truth

    # Run analysis for case when it should found
    allTruths_test, summary_test = analyzeObservations(observations,    
                                                       minObs=minObs[0] - 1,
                                                       unknownIDs=[names[-1]],
                                                       columnMapping=columnMapping, 
                                                       verbose=True)

    ### Test allTruths
    # Test the length of allTruths is three
    assert len(allTruths_test) == 3

    # Test number of linkages is consistent
    np.testing.assert_equal(allTruths_test["findable"].values, np.array([0, 1, 1]))

    # Test that the truth is the name
    np.testing.assert_equal(set(allTruths_test[columnMapping["truth"]].values), set(names))

    ### Test summary
    summary = pd.DataFrame({
        "num_unique_truths" : [3], 
        "num_unique_known_truths" : [2],
        "num_unique_known_truths_findable" : [2],
        "num_known_truth_observations": [np.sum(minObs) - minObs[-1]],
        "num_unknown_truth_observations": [minObs[-1]],
        "num_false_positive_observations": [0],
        "percent_known_truth_observations": [(np.sum(minObs) - minObs[-1]) / np.sum(minObs) * 100.],
        "percent_unknown_truth_observations": [(minObs[-1]) / np.sum(minObs) * 100.],
        "percent_false_positive_observations": [0.0]
    })
    
    # Re-arange columns in case order is changed (python 3.5 and earlier)
    summary = summary[[
        "num_unique_truths",
        "num_unique_known_truths",
        "num_unique_known_truths_findable",
        "num_known_truth_observations",
        "num_unknown_truth_observations",
        "num_false_positive_observations",
        "percent_known_truth_observations",
        "percent_unknown_truth_observations",
        "percent_false_positive_observations"]]

    assert_frame_equal(summary, summary_test)

    # Case 2e: Two known truths are findable, one object is an false positive truth

    # Run analysis for case when it should found
    allTruths_test, summary_test = analyzeObservations(observations,    
                                                       minObs=minObs[0] - 1,
                                                       falsePositiveIDs=[names[-1]],
                                                       columnMapping=columnMapping, 
                                                       verbose=True)

    ### Test allTruths
    # Test the length of allTruths is three
    assert len(allTruths_test) == 3

    # Test number of linkages is consistent
    np.testing.assert_equal(allTruths_test["findable"].values, np.array([0, 1, 1]))

    # Test that the truth is the name
    np.testing.assert_equal(set(allTruths_test[columnMapping["truth"]].values), set(names))

    ### Test summary
    summary = pd.DataFrame({
        "num_unique_truths" : [3], 
        "num_unique_known_truths" : [2],
        "num_unique_known_truths_findable" : [2],
        "num_known_truth_observations": [np.sum(minObs) - minObs[-1]],
        "num_unknown_truth_observations": [0],
        "num_false_positive_observations": [minObs[-1]],
        "percent_known_truth_observations": [(np.sum(minObs) - minObs[-1]) / np.sum(minObs) * 100.],
        "percent_unknown_truth_observations": [0.0],
        "percent_false_positive_observations": [(minObs[-1]) / np.sum(minObs) * 100.]
    })
    
    # Re-arange columns in case order is changed (python 3.5 and earlier)
    summary = summary[[
        "num_unique_truths",
        "num_unique_known_truths",
        "num_unique_known_truths_findable",
        "num_known_truth_observations",
        "num_unknown_truth_observations",
        "num_false_positive_observations",
        "percent_known_truth_observations",
        "percent_unknown_truth_observations",
        "percent_false_positive_observations"]]

    assert_frame_equal(summary, summary_test)

    
def test_analyzeObservations_emptyDataFrames():
    # Case 3a: Pass empty observations DataFramee

    # Randomly assign names to the columns
    columnMapping = {
        "obs_id" : ''.join([random.choice(string.ascii_letters + string.digits) for n in range(np.random.randint(64))]),
        "truth" : ''.join([random.choice(string.ascii_letters + string.digits) for n in range(np.random.randint(64))])
    }

    # Create the observations dataframe
    observations_empty = pd.DataFrame(columns=[
        columnMapping["obs_id"],
        columnMapping["truth"]
    ])

    # Test an error is raised when the observations dataframe is empy
    with pytest.raises(ValueError):
        allTruths_test, summary_test = analyzeObservations(observations_empty, 
                                                           unknownIDs=[],
                                                           falsePositiveIDs=[],
                                                           minObs=5, 
                                                           columnMapping=columnMapping, 
                                                           verbose=True)

    
