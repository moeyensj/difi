import pytest
import numpy as np
import pandas as pd 
from pandas.testing import assert_frame_equal

from ..cifi import analyzeObservations
from .create_test_data import createTestDataSet

MIN_OBS = range(9, 10)

def test_analyzeObservations_noClasses():
    ### Test analyzeObservations when no truth classes are given
    
    # Create test data
    for min_obs in MIN_OBS:
        # Generate test data set
        observations_test, all_truths_test, linkage_members_test, all_linkages_test, summary_test = createTestDataSet(
            min_obs, 
            5, 
            20)

        # Build the all_truths and summary data frames
        all_truths, summary = analyzeObservations(observations_test, min_obs=min_obs, classes=None)
        
        # Assert equality among the returned columns
        assert_frame_equal(all_truths, all_truths_test[["truth", "num_obs", "findable"]])
        assert_frame_equal(summary, summary_test[summary_test["class"] == "All"][["class", "num_members", "num_obs", "findable"]])
        
    return

def test_analyzeObservations_withClassesColumn():
    ### Test analyzeObservations when a class column is given
    
    # Create test data
    for min_obs in MIN_OBS:
        # Generate test data set
        observations_test, all_truths_test, linkage_members_test, all_linkages_test, summary_test = createTestDataSet(
            min_obs, 
            5, 
            20)

        # Build the all_truths and summary data frames
        all_truths, summary = analyzeObservations(observations_test, min_obs=min_obs, classes="class")
        
        # Assert equality among the returned columns
        assert_frame_equal(all_truths, all_truths_test[["truth", "num_obs", "findable"]])
        assert_frame_equal(summary, summary_test[["class", "num_members", "num_obs", "findable"]])
        
    return

def test_analyzeObservations_withClassesDictionary():
    ### Test analyzeObservations when a class dictionary is given
    
    # Create test data
    for min_obs in MIN_OBS:
        # Generate test data set
        observations_test, all_truths_test, linkage_members_test, all_linkages_test, summary_test = createTestDataSet(
            min_obs, 
            5, 
            20)

        classes = {}
        for c in ["blue", "red", "green"]:
            classes[c] = observations_test[observations_test["truth"].str.contains(c)]["truth"].unique()

        # Build the all_truths and summary data frames
        all_truths, summary = analyzeObservations(observations_test, min_obs=min_obs, classes=classes)
        
        # Assert equality among the returned columns
        assert_frame_equal(all_truths, all_truths_test[["truth", "num_obs", "findable"]])
        assert_frame_equal(summary, summary_test[["class", "num_members", "num_obs", "findable"]])
        
    return

def test_analyzeObservations_noObservations():
    ### Test analyzeObservations when the observations data frame is empty
    
    observations_test, all_truths_test, linkage_members_test, all_linkages_test, summary_test = createTestDataSet(
        5, 
        5, 
        20)

    observations_test = observations_test.drop(observations_test.index)

    with pytest.raises(ValueError):

        # Build the all_truths and summary data frames
        all_truths, summary = analyzeObservations(observations_test, min_obs=5, classes=None)
    
    return
