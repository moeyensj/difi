import pytest
import numpy as np
import pandas as pd 
from pandas.testing import assert_frame_equal

from ..difi import analyzeLinkages
from .create_test_data import createTestDataSet

MAX_CONTAMINATION_PERCENTAGE = [0, 20, 40]
MIN_OBS = [5, 7, 9]
MIN_LINKAGE_LENGTHS = [3, 5, 7]

def test_analyzeLinkages_noClasses():
    ### Test analyzeLinkages when no classes are given

    for max_contamination_percentage in MAX_CONTAMINATION_PERCENTAGE:
        for min_obs in MIN_OBS:
            for min_linkage_length in MIN_LINKAGE_LENGTHS:

                print("min_obs: {}, min_linkage_length: {}, max_contamination_percentage: {}".format(
                        min_obs,
                        min_linkage_length,
                        max_contamination_percentage
                    )
                )
                
                # Generate test data set
                observations_test, all_truths_test, linkage_members_test, all_linkages_test, summary_test = createTestDataSet(
                    min_obs, 
                    min_linkage_length, 
                    max_contamination_percentage)

                # Analyze linkages
                all_linkages, all_truths, summary = analyzeLinkages(
                            observations_test, 
                            linkage_members_test, 
                            all_truths=None,
                            min_obs=min_obs,
                            contamination_percentage=max_contamination_percentage,
                            classes=None)
                
                # Compare to test data set 
                assert all_truths["findable"].isna().all() == True

                # We did not pass an all_truths data frame
                # so findability is not known to analyzeLinkages
                nan_cols = [
                    "completeness",
                    "findable",
                    "findable_found",
                    "findable_missed",
                    "not_findable_found",
                    "not_findable_missed",
                ]
                summary_test_ = summary_test.copy()
                summary_test_.loc[:, nan_cols] = np.NaN

                assert_frame_equal(all_linkages.fillna(-999), all_linkages_test.fillna(-999))
                assert_frame_equal(all_truths.loc[: , all_truths.columns != "findable"], all_truths_test.loc[: , all_truths_test.columns != "findable"])
                assert_frame_equal(summary.fillna(-999), summary_test_[summary_test_["class"] == "All"].fillna(-999))

                # Analyze linkages this time when all_truths is passed
                all_linkages, all_truths, summary = analyzeLinkages(
                            observations_test, 
                            linkage_members_test, 
                            all_truths=all_truths_test[["truth", "num_obs", "findable"]],
                            min_obs=min_obs,
                            contamination_percentage=max_contamination_percentage,
                            classes=None)

                assert_frame_equal(all_linkages.fillna(-999), all_linkages_test.fillna(-999))
                assert_frame_equal(all_truths, all_truths_test)
                assert_frame_equal(summary.fillna(-999), summary_test[summary_test["class"] == "All"].fillna(-999))

    return

def test_analyzeLinkages_withClassesColumn():
    ### Test analyzeLinkages when a class column is given

    for max_contamination_percentage in MAX_CONTAMINATION_PERCENTAGE:
        for min_obs in MIN_OBS:
            for min_linkage_length in MIN_LINKAGE_LENGTHS:

                print("min_obs: {}, min_linkage_length: {}, max_contamination_percentage: {}".format(
                        min_obs,
                        min_linkage_length,
                        max_contamination_percentage
                    )
                )
                
                # Generate test data set
                observations_test, all_truths_test, linkage_members_test, all_linkages_test, summary_test = createTestDataSet(
                    min_obs, 
                    min_linkage_length, 
                    max_contamination_percentage)

                # Analyze linkages
                all_linkages, all_truths, summary = analyzeLinkages(
                            observations_test, 
                            linkage_members_test, 
                            all_truths=None,
                            min_obs=min_obs,
                            contamination_percentage=max_contamination_percentage,
                            classes="class")
                
                # Compare to test data set 
                assert all_truths["findable"].isna().all() == True

                # We did not pass an all_truths data frame
                # so findability is not known to analyzeLinkages
                nan_cols = [
                    "completeness",
                    "findable",
                    "findable_found",
                    "findable_missed",
                    "not_findable_found",
                    "not_findable_missed",
                ]
                summary_test_ = summary_test.copy()
                summary_test_.loc[:, nan_cols] = np.NaN

                assert_frame_equal(all_linkages.fillna(-999), all_linkages_test.fillna(-999))
                assert_frame_equal(all_truths.loc[: , all_truths.columns != "findable"], all_truths_test.loc[: , all_truths_test.columns != "findable"])
                assert_frame_equal(summary.fillna(-999), summary_test_.fillna(-999))

                # Analyze linkages this time when all_truths is passed
                all_linkages, all_truths, summary = analyzeLinkages(
                            observations_test, 
                            linkage_members_test, 
                            all_truths=all_truths_test[["truth", "num_obs", "findable"]],
                            min_obs=min_obs,
                            contamination_percentage=max_contamination_percentage,
                            classes="class")

                assert_frame_equal(all_linkages.fillna(-999), all_linkages_test.fillna(-999))
                assert_frame_equal(all_truths, all_truths_test)
                assert_frame_equal(summary.fillna(-999), summary_test.fillna(-999))

    return

def test_analyzeLinkages_withClassesDictionary():
    ### Test analyzeLinkages when a class dictionary is given

    for max_contamination_percentage in MAX_CONTAMINATION_PERCENTAGE:
        for min_obs in MIN_OBS:
            for min_linkage_length in MIN_LINKAGE_LENGTHS:

                print("min_obs: {}, min_linkage_length: {}, max_contamination_percentage: {}".format(
                        min_obs,
                        min_linkage_length,
                        max_contamination_percentage
                    )
                )
                
                # Generate test data set
                observations_test, all_truths_test, linkage_members_test, all_linkages_test, summary_test = createTestDataSet(
                    min_obs, 
                    min_linkage_length, 
                    max_contamination_percentage)

                classes = {}
                for c in ["blue", "red", "green"]:
                    classes[c] = observations_test[observations_test["truth"].str.contains(c)]["truth"].unique()


                # Analyze linkages
                all_linkages, all_truths, summary = analyzeLinkages(
                            observations_test, 
                            linkage_members_test, 
                            all_truths=None,
                            min_obs=min_obs,
                            contamination_percentage=max_contamination_percentage,
                            classes=classes)
                
                # Compare to test data set 
                assert all_truths["findable"].isna().all() == True

                # We did not pass an all_truths data frame
                # so findability is not known to analyzeLinkages
                nan_cols = [
                    "completeness",
                    "findable",
                    "findable_found",
                    "findable_missed",
                    "not_findable_found",
                    "not_findable_missed",
                ]
                summary_test_ = summary_test.copy()
                summary_test_.loc[:, nan_cols] = np.NaN

                assert_frame_equal(all_linkages.fillna(-999), all_linkages_test.fillna(-999))
                assert_frame_equal(all_truths.loc[: , all_truths.columns != "findable"], all_truths_test.loc[: , all_truths_test.columns != "findable"])
                assert_frame_equal(summary.fillna(-999), summary_test_.fillna(-999))

                # Analyze linkages this time when all_truths is passed
                all_linkages, all_truths, summary = analyzeLinkages(
                            observations_test, 
                            linkage_members_test, 
                            all_truths=all_truths_test[["truth", "num_obs", "findable"]],
                            min_obs=min_obs,
                            contamination_percentage=max_contamination_percentage,
                            classes=classes)

                assert_frame_equal(all_linkages.fillna(-999), all_linkages_test.fillna(-999))
                assert_frame_equal(all_truths, all_truths_test)
                assert_frame_equal(summary.fillna(-999), summary_test.fillna(-999))

    return

def test_analyzeLinkages_emptyLinkageMembers():
    ### Test analyzeLinkages when a class dictionary is given
    min_obs = 5
    min_linkage_length = 5
    max_contamination_percentage = 30.

    # Generate test data set
    observations_test, all_truths_test, linkage_members_test, all_linkages_test, summary_test = createTestDataSet(
        min_obs, 
        min_linkage_length, 
        max_contamination_percentage
    )
    
    # Make expected linkage_members empty
    linkage_members_test.drop(
        linkage_members_test.index, 
        inplace=True
    )
    
    # Make expected all_linkages empty
    all_linkages_test.drop(
        all_linkages_test.index, 
        inplace=True
    )
    
    # Set all_truth columns to 0 since none should be 
    # retrieved without linkages present
    all_truths_cols = [
        "found_pure",
        "found_partial",
        "found",
        "pure",
        "pure_complete",
        "partial",
        "partial_contaminant",
        "mixed",
        "obs_in_pure",
        "obs_in_pure_complete",
        "obs_in_partial",
        "obs_in_partial_contaminant",
        "obs_in_mixed",
    ]
    all_truths_test.loc[:, all_truths_cols] = 0
    
    # Set summary columns to 0 since none should be 
    # retrieved without linkages present
    summary_cols = [
        "found", 
        "linkages", 
        "found_pure_linkages",
        "found_partial_linkages",
        "pure_linkages",
        "pure_complete_linkages",
        "partial_linkages",
        "partial_contaminant_linkages",
        "mixed_linkages",
        "unique_in_pure_linkages",
        "unique_in_pure_complete_linkages",
        "unique_in_pure_linkages_only",
        "unique_in_partial_linkages_only",
        "unique_in_pure_and_partial_linkages",
        "unique_in_partial_linkages",
        "unique_in_partial_contaminant_linkages",
        "unique_in_mixed_linkages",
        "obs_in_pure_linkages",
        "obs_in_pure_complete_linkages",
        "obs_in_partial_linkages",
        "obs_in_partial_contaminant_linkages",
        "obs_in_mixed_linkages"
    ]
    summary_test.loc[:, summary_cols] = 0

    # Completeness is a float so set it accordingly
    summary_test.loc[~summary_test["completeness"].isna(), "completeness"] = 0.0
    
    # Those objects that should be found are now missed without any linkages 
    # present so update the summary dataframe accordingly
    summary_test.loc[:, "findable_missed"] = summary_test["findable_found"]
    for c in ["findable_found", "not_findable_found"]:
        summary_test.loc[:, c] = 0

    classes = {}
    for c in ["blue", "red", "green"]:
        classes[c] = observations_test[observations_test["truth"].str.contains(c)]["truth"].unique()

    # Analyze linkages
    all_linkages, all_truths, summary = analyzeLinkages(
                observations_test, 
                linkage_members_test, 
                all_truths=None,
                min_obs=min_obs,
                contamination_percentage=max_contamination_percentage,
                classes=classes)
    
    # Compare to test data set 
    assert all_truths["findable"].isna().all() == True

    # We did not pass an all_truths data frame
    # so findability is not known to analyzeLinkages
    nan_cols = [
        "completeness",
        "findable",
        "findable_found",
        "findable_missed",
        "not_findable_found",
        "not_findable_missed",
    ]
    summary_test_ = summary_test.copy()
    summary_test_.loc[:, nan_cols] = np.NaN

    assert_frame_equal(all_linkages.fillna(-999), all_linkages_test.fillna(-999))
    assert_frame_equal(all_truths.loc[: , all_truths.columns != "findable"], all_truths_test.loc[: , all_truths_test.columns != "findable"])
    assert_frame_equal(summary.fillna(-999), summary_test_.fillna(-999))

    # Analyze linkages this time when all_truths is passed
    all_linkages, all_truths, summary = analyzeLinkages(
                observations_test, 
                linkage_members_test, 
                all_truths=all_truths_test[["truth", "num_obs", "findable"]],
                min_obs=min_obs,
                contamination_percentage=max_contamination_percentage,
                classes=classes)

    assert_frame_equal(all_linkages.fillna(-999), all_linkages_test.fillna(-999))
    assert_frame_equal(all_truths, all_truths_test)
    assert_frame_equal(summary.fillna(-999), summary_test.fillna(-999))
    
    return summary

def test_analyzeLinkages_errors():
    ### Test analyzeLinkages when incorrect data products are given
    min_obs = 5
    min_linkage_length = 5
    max_contamination_percentage = 30.

    # Generate test data set
    observations_test, all_truths_test, linkage_members_test, all_linkages_test, summary_test = createTestDataSet(
        min_obs, 
        min_linkage_length, 
        max_contamination_percentage
    )
    
    # Make expected observations_test empty
    observations_test_ = observations_test.copy()
    observations_test_.drop(
        observations_test_.index, 
        inplace=True
    )

    # Check for ValueError when observations are empty
    with pytest.raises(ValueError):
        all_linkages, all_truths, summary = analyzeLinkages(
        observations_test_, 
        linkage_members_test, 
        #all_truths=all_truths_test[["truth", "num_obs", "findable"]],
        min_obs=min_obs,
        contamination_percentage=max_contamination_percentage,
        classes=None)

    with pytest.warns(UserWarning):
        all_linkages, all_truths, summary = analyzeLinkages(
        observations_test, 
        linkage_members_test, 
        all_truths=all_truths_test[["truth", "num_obs"]],
        min_obs=min_obs,
        contamination_percentage=max_contamination_percentage,
        classes=None)

    return