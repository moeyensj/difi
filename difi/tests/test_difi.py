import pytest
import numpy as np
import pandas as pd 
from pandas.testing import assert_frame_equal

from ..difi import analyzeLinkages
from .create_test_data import createTestDataSet

MAX_CONTAMINATION_PERCENTAGE = [20]
MIN_OBS = [5]
MIN_LINKAGE_LENGTHS = [5]

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
