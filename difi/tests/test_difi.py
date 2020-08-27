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
    for max_contamination_percentage in MAX_CONTAMINATION_PERCENTAGE:
        for min_obs in MIN_OBS:
            for min_linkage_length in MIN_LINKAGE_LENGTHS:
                
                # Generate test data set
                observations_test, all_truths_test, linkage_members_test, all_linkages_test, summary_test = createTestDataSet(
                    min_obs, 
                    min_linkage_length, 
                    max_contamination_percentage)

                # Analyze linkages
                allLinkages, allTruths, summary = analyzeLinkages(
                            observations_test, 
                            linkage_members_test, 
                            all_truths=None,
                            min_obs=min_obs,
                            contamination_percentage=max_contamination_percentage,
                            classes=None)
                
                # Compare to test data set 
                assert allTruths["findable"].isna().all() == True

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

                pd.testing.assert_frame_equal(allLinkages.fillna(-999), all_linkages_test.fillna(-999))
                pd.testing.assert_frame_equal(allTruths.loc[: , allTruths.columns != "findable"], all_truths_test.loc[: , all_truths_test.columns != "findable"])
                pd.testing.assert_frame_equal(summary.fillna(-999), summary_test_[summary_test_["class"] == "All"].fillna(-999))

    return