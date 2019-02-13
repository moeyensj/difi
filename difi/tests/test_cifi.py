import os
import math
import string
import random
import pytest
import numpy as np
import pandas as pd 
from pandas.util.testing import assert_frame_equal

from difi import analyzeObservations

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

    
