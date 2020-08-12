import pytest
import numpy as np
import pandas as pd

from ..utils import _checkColumnTypes
from ..utils import _checkColumnTypesEqual
from ..utils import _percentHandler


def test__checkColumnTypes():
    # Define a columnMapping dictionary
    columnMapping = {
        "obs_id" : "obsId",
        "linkage_id": "linkageId",
        "truth" : "name",
    }

    # List of columns in dataFrame
    cols = [
        columnMapping["obs_id"],
        columnMapping["linkage_id"],
        columnMapping["truth"],
    ]
    
    # Loop through each 'wrong' dtype and test that an error is returned
    for dtype in [int, float, np.int64, np.float64, np.int32, np.float32]:
        num = 10000
        truths = np.arange(1, num, dtype=dtype)
        obs_ids = np.arange(1, num, dtype=dtype)
        linkage_ids = np.arange(1, num, dtype=dtype)
        
        df = pd.DataFrame(
            np.vstack([obs_ids, linkage_ids, truths]).T, 
            columns=cols
        )

        with pytest.raises(TypeError):
            _checkColumnTypes(df, ["truth", "obs_id", "linkage_id"], columnMapping)
            
    # Convert to correct dtype and insure no errors are raised
    df[columnMapping["obs_id"]] = df[columnMapping["obs_id"]].astype(str)
    df[columnMapping["linkage_id"]] = df[columnMapping["linkage_id"]].astype(str)
    df[columnMapping["truth"]] = df[columnMapping["truth"]].astype(str)
    
    _checkColumnTypes(df, ["truth", "obs_id", "linkage_id"], columnMapping)
    
    return

def test__checkColumnTypesEqual():
    # Define a columnMapping dictionary
    columnMapping = {
        "obs_id" : "obsId",
        "linkage_id": "linkageId",
        "truth" : "name",
    }

    # List of columns in dataFrame
    cols = [
        columnMapping["obs_id"],
        columnMapping["linkage_id"],
        columnMapping["truth"],
    ]
    
    # Loop through each 'wrong' dtype combinations and test that an error is returned
    for dtype in [int, float, np.int64, np.float64, np.int32]:
        num = 10000
        truths = np.arange(1, num, dtype=dtype)
        obs_ids = np.arange(1, num, dtype=dtype)
        linkage_ids = np.arange(1, num, dtype=dtype)
        
        df1 = pd.DataFrame(
            np.vstack([obs_ids, linkage_ids, truths]).T, 
            columns=cols
        )

        truths = np.arange(1, num, dtype=np.float32)
        obs_ids = np.arange(1, num, dtype=np.float32)
        linkage_ids = np.arange(1, num, dtype=np.float32)

        df2 = pd.DataFrame(
            np.vstack([obs_ids, linkage_ids, truths]).T, 
            columns=cols
        )

        with pytest.raises(TypeError):
            _checkColumnTypesEqual(df1, df2, ["truth", "obs_id", "linkage_id"], columnMapping)
            
    # Convert to correct dtype and insure no errors are raised
    df1[columnMapping["obs_id"]] = df1[columnMapping["obs_id"]].astype(str)
    df1[columnMapping["linkage_id"]] = df1[columnMapping["linkage_id"]].astype(str)
    df1[columnMapping["truth"]] = df1[columnMapping["truth"]].astype(str)
    
    _checkColumnTypesEqual(df1, df1, ["truth", "obs_id", "linkage_id"], columnMapping)
    
    return

def test__percentHandler():
    # If the denominator is 0, then _percentHandler should return np.NaN
    number = np.random.choice(np.arange(0, 10000))
    total_number = 0
    assert np.isnan(_percentHandler(number, total_number))
    
    # Test that a percentage is calculated correctly
    for i in range(1000):
        number = np.random.choice(np.arange(1, 10000))
        total_number = number = np.random.choice(np.arange(1, 10000))
        percent = 100. * number / total_number
        assert percent == _percentHandler(number, total_number)
        
    return