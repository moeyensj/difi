import pytest
import numpy as np
import pandas as pd

from difi import _checkColumnTypes

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