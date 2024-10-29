import numpy as np
import pandas as pd

__all__ = ["readLinkagesByLineFile"]


def readLinkagesByLineFile(
    linkages_file: str,
    linkage_id_start: int = 1,
) -> pd.DataFrame:
    """
    Reads a file that contains linkages where each linkage is written in terms of its
    observations line by line.

    Example:
    137541512 137543165 137615070 137620728 138216303 138216866 138221227
    137541512 137543165 137615070 137620728 138216303 138216866 138221227 144513728 144533645
    137541512 137543165 137615070 137620728 138216303 138216866 138221227 144513728 144533645 146991832
    137541512 137543165 137615070 137620728 138216303 138216866 138221227 144514371 144534274
    137541512 137543165 137615070 137620728 142747928 142763154
    137541512 137543165 137615070 137620728 142748009 142763229
    137541512 137543165 137615070 137620728 142748009 142763229 144513839 144533746
    137541512 137543165 137615070 137620728 142748120 142763338
    137541512 137543165 137615070 137620728 142748305 142763529
    137541512 137543165 137615070 137620728 142748337 142763570

    Parameters
    ----------
    linkages_file : str
        Path the linkages file that needs to be converted.
    linkage_id_start = 1
        Number at which to start the linkage ID count.
        [Default = 1]

    Returns
    -------
    linkage_members : `~pandas.DataFrame`
        DataFrame with two columns: the linkage ID and a second column with one row
        per observation ID.
    """
    # Read initial file
    linkages = pd.read_table(linkages_file, header=None, names=["obs_id"])

    # Make array of linkage IDs
    linkage_ids = np.arange(linkage_id_start, linkage_id_start + len(linkages), dtype=int)

    # Split each linkage into its different observation IDs
    linkage_list = linkages["obs_id"].str.split(" ").tolist()

    # Build initial DataFrame
    linkage_members = pd.DataFrame(
        pd.DataFrame(linkage_list, index=linkage_ids).stack(),
        columns=["obs_id"],
    )

    # Reset index
    linkage_members.reset_index(1, drop=True, inplace=True)

    # Make linkage_id its own column
    linkage_members["linkage_id"] = linkage_members.index

    # Re-arrange column order
    linkage_members = linkage_members[["linkage_id", "obs_id"]]

    # Not all linkages have the same number of detections, empty detections needs to be dropped
    linkage_members["obs_id"].replace("", np.nan, inplace=True)
    linkage_members.dropna(inplace=True)
    linkage_members.reset_index(drop=True, inplace=True)
    linkage_members["linkage_id"] = linkage_members["linkage_id"].astype(str)
    linkage_members["obs_id"] = linkage_members["obs_id"].astype(str)

    return linkage_members
