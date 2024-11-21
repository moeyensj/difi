import warnings
from typing import List, Tuple, Union

import numpy as np
import pandas as pd


def _classHandler(
    classes: Union[str, dict, None], dataframe: pd.DataFrame
) -> Tuple[List[str], List[List[str]]]:
    """
    Tests that the `classes` keyword argument is defined correctly.
    `classes` should one of the following:
        str : Name of the column in the dataframe which identifies
            the class of each object.
        dict : A dictionary with class names as keys and a list of unique
            objects belonging to each class as values.
        None : If there are no classes of objects.

    Parameters
    ----------
    classes : {str, dict, or None}
        Declares if the objects in a data frame have classes.
    dataframe : `~pandas.DataFrame`
        A pandas data frame containing a column of objects and optionally
        a column that specifies the class (if classes is a str).

    Returns
    -------
    class_list : list
        A list of class names.
    object_ids_list : list
        A list of the objects belonging to each class.

    Raises
    ------
    UserWarning : If not all objects in the dataframe are assigned a class
    """
    class_list = ["All"]
    object_ids_list = [[]]  # type: ignore
    unique_objects = []

    if classes is None:
        object_ids_list = [dataframe["object_id"].unique()]
        unique_objects = [object_ids_list[0]]

    elif isinstance(classes, str):
        if classes not in dataframe.columns:
            err = "Could not find class column ({}) in observations."
            raise ValueError(err.format(classes))
        else:
            for c in dataframe[~dataframe[classes].isna()][classes].unique():
                class_list.append(c)
                class_objects = dataframe[dataframe[classes].isin([c])]["object_id"].unique()
                unique_objects.append(class_objects)
                object_ids_list.append(class_objects)

        object_ids_list[0] = dataframe["object_id"].unique()

    elif isinstance(classes, dict):
        for c, t in classes.items():
            if len(np.unique(t)) != len(t):
                err = "Truths for class {} are not unique."
                raise ValueError(err.format(c))
            else:
                class_list.append(c)
                object_ids_list[0].append(t)
                unique_objects.append(t)
                if type(t) is list:
                    object_ids_list.append(np.array(t))
                else:
                    object_ids_list.append(t)

        object_ids_list[0] = np.hstack(object_ids_list[0])

    else:
        err = (
            "Classes should be one of:\n"
            "  str : Name of the column in the dataframe which\n"
            "        identifies the class of each object.\n"
            "  dict : A dictionary with class names as keys\n"
            "        and a list of unique objects belonging to each class\n"
            "        as values.\n"
            "  None : If there are no classes of objects."
        )
        raise ValueError(err)

    # Test that the unique objects are in fact unique
    unique_objects = np.concatenate(unique_objects)
    if not len(np.unique(unique_objects)) == len(unique_objects):
        err = "Some objects are duplicated in multiple classes."
        raise ValueError(err)

    if not dataframe["object_id"].isin(unique_objects).all():
        warning = (
            "Some objects do not have an assigned class.\n"
            "Unclassified objects have been added as 'Unclassified'."
        )

        unclassified = dataframe[~dataframe["object_id"].isin(unique_objects)]["object_id"].unique()
        class_list.append("Unclassified")
        object_ids_list.append(unclassified)
        object_ids_list[0] = np.concatenate([object_ids_list[0], unclassified])
        warnings.warn(warning, UserWarning)

    return class_list, object_ids_list
