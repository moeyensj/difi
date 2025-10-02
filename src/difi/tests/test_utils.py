import copy

import numpy as np
import pandas as pd
import pytest

from ..utils import _classHandler


def test__classHandler():
    # --- Test _classHandler for three different class arguments: None, dictionary and column
    # name in observations dataframe
    classes_dict = {}
    all_truths = []

    # Create list of expected outputs from _classHandler
    class_list_test = ["All"]
    truths_list_test = []

    # Add to the expected outputs and build the expected
    # input observations dataframe and classes dictionary
    for c in ["green", "blue", "red"]:
        truths = ["{}{:02d}".format(c, i) for i in range(10)]
        truths_list_test.append(truths)
        all_truths += truths
        classes_dict[c] = truths
        class_list_test.append(c)
    truths_list_test.insert(0, all_truths)

    observations = pd.DataFrame(
        {
            "obs_id": ["obs{:02d}".format(i) for i in range(len(all_truths))],
            "object_id": all_truths,
        }
    )
    for c in ["green", "blue", "red"]:
        observations.loc[observations["object_id"].isin(classes_dict[c]), "class"] = c

    # Test when no classes are given
    class_list, truths_list = _classHandler(None, observations)

    assert np.all(class_list[0] == class_list_test[0])
    assert np.all(truths_list[0] == truths_list_test[0])

    # Test when a class dictionary is given
    class_list, truths_list = _classHandler(classes_dict, observations)

    assert np.all(class_list == class_list_test)
    for t_i, t_test_i in zip(truths_list, truths_list_test):
        assert np.all(t_i == t_test_i)

    # Test when a class column is given
    class_list, truths_list = _classHandler("class", observations)

    assert np.all(class_list == class_list_test)
    for t_i, t_test_i in zip(truths_list, truths_list_test):
        assert np.all(t_i == t_test_i)

    return


def test__classHandler_errors():
    # --- Test _classHandler for error raises
    classes_dict = {}
    all_truths = []

    # Create list of expected outputs from _classHandler
    class_list_test = ["All"]
    truths_list_test = []

    # Add to the expected outputs and build the expected
    # input observations dataframe and classes dictionary
    for c in ["green", "blue", "red"]:
        truths = ["{}{:02d}".format(c, i) for i in range(10)]
        truths_list_test.append(truths)
        all_truths += truths
        classes_dict[c] = truths
        class_list_test.append(c)
    truths_list_test.insert(0, all_truths)

    observations = pd.DataFrame(
        {
            "obs_id": ["obs{:02d}".format(i) for i in range(len(all_truths))],
            "object_id": all_truths,
        }
    )
    for c in ["green", "blue", "red"]:
        observations.loc[observations["object_id"].isin(classes_dict[c]), "class"] = c

    # Test for ValueError when an unsupported class argument is given
    with pytest.raises(ValueError):
        class_list, truths_list = _classHandler([], observations)

    # Test for ValueError when a truth appears twice for the same class
    with pytest.raises(ValueError):
        classes_dict_ = copy.deepcopy(classes_dict)
        classes_dict_["green"].append(classes_dict_["green"][-1])

        class_list, truths_list = _classHandler(classes_dict_, observations)

    # Test for ValueError when an incorrect column name is given
    with pytest.raises(ValueError):
        class_list, truths_list = _classHandler("abc", observations)

    # Test for ValueError when a truth appears in more than two classes
    classes_dict["blue"].append(classes_dict["green"][-1])
    with pytest.raises(ValueError):
        class_list, truths_list = _classHandler(classes_dict, observations)

    return


def test__classHandler_warnings():
    # --- Test _classHandler for warnings
    classes_dict = {}
    all_truths = []

    # Create list of expected outputs from _classHandler
    class_list_test = ["All"]
    truths_list_test = []

    # Add to the expected outputs and build the expected
    # input observations dataframe and classes dictionary
    for c in ["green", "blue", "red", "orange"]:
        truths = ["{}{:02d}".format(c, i) for i in range(10)]
        truths_list_test.append(truths)
        all_truths += truths
        classes_dict[c] = truths
        class_list_test.append(c)
    truths_list_test.insert(0, all_truths)

    observations = pd.DataFrame(
        {
            "obs_id": ["obs{:02d}".format(i) for i in range(len(all_truths))],
            "object_id": all_truths,
        }
    )
    for c in ["green", "blue", "red", "orange"]:
        observations.loc[observations["object_id"].isin(classes_dict[c]), "class"] = c

    # Remove the orange class from classes dict
    classes_dict_ = copy.deepcopy(classes_dict)
    classes_dict_.pop("orange")
    observations.loc[observations["class"].isin(["orange"]), "class"] = np.nan

    # Test for UserWarning when not all truths have an assigned class
    with pytest.warns(UserWarning):
        class_list, truths_list = _classHandler(classes_dict_, observations)

        assert "Unclassified" in class_list
        assert np.all(truths_list[-1] == classes_dict["orange"])

    with pytest.warns(UserWarning):
        class_list, truths_list = _classHandler("class", observations)

        assert "Unclassified" in class_list
        assert np.all(truths_list[-1] == classes_dict["orange"])

    return
