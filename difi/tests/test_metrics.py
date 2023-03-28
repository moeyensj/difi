import numpy as np

from ..metrics import calcFindableMinObs, calcFindableNightlyLinkages
from .create_test_data import createTestDataSet

MIN_OBS = range(5, 10)


def test_calcFindableMinObs():
    # --- Test calcFindableMinObs against the test data set

    column_mapping = {
        "truth": "truth",
        "obs_id": "obs_id",
    }

    for min_obs in MIN_OBS:
        # Generate test data set
        (
            observations_test,
            all_truths_test,
            linkage_members_test,
            all_linkages_test,
            summary_test,
        ) = createTestDataSet(min_obs, 5, 20)

        findable_observations = calcFindableMinObs(
            observations_test, min_obs=min_obs, column_mapping=column_mapping
        )

        for truth in findable_observations[column_mapping["truth"]].unique():
            # Make sure all observations are correctly identified as findable
            obs_ids = findable_observations[findable_observations[column_mapping["truth"]].isin([truth])][
                "obs_ids"
            ].values[0]
            np.testing.assert_array_equal(
                obs_ids,
                observations_test[observations_test["truth"] == truth]["obs_id"].values,
            )

            # Make sure all objects with not findable are not included in the findable_observations dataframe
            not_findable_truths_test = all_truths_test[all_truths_test["findable"] == 0]["truth"].values
            assert (
                len(
                    findable_observations[
                        findable_observations[column_mapping["truth"]].isin(not_findable_truths_test)
                    ]
                )
                == 0
            )

    return


def test_calcFindableNightlyLinkages():
    # --- Test calcFindableNightlyLinkages against the test data set
    column_mapping = {
        "truth": "truth",
        "obs_id": "obs_id",
        "time": "time",
        "night": "night",
    }
    # Generate test data set
    (
        observations_test,
        all_truths_test,
        linkage_members_test,
        all_linkages_test,
        summary_test,
    ) = createTestDataSet(5, 5, 20)

    # For every single truth in blue, their observations are seperated by a half day
    for truth in observations_test[observations_test["class"] == "blue"]["truth"].unique():
        mask = observations_test["truth"] == truth
        observations_test.loc[mask, "time"] = np.arange(0, len(observations_test[mask]) / 2, 0.5)

    # For every single truth in red, their observations are seperated by a quarter day
    for truth in observations_test[observations_test["class"] == "red"]["truth"].unique():
        mask = observations_test["truth"] == truth
        observations_test.loc[mask, "time"] = np.arange(0, len(observations_test[mask]) / 4, 0.25)

    # Observation times for greens are selected at random from the available ones in blues and greens
    observations_test.loc[observations_test["class"] == "green", "time"] = np.random.choice(
        observations_test[~observations_test["time"].isna()]["time"].values,
        len(observations_test[observations_test["class"] == "green"]),
        replace=True,
    )

    # Lets add a night column which is simply the floor of the observation time
    observations_test["night"] = np.floor(observations_test["time"]).astype(int)

    # With a maximum separation of 0.25 only reds should be findable
    findable_observations = calcFindableNightlyLinkages(
        observations_test,
        linkage_min_obs=2,
        max_obs_separation=0.25,
        min_linkage_nights=1,
        column_mapping=column_mapping,
    )

    for truth in findable_observations[column_mapping["truth"]].unique():
        # Make sure all observations are correctly identified as findable
        obs_ids = findable_observations[findable_observations[column_mapping["truth"]].isin([truth])][
            "obs_ids"
        ].values[0]
        np.testing.assert_array_equal(
            obs_ids,
            observations_test[observations_test["truth"] == truth]["obs_id"].values,
        )

    # Make sure that only reds were found
    classes_found = observations_test[
        observations_test["truth"].isin(findable_observations[column_mapping["truth"]].values)
    ]["class"].unique()
    np.testing.assert_array_equal(classes_found, np.array(["red"]))

    # With a maximum separation of 0.5 reds and blues should be findable
    findable_observations = calcFindableNightlyLinkages(
        observations_test,
        linkage_min_obs=2,
        max_obs_separation=0.5,
        min_linkage_nights=1,
        column_mapping=column_mapping,
    )

    for truth in findable_observations[column_mapping["truth"]].unique():
        # Make sure all observations are correctly identified as findable
        obs_ids = findable_observations[findable_observations[column_mapping["truth"]].isin([truth])][
            "obs_ids"
        ].values[0]
        np.testing.assert_array_equal(
            obs_ids,
            observations_test[observations_test["truth"] == truth]["obs_id"].values,
        )

    # Make sure that only reds and blues were found
    classes_found = observations_test[
        observations_test["truth"].isin(findable_observations[column_mapping["truth"]].values)
    ]["class"].unique()
    np.testing.assert_array_equal(classes_found, np.array(["red", "blue"]))

    # With a minimum linkage length of 1, everything should be findable
    findable_observations = calcFindableNightlyLinkages(
        observations_test,
        linkage_min_obs=1,
        max_obs_separation=0.5,
        min_linkage_nights=1,
        column_mapping=column_mapping,
    )

    for truth in findable_observations[column_mapping["truth"]].unique():
        # Make sure all observations are correctly identified as findable
        obs_ids = findable_observations[findable_observations[column_mapping["truth"]].isin([truth])][
            "obs_ids"
        ].values[0]
        np.testing.assert_array_equal(
            obs_ids,
            observations_test[observations_test["truth"] == truth]["obs_id"].values,
        )

    # Make sure that all reds, blues, and greens were found
    classes_found = observations_test[
        observations_test["truth"].isin(findable_observations[column_mapping["truth"]].values)
    ]["class"].unique()
    np.testing.assert_array_equal(classes_found, np.array(["red", "blue", "green"]))

    # With a minimum linkage length of 100, nothing should be findable
    findable_observations = calcFindableNightlyLinkages(
        observations_test,
        linkage_min_obs=100,
        max_obs_separation=0.5,
        min_linkage_nights=1,
        column_mapping=column_mapping,
    )
    assert len(findable_observations) == 0

    # --- These next few tests focus on red05 which has the following observations:
    #  obs_id   truth class time night
    # obs00000  red05  red  0.00  0
    # obs00008  red05  red  0.25  0
    # obs00013  red05  red  0.50  0
    # obs00024  red05  red  0.75  0
    # obs00049  red05  red  1.00  1
    # obs00051  red05  red  1.25  1
    # obs00057  red05  red  1.50  1
    # obs00070  red05  red  1.75  1
    # obs00085  red05  red  2.00  2
    # obs00096  red05  red  2.25  2

    # Lets set min_linkage nights to 3 with a maximum separation of 0.25, only red05 should be findable
    findable_observations = calcFindableNightlyLinkages(
        observations_test,
        linkage_min_obs=2,
        max_obs_separation=0.25,
        min_linkage_nights=3,
        column_mapping=column_mapping,
    )

    for truth in findable_observations[column_mapping["truth"]].unique():
        # Make sure all observations are correctly identified as findable
        obs_ids = findable_observations[findable_observations[column_mapping["truth"]].isin([truth])][
            "obs_ids"
        ].values[0]
        np.testing.assert_array_equal(
            obs_ids,
            observations_test[observations_test["truth"] == truth]["obs_id"].values,
        )

    # Make sure that only red05 should be findable
    classes_found = observations_test[
        observations_test["truth"].isin(findable_observations[column_mapping["truth"]].values)
    ]["class"].unique()
    np.testing.assert_array_equal(classes_found, np.array(["red"]))
    np.testing.assert_array_equal(findable_observations["truth"].values, np.array(["red05"]))

    # Keep min_linkage nights to 3 with a maximum separation of 0.25, set the last of red05's
    # observations to be outside the time separation resulting in only two viable tracklet nights,
    # it should no longer be findable
    observations_test.loc[observations_test["obs_id"] == "obs00096", "time"] = 2.26

    #  obs_id   truth class time night findable
    # obs00000  red05  red  0.00  0        Y
    # obs00008  red05  red  0.25  0        Y
    # obs00013  red05  red  0.50  0        Y
    # obs00024  red05  red  0.75  0        Y
    # obs00049  red05  red  1.00  1        Y
    # obs00051  red05  red  1.25  1        Y
    # obs00057  red05  red  1.50  1        Y
    # obs00070  red05  red  1.75  1        Y
    # obs00085  red05  red  2.00  2        N
    # obs00096  red05  red  2.26  2        N
    # red05 findable : N
    findable_observations = calcFindableNightlyLinkages(
        observations_test,
        linkage_min_obs=2,
        max_obs_separation=0.25,
        min_linkage_nights=3,
        column_mapping=column_mapping,
    )

    # Red05 should no longer be findable
    classes_found = observations_test[
        observations_test["truth"].isin(findable_observations[column_mapping["truth"]].values)
    ]["class"].unique()
    np.testing.assert_array_equal(classes_found, np.array([]))

    # Set the observation back to its original time
    observations_test.loc[observations_test["obs_id"] == "obs00096", "time"] = 2.25

    # Keep min_linkage nights to 3 with a maximum separation of 0.25, set the two of the observations
    # on night 1 to not be findable, red05 should still be findable with the remaining observations
    # but those unfindable observations should not be returned as findable observations
    observations_test.loc[observations_test["obs_id"] == "obs00057", "time"] = 1.51
    observations_test.loc[observations_test["obs_id"] == "obs00070", "time"] = 1.77
    # This observation needs to be shifted so that it is more than 0.25 from the previous exposure time
    # so we dont count a linkage across nights
    observations_test.loc[observations_test["obs_id"] == "obs00085", "time"] = 2.10

    #  obs_id   truth class time night findable
    # obs00000  red05  red  0.00  0        Y
    # obs00008  red05  red  0.25  0        Y
    # obs00013  red05  red  0.50  0        Y
    # obs00024  red05  red  0.75  0        Y
    # obs00049  red05  red  1.00  1        Y
    # obs00051  red05  red  1.25  1        Y
    # obs00057  red05  red  1.51  1        N
    # obs00070  red05  red  1.76  1        N
    # obs00085  red05  red  2.10  2        Y
    # obs00096  red05  red  2.25  2        Y
    # red05 findable : Y
    findable_observations = calcFindableNightlyLinkages(
        observations_test,
        linkage_min_obs=2,
        max_obs_separation=0.25,
        min_linkage_nights=3,
        column_mapping=column_mapping,
    )

    for truth in findable_observations[column_mapping["truth"]].unique():
        # Make sure all observations are correctly identified as findable
        obs_ids = findable_observations[findable_observations[column_mapping["truth"]].isin([truth])][
            "obs_ids"
        ].values[0]
        np.testing.assert_array_equal(
            obs_ids,
            observations_test[
                (observations_test["truth"] == truth)
                & (~observations_test["obs_id"].isin(["obs00057", "obs00070"]))
            ]["obs_id"].values,
        )

    # Make sure that only red05 should be findable
    classes_found = observations_test[
        observations_test["truth"].isin(findable_observations[column_mapping["truth"]].values)
    ]["class"].unique()
    np.testing.assert_array_equal(classes_found, np.array(["red"]))
    np.testing.assert_array_equal(findable_observations["truth"].values, np.array(["red05"]))

    # Set the observations back to their previous values
    observations_test.loc[observations_test["obs_id"] == "obs00057", "time"] = 1.50
    observations_test.loc[observations_test["obs_id"] == "obs00070", "time"] = 1.75
    observations_test.loc[observations_test["obs_id"] == "obs00085", "time"] = 2.00

    # Keep min_linkage nights to 3 with a maximum separation of 0.25, remove some of red05's observations
    # so that there are only two observations on each night -- it should still be the only object findable
    observations_test = observations_test[
        ~observations_test["obs_id"].isin(["obs00000", "obs00008", "obs00057", "obs00070"])
    ]

    #  obs_id   truth class time night findable
    # obs00013  red05  red  0.50  0        Y
    # obs00024  red05  red  0.75  0        Y
    # obs00049  red05  red  1.00  1        Y
    # obs00070  red05  red  1.75  1        Y
    # obs00085  red05  red  2.00  2        Y
    # obs00096  red05  red  2.25  2        Y
    # red05 findable : Y
    findable_observations = calcFindableNightlyLinkages(
        observations_test,
        linkage_min_obs=2,
        max_obs_separation=0.25,
        min_linkage_nights=3,
        column_mapping=column_mapping,
    )

    for truth in findable_observations[column_mapping["truth"]].unique():
        # Make sure all observations are correctly identified as findable
        obs_ids = findable_observations[findable_observations[column_mapping["truth"]].isin([truth])][
            "obs_ids"
        ].values[0]
        np.testing.assert_array_equal(
            obs_ids,
            observations_test[observations_test["truth"] == truth]["obs_id"].values,
        )

    # Make sure that only red05 should be findable
    classes_found = observations_test[
        observations_test["truth"].isin(findable_observations[column_mapping["truth"]].values)
    ]["class"].unique()
    np.testing.assert_array_equal(classes_found, np.array(["red"]))
    np.testing.assert_array_equal(findable_observations["truth"].values, np.array(["red05"]))

    # Keep min_linkage nights to 3 with a maximum separation of 0.25, set one of red05's
    # observations to be outside the time separation for a linkage -- it now should not be findable
    observations_test.loc[observations_test["obs_id"] == "obs00096", "time"] = 2.26

    #  obs_id   truth class time night findable
    # obs00013  red05  red  0.50  0        Y
    # obs00024  red05  red  0.75  0        Y
    # obs00049  red05  red  1.00  1        Y
    # obs00070  red05  red  1.75  1        Y
    # obs00085  red05  red  2.00  2        N
    # obs00096  red05  red  2.26  2        N
    # red05 findable : N
    findable_observations = calcFindableNightlyLinkages(
        observations_test,
        linkage_min_obs=2,
        max_obs_separation=0.25,
        min_linkage_nights=3,
        column_mapping=column_mapping,
    )

    # Red05 should no longer be findable
    classes_found = observations_test[
        observations_test["truth"].isin(findable_observations[column_mapping["truth"]].values)
    ]["class"].unique()
    np.testing.assert_array_equal(classes_found, np.array([]))

    return
