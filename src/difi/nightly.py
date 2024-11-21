from typing import Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
from adam_core.orbit_determination import FittedOrbitMembers

from difi.observations import Observations

from .metrics import haversine_distance


class OrbitSummary(qv.Table):
    #: Orbit ID
    orbit_id = qv.LargeStringColumn()
    #: Number of observations
    num_obs = qv.Int64Column()
    #: Number of nights
    num_nights = qv.Int64Column()
    #: Number of singletons
    num_singletons = qv.Int64Column()
    #: Number of tracklets
    num_tracklets = qv.Int64Column()

    @classmethod
    def create(cls, nightly_orbit_summary: "NightlyOrbitSummary") -> "OrbitSummary":
        """
        Create an OrbitSummary table from a NightlyOrbitSummary table. This table describes
        the number of observed tracklets and singletons for each orbit.

        Parameters
        ----------
        nightly_orbit_summary : NightlyOrbitSummary
            NightlyOrbitSummary table.

        Returns
        -------
        OrbitSummary
            OrbitSummary table.
        """
        nightly_orbit_summary_table = (
            nightly_orbit_summary.flattened_table().sort_by(
                [("orbit_id", "ascending"), ("night", "ascending")]
            )
        ).combine_chunks()

        orbit_summary = nightly_orbit_summary_table.group_by(["orbit_id"], use_threads=False).aggregate(
            [
                ("night", "count"),
                # ("dtime", "sum"),
                # ("dsky", "sum"),
                ("num_obs", "sum"),
            ]
        )

        singletons = nightly_orbit_summary_table.filter(pc.equal(nightly_orbit_summary_table["num_obs"], 1))
        singletons = (
            singletons.group_by(["orbit_id"], use_threads=False)
            .aggregate(
                [
                    ("num_obs", "count"),
                ]
            )
            .rename_columns({"num_obs_count": "singletons"})
        )
        orbit_summary = orbit_summary.join(singletons, "orbit_id", "orbit_id")

        return OrbitSummary.from_kwargs(
            orbit_id=orbit_summary["orbit_id"],
            num_nights=orbit_summary["night_count"],
            num_obs=orbit_summary["num_obs_sum"],
            num_singletons=orbit_summary["singletons"],
            num_tracklets=pc.subtract(orbit_summary["night_count"], orbit_summary["singletons"]),
        )


class NightlyOrbitSummary(qv.Table):
    #: Orbit ID
    orbit_id = qv.LargeStringColumn()
    #: Night
    night = qv.Int64Column()
    #: Number of observations for this night
    num_obs = qv.Int64Column()
    #: Number of unique filters
    num_filters = qv.Int64Column()
    #: Time span of the observations in seconds
    dtime = qv.Float64Column()
    #: Right ascension difference between the first and last observation in arcseconds
    dra = qv.Float64Column()
    #: Declination difference between the first and last observation in arcseconds
    ddec = qv.Float64Column()
    #: Sky distance between the first and last observation in arcseconds
    dsky = qv.Float64Column()
    #: Sky velocity between the first and last observation in arcseconds per second
    vsky = qv.Float64Column()
    #: Magnitude difference between the brightest and faintest observation
    dmag = qv.Float64Column()
    #: Magnitude mean of the observations
    mag_mean = qv.Float64Column()
    #: Magnitude standard deviation of the observations
    mag_sigma = qv.Float64Column()
    #: Sum of the chi2 values of the residuals
    chi2_sum = qv.Float64Column(nullable=True)
    #: Mean of the chi2 values of the residuals
    chi2_mean = qv.Float64Column(nullable=True)
    #: Standard deviation of the chi2 values of the residuals
    chi2_sigma = qv.Float64Column(nullable=True)
    #: Difference between the maximum and minimum chi2 values of the residuals
    dchi2 = qv.Float64Column(nullable=True)

    @classmethod
    def create(cls, orbit_members: FittedOrbitMembers, observations: Observations) -> "NightlyOrbitSummary":
        """
        Create a nightly summary for the given orbit members and observations.

        Parameters
        ----------
        orbit_members : FittedOrbitMembers
            The fitted orbit members.
        observations : Observations
            The observations table.

        Returns
        -------
        NightlyOrbitSummary
            The nightly summary table.
        """
        assert pc.all(pc.is_in(orbit_members.obs_id.unique(), observations.id)).as_py()

        members_table = orbit_members.flattened_table().drop_columns(["residuals.values"])
        members_table = members_table.join(
            observations.flattened_table()
            .append_column("mjd", observations.time.mjd())
            .select(
                [
                    "id",
                    "night",
                    "mjd",
                    "ra",
                    "dec",
                    "mag",
                    "filter",
                ]
            ),
            "obs_id",
            "id",
        )

        members_table = members_table.sort_by(
            [("orbit_id", "ascending"), ("night", "ascending"), ("mjd", "ascending")]
        )

        nightly_summary = members_table.group_by(["orbit_id", "night"], use_threads=False).aggregate(
            [
                ("mjd", "min"),
                ("mjd", "max"),
                ("night", "count"),
                ("ra", "first"),
                ("ra", "last"),
                ("dec", "first"),
                ("dec", "last"),
                ("dec", "mean"),
                ("mag", "max"),
                ("mag", "min"),
                ("mag", "mean"),
                ("mag", "stddev"),
                ("residuals.chi2", "min"),
                ("residuals.chi2", "max"),
                ("residuals.chi2", "mean"),
                ("residuals.chi2", "stddev"),
                ("residuals.chi2", "sum"),
                ("filter", "count_distinct"),
            ]
        )

        nightly_summary = nightly_summary.append_column(
            "dtime",
            pa.array(
                86400.0
                * (
                    nightly_summary["mjd_max"].to_numpy(zero_copy_only=False)
                    - nightly_summary["mjd_min"].to_numpy(zero_copy_only=False)
                )
            ),
        )

        ra1 = nightly_summary["ra_first"].to_numpy(zero_copy_only=False)
        ra2 = nightly_summary["ra_last"].to_numpy(zero_copy_only=False)
        dec1 = nightly_summary["dec_first"].to_numpy(zero_copy_only=False)
        dec2 = nightly_summary["dec_last"].to_numpy(zero_copy_only=False)
        dtime = nightly_summary["dtime"].to_numpy(zero_copy_only=False)

        dra = np.where(ra2 - ra1 > 180, ra2 - ra1 - 360, ra2 - ra1)
        ddec = dec2 - dec1

        # Compute the haversine distance between the first and last observation
        # for each orbit-night pair
        dsky = haversine_distance(ra1, dec1, ra2, dec2)
        vsky = dsky / dtime
        vsky = np.where(np.isnan(vsky), 0.0, vsky)
        nightly_summary = nightly_summary.append_column(
            "dsky",
            pa.array(dsky),
        )
        nightly_summary = nightly_summary.append_column(
            "vsky",
            pa.array(vsky),
        )
        nightly_summary = nightly_summary.append_column(
            "dra",
            pa.array(dra),
        )
        nightly_summary = nightly_summary.append_column(
            "ddec",
            pa.array(ddec),
        )
        nightly_summary = nightly_summary.append_column(
            "dmag", pc.subtract(nightly_summary["mag_max"], nightly_summary["mag_min"])
        )
        nightly_summary = nightly_summary.append_column(
            "dchi2", pc.subtract(nightly_summary["residuals.chi2_max"], nightly_summary["residuals.chi2_min"])
        )

        nightly_summary = nightly_summary.sort_by([("orbit_id", "ascending"), ("night", "ascending")])

        return NightlyOrbitSummary.from_kwargs(
            orbit_id=nightly_summary["orbit_id"],
            night=nightly_summary["night"],
            num_obs=nightly_summary["night_count"],
            num_filters=nightly_summary["filter_count_distinct"],
            dtime=nightly_summary["dtime"],
            dra=pc.multiply(nightly_summary["dra"], 3600.0),
            ddec=pc.multiply(nightly_summary["ddec"], 3600.0),
            dsky=pc.multiply(nightly_summary["dsky"], 3600.0),
            vsky=pc.multiply(nightly_summary["vsky"], 3600.0),
            dmag=nightly_summary["dmag"],
            mag_mean=nightly_summary["mag_mean"],
            mag_sigma=nightly_summary["mag_stddev"],
            chi2_sum=nightly_summary["residuals.chi2_sum"],
            chi2_mean=nightly_summary["residuals.chi2_mean"],
            chi2_sigma=nightly_summary["residuals.chi2_stddev"],
            dchi2=nightly_summary["dchi2"],
        )


def compute_orbit_summaries(
    orbit_members: FittedOrbitMembers, observations: Observations
) -> Tuple[OrbitSummary, NightlyOrbitSummary]:
    """
    Compute the nightly and orbit summaries for the given orbit members and observations.

    Parameters
    ----------
    orbit_members : FittedOrbitMembers
        The fitted orbit members.
    observations : Observations
        The observations table.

    Returns
    -------
    Tuple[OrbitSummary, NightlyOrbitSummary]
        The orbit and nightly summaries.
    """
    nightly_summary = NightlyOrbitSummary.create(orbit_members, observations)
    orbit_summary = OrbitSummary.create(nightly_summary)
    return orbit_summary, nightly_summary
