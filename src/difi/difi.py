import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv

from .cifi import AllObjects
from .observations import Observations
from .partitions import Partitions, PartitionSummary


class PartitionMapping(qv.Table):
    linkage_id = qv.LargeStringColumn()
    partition_id = qv.LargeStringColumn()


class LinkageMembers(qv.Table):
    linkage_id = qv.LargeStringColumn()
    obs_id = qv.LargeStringColumn()


class AllLinkages(qv.Table):
    linkage_id = qv.LargeStringColumn()
    partition_id = qv.LargeStringColumn()
    linked_object_id = qv.LargeStringColumn(nullable=True)
    num_obs = qv.Int64Column()
    num_obs_outside_partition = qv.Int64Column()
    num_members = qv.Int64Column()
    pure = qv.BooleanColumn()
    pure_complete = qv.BooleanColumn()
    contaminated = qv.BooleanColumn()
    contamination = qv.Float64Column()
    mixed = qv.BooleanColumn()
    found_pure = qv.BooleanColumn()
    found_contaminated = qv.BooleanColumn()

    @classmethod
    def create(
        cls,
        observations: Observations,
        partition_summary: PartitionSummary,
        linkage_members: LinkageMembers,
        min_obs: int = 6,
        contamination_percentage: float = 0.0,
    ) -> "AllLinkages":
        """
        Create a table of all linkages and classify them as pure, contaminated, or mixed.
        All linkages in linkage members are assumed to be within the given partition though
        observations may be outside the partition.

        Parameters
        ----------
        observations : Observations
            Table of observations
        linkage_members : LinkageMembers
            Table of linkage members.
        partition_summary : PartitionSummary
            The partition summary table (must only be a single partition).
        min_obs : int
            Minimum number of observations required to consider a linkage found.
        contamination_percentage : float
            Maximum percentage of observations that can belong to a different object
            for a linkage to be considered contaminated. Otherwise, it is considered
            mixed.

        Returns
        -------
        AllLinkages
            Table of all linkages.
        """
        if len(partition_summary) != 1:
            raise ValueError("Partition summary must contain exactly one partition.")
        if not pc.all(pc.is_in(linkage_members.obs_id.unique(), observations.id.unique())).as_py():
            raise ValueError("All linkage members must be in the observations.")

        # Join linkage members with observations so we get the object ID for each
        # linkage's constituent observations
        linkage_member_associations = (
            linkage_members.flattened_table()
            .select(["linkage_id", "obs_id"])
            .join(observations.table.select(["id", "object_id", "night"]), "obs_id", "id")
        )

        # Identify observations as outside_partition
        linkage_member_associations = linkage_member_associations.append_column(
            "outside_partition",
            pc.invert(
                pc.and_(
                    pc.greater_equal(
                        linkage_member_associations["night"], partition_summary.start_night[0].as_py()
                    ),
                    pc.less_equal(
                        linkage_member_associations["night"], partition_summary.end_night[0].as_py()
                    ),
                )
            ),
        )

        # Observation counts per object
        observations_per_object = observations.table.group_by("object_id").aggregate([("id", "count")])

        unique_object_members = (
            linkage_member_associations.group_by("linkage_id")
            .aggregate(
                [
                    ("obs_id", "count_distinct"),
                    ("object_id", "count_distinct"),
                    ("object_id", "distinct"),
                    ("outside_partition", "sum"),
                ]
            )
            .rename_columns(
                ["linkage_id", "num_obs", "num_members", "object_ids", "num_obs_outside_partition"]
            )
        )

        all_linkages = (
            linkage_member_associations.group_by(("linkage_id", "object_id"))
            .aggregate([([], "count_all")])
            .rename_columns(["linkage_id", "object_id", "object_id_counts"])
            .join(
                unique_object_members.select(
                    ["linkage_id", "num_obs", "num_members", "num_obs_outside_partition"]
                ),
                "linkage_id",
            )
        )
        all_linkages = (
            all_linkages.append_column(
                "percentage_in_linkage",
                pc.divide(
                    pc.cast(all_linkages["object_id_counts"], pa.float64()),
                    pc.cast(all_linkages["num_obs"], pa.float64()),
                ),
            )
            .sort_by([("linkage_id", "ascending"), ("percentage_in_linkage", "descending")])
            .group_by("linkage_id", use_threads=False)
            .aggregate(
                [("object_id", "first"), ("object_id_counts", "first"), ("percentage_in_linkage", "first")]
            )
            .join(
                unique_object_members.select(
                    ["linkage_id", "num_obs", "num_members", "num_obs_outside_partition"]
                ),
                "linkage_id",
            )
        )
        all_linkages = all_linkages.append_column(
            "contamination",
            pc.round(
                pc.multiply(
                    pc.subtract(1.0, all_linkages["percentage_in_linkage_first"]),
                    100.0,
                ),
                10,
            ),
        )

        # Identify those linkages that are pure
        all_linkages = all_linkages.append_column(
            "pure",
            pc.if_else(pc.equal(all_linkages["contamination"], 0.0), pa.scalar(True), pa.scalar(False)),
        )

        # Identify those linkages that are contaminated
        all_linkages = all_linkages.append_column(
            "contaminated",
            pc.if_else(
                pc.and_(
                    pc.less_equal(all_linkages["contamination"], contamination_percentage),
                    pc.greater(all_linkages["contamination"], 0.0),
                ),
                pa.scalar(True),
                pa.scalar(False),
            ),
        )

        # Identify those linkages that are mixed
        all_linkages = all_linkages.append_column(
            "mixed",
            pc.if_else(
                pc.and_(pc.equal(all_linkages["pure"], False), pc.equal(all_linkages["contaminated"], False)),
                pa.scalar(True),
                pa.scalar(False),
            ),
        )

        # Identify the linked object ID for each linkage if the linkage is pure or contaminated
        all_linkages = all_linkages.append_column(
            "linked_object_id",
            pc.if_else(
                pc.or_(all_linkages["pure"], all_linkages["contaminated"]),
                all_linkages["object_id_first"],
                pa.scalar(None),
            ),
        )

        # Identify those linkages that are pure and complete
        # Pure complete linkages are linkages that contain all observations of the linked object
        # within the given observations
        all_linkages = all_linkages.join(observations_per_object, "linked_object_id", "object_id")

        # Count the number of observations in the partition and in the linkage
        obs_partition_counts = (
            observations.filter_partition(partition_summary)
            .flattened_table()
            .group_by("object_id")
            .aggregate([("id", "count")])
            .rename_columns(["object_id", "num_obs_in_partition"])
        )
        linkage_obs_partition_counts = (
            linkage_member_associations.filter(
                pc.equal(linkage_member_associations["outside_partition"], False)
            )
            .group_by("linkage_id")
            .aggregate([("obs_id", "count_distinct")])
            .rename_columns(["linkage_id", "num_linkage_obs_inside_partition"])
        )

        all_linkages = all_linkages.join(obs_partition_counts, "linked_object_id", "object_id").join(
            linkage_obs_partition_counts, "linkage_id"
        )
        all_linkages = all_linkages.append_column(
            "pure_complete",
            pc.if_else(
                pc.and_(
                    pc.equal(all_linkages["pure"], True),
                    pc.and_(
                        pc.equal(
                            all_linkages["num_obs_in_partition"],
                            all_linkages["num_linkage_obs_inside_partition"],
                        ),
                        pc.invert(pc.is_null(all_linkages["num_obs_in_partition"])),
                    ),
                ),
                pa.scalar(True),
                pa.scalar(False),
            ),
        )

        # Identify those linkages that are found and pure
        # Pure linkages are linkages that contain at least min_obs observations
        # of the linked object
        all_linkages = all_linkages.append_column(
            "found_pure",
            pc.if_else(
                pc.and_(
                    pc.equal(all_linkages["pure"], True), pc.greater_equal(all_linkages["num_obs"], min_obs)
                ),
                pa.scalar(True),
                pa.scalar(False),
            ),
        )

        # Identify those linkages that are found and contaminated
        all_linkages = all_linkages.append_column(
            "found_contaminated",
            pc.if_else(
                pc.and_(
                    pc.equal(all_linkages["contaminated"], True),
                    pc.greater_equal(all_linkages["object_id_counts_first"], min_obs),
                ),
                pa.scalar(True),
                pa.scalar(False),
            ),
        )

        return AllLinkages.from_kwargs(
            linkage_id=all_linkages["linkage_id"],
            partition_id=pa.repeat(partition_summary.id[0], len(all_linkages)),
            linked_object_id=all_linkages["linked_object_id"],
            num_obs=all_linkages["num_obs"],
            num_obs_outside_partition=all_linkages["num_obs_outside_partition"],
            num_members=all_linkages["num_members"],
            pure=all_linkages["pure"],
            pure_complete=pc.fill_null(all_linkages["pure_complete"], False),
            contaminated=all_linkages["contaminated"],
            contamination=all_linkages["contamination"],
            mixed=all_linkages["mixed"],
            found_pure=all_linkages["found_pure"],
            found_contaminated=all_linkages["found_contaminated"],
        )


def update_all_objects(
    all_objects: AllObjects,
    observations: Observations,
    linkage_members: LinkageMembers,
    all_linkages: AllLinkages,
    min_obs: int = 6,
) -> AllObjects:
    """
    Update the AllObjects table using the AllLinkages table. This function updates the
    pure, pure_complete, contaminated, contaminant, mixed, obs_in_pure, obs_in_pure_complete,
    obs_in_contaminated, obs_as_contaminant, obs_in_mixed, found_pure, found_contaminated
    columns.

    Parameters
    ----------
    all_objects : AllObjects
        Table of all objects.
    observations : Observations
        Table of observations.
    linkage_members : LinkageMembers
        Table of linkage members.
    all_linkages : AllLinkages
        Table of all linkages.
    min_obs : int
        Minimum number of observations required to consider an object found in
        either a pure or contaminated linkage.

    Returns
    -------
    all_objects : AllObjects
        Updated table of all objects.
    """
    # Create a table of linkage members and their associated object IDs
    linkage_member_associations = (
        linkage_members.flattened_table()
        .select(["linkage_id", "obs_id"])
        .join(observations.table.select(["id", "object_id"]), "obs_id", "id")
    )

    unique_object_members = (
        linkage_member_associations.group_by(("linkage_id", "object_id"))
        .aggregate([([], "count_all")])
        .rename_columns(["linkage_id", "object_id", "object_id_counts"])
        .join(
            all_linkages.table.select(
                ["linkage_id", "linked_object_id", "pure", "pure_complete", "contaminated", "mixed"]
            ),
            "linkage_id",
        )
    )

    all_objects_linkages = (
        unique_object_members.group_by("object_id").aggregate([]).rename_columns(["object_id"])
    )

    # Count True values rather than rows for pure/pure_complete/contaminated
    uom_counts = (
        unique_object_members.append_column("pure_int", pc.cast(unique_object_members["pure"], pa.int64()))
        .append_column("pure_complete_int", pc.cast(unique_object_members["pure_complete"], pa.int64()))
        .append_column("contaminated_int", pc.cast(unique_object_members["contaminated"], pa.int64()))
    )

    all_objects_linkages_counts = (
        uom_counts.filter(pc.equal(uom_counts["object_id"], uom_counts["linked_object_id"]))
        .group_by("object_id")
        .aggregate(
            [
                ("pure_int", "sum"),
                ("pure_complete_int", "sum"),
                ("contaminated_int", "sum"),
            ]
        )
        .rename_columns(["object_id", "pure", "pure_complete", "contaminated"])
    )
    all_objects_linkages = all_objects_linkages.join(all_objects_linkages_counts, "object_id", "object_id")

    all_objects_linkages_contaminant = (
        unique_object_members.filter(
            pc.and_(
                pc.invert(
                    pc.equal(unique_object_members["object_id"], unique_object_members["linked_object_id"])
                ),
                pc.equal(unique_object_members["contaminated"], True),
            )
        )
        .group_by("object_id")
        .aggregate(
            [
                ("contaminated", "count"),
            ]
        )
        .rename_columns(["object_id", "contaminant"])
    )
    all_objects_linkages = all_objects_linkages.join(
        all_objects_linkages_contaminant, "object_id", "object_id"
    )

    all_objects_linkages_mixed = (
        unique_object_members.filter(pc.equal(unique_object_members["mixed"], True))
        .group_by("object_id")
        .aggregate(
            [
                ("mixed", "count"),
            ]
        )
        .rename_columns(["object_id", "mixed"])
    )
    all_objects_linkages = all_objects_linkages.join(all_objects_linkages_mixed, "object_id", "object_id")

    obs_in_pure = (
        unique_object_members.filter(pc.equal(unique_object_members["pure"], True))
        .group_by("object_id")
        .aggregate([("object_id_counts", "sum")])
        .rename_columns(["object_id", "obs_in_pure"])
    )
    all_objects_linkages = all_objects_linkages.join(obs_in_pure, "object_id", "object_id")

    obs_in_pure_complete = (
        unique_object_members.filter(pc.equal(unique_object_members["pure_complete"], True))
        .group_by("object_id")
        .aggregate([("object_id_counts", "sum")])
        .rename_columns(["object_id", "obs_in_pure_complete"])
    )
    all_objects_linkages = all_objects_linkages.join(obs_in_pure_complete, "object_id", "object_id")

    obs_in_contaminated = (
        unique_object_members.filter(
            pc.and_(
                pc.equal(unique_object_members["object_id"], unique_object_members["linked_object_id"]),
                pc.equal(unique_object_members["contaminated"], True),
            )
        )
        .group_by("object_id")
        .aggregate([("object_id_counts", "sum")])
        .rename_columns(["object_id", "obs_in_contaminated"])
    )
    all_objects_linkages = all_objects_linkages.join(obs_in_contaminated, "object_id", "object_id")

    obs_in_contaminated_as_contaminant = (
        unique_object_members.filter(
            pc.and_(
                pc.invert(
                    pc.equal(unique_object_members["object_id"], unique_object_members["linked_object_id"])
                ),
                pc.equal(unique_object_members["contaminated"], True),
            )
        )
        .group_by("object_id")
        .aggregate([("object_id_counts", "sum")])
        .rename_columns(["object_id", "obs_as_contaminant"])
    )
    all_objects_linkages = all_objects_linkages.join(
        obs_in_contaminated_as_contaminant, "object_id", "object_id"
    )

    obs_in_mixed = (
        unique_object_members.filter(pc.equal(unique_object_members["mixed"], True))
        .group_by("object_id")
        .aggregate([("object_id_counts", "sum")])
        .rename_columns(["object_id", "obs_in_mixed"])
    )
    all_objects_linkages = all_objects_linkages.join(obs_in_mixed, "object_id", "object_id")

    # Now determine which objects were found (in pure or contaminated linkages with at least min_obs observations belonging to the object)
    found_pure = (
        unique_object_members.filter(
            pc.and_(
                pc.equal(unique_object_members["pure"], True),
                pc.greater_equal(unique_object_members["object_id_counts"], min_obs),
            )
        )
        .group_by("object_id")
        .aggregate([("linkage_id", "count_distinct")])
        .rename_columns(["object_id", "found_pure"])
    )
    all_objects_linkages = all_objects_linkages.join(found_pure, "object_id", "object_id")

    found_contaminated = (
        unique_object_members.filter(
            pc.and_(
                pc.equal(unique_object_members["contaminated"], True),
                pc.greater_equal(unique_object_members["object_id_counts"], min_obs),
            )
        )
        .group_by("object_id")
        .aggregate([("linkage_id", "count_distinct")])
        .rename_columns(["object_id", "found_contaminated"])
    )
    all_objects_linkages = all_objects_linkages.join(found_contaminated, "object_id", "object_id")

    all_objects = all_objects.table.select(
        [
            "object_id",
            "partition_id",
            "mjd_min",
            "mjd_max",
            "arc_length",
            "num_obs",
            "num_observatories",
            "findable",
        ]
    ).join(all_objects_linkages, "object_id", "object_id")

    return AllObjects.from_kwargs(
        object_id=all_objects["object_id"],
        partition_id=all_objects["partition_id"],
        mjd_min=all_objects["mjd_min"],
        mjd_max=all_objects["mjd_max"],
        arc_length=all_objects["arc_length"],
        num_obs=all_objects["num_obs"],
        num_observatories=all_objects["num_observatories"],
        findable=all_objects["findable"],
        found_pure=pc.fill_null(all_objects["found_pure"], 0).cast(pa.int64()),
        found_contaminated=pc.fill_null(all_objects["found_contaminated"], 0).cast(pa.int64()),
        pure=pc.fill_null(all_objects["pure"], 0).cast(pa.int64()),
        pure_complete=pc.fill_null(all_objects["pure_complete"], 0).cast(pa.int64()),
        contaminated=pc.fill_null(all_objects["contaminated"], 0).cast(pa.int64()),
        contaminant=pc.fill_null(all_objects["contaminant"], 0).cast(pa.int64()),
        mixed=pc.fill_null(all_objects["mixed"], 0).cast(pa.int64()),
        obs_in_pure=pc.fill_null(all_objects["obs_in_pure"], 0).cast(pa.int64()),
        obs_in_pure_complete=pc.fill_null(all_objects["obs_in_pure_complete"], 0).cast(pa.int64()),
        obs_in_contaminated=pc.fill_null(all_objects["obs_in_contaminated"], 0).cast(pa.int64()),
        obs_as_contaminant=pc.fill_null(all_objects["obs_as_contaminant"], 0).cast(pa.int64()),
        obs_in_mixed=pc.fill_null(all_objects["obs_in_mixed"], 0).cast(pa.int64()),
    )


def analyze_linkages(
    observations: Observations,
    linkage_members: LinkageMembers,
    all_objects: AllObjects,
    partition_summary: PartitionSummary | None = None,
    partition_mapping: PartitionMapping | None = None,
    min_obs: int = 6,
    contamination_percentage: float = 20.0,
) -> tuple[AllObjects, AllLinkages, PartitionSummary]:
    """
    Did I Find It?

    Given a table of observations and a table of linkage members, this function
    determines which objects were found in the observations and updates the
    all_objects table with the results. It also returns an AllLinkages table that classifies
    each linkage as pure, contaminated, or mixed.

    A pure linkage is one where all constieuent observations belong to the same object. A contaminated
    linkage is where up to the contamination_percentage of the observations belong to a different object but
    the rest belong to the same object. A mixed linkage is where the observations belong to multiple objects.

    Parameters
    ----------
    observations : Observations
        Table of observations.
    linkage_members : LinkageMembers
        Table of linkage members.
    all_objects : AllObjects
        Table of all objects.
    min_obs : int
        Minimum number of observations required to consider an object found in
        either a pure or contaminated linkage.
    contamination_percentage : float
        Maximum percentage of observations that can belong to a different object
        for a linkage to be considered contaminated. Otherwise, it is considered
        mixed.

    Returns
    -------
    all_objects : AllObjects
        Updated table of all objects.
    all_linkages : AllLinkages
        Table of all linkages.
    """

    all_linkages = AllLinkages.empty()
    all_objects_updated = AllObjects.empty()
    partition_summaries_updated = PartitionSummary.empty()

    # If no partition summary is provided, create a single partition covering all observations
    if partition_summary is None:
        partitions = Partitions.create_single(observations.night)
        partition_summary = PartitionSummary.create(observations, partitions)

    # If no partition mapping is provided, assume all linkages belong to the single partition
    if partition_mapping is None:
        linkage_ids_unique = linkage_members.linkage_id.unique()
        partition_mapping = PartitionMapping.from_kwargs(
            linkage_id=linkage_ids_unique,
            partition_id=pa.repeat(partition_summary.id[0], len(linkage_ids_unique)),
        )

    # Enforce exactly one partition in analyze_linkages
    if len(partition_summary) != 1:
        raise ValueError("analyze_linkages requires exactly one partition in partition_summary")

    for partition in partition_summary:
        partition_id = partition.id[0].as_py()

        linkage_ids = partition_mapping.select("partition_id", partition_id).linkage_id
        linkage_members_partition = linkage_members.apply_mask(
            pc.is_in(linkage_members.linkage_id, linkage_ids)
        )

        # Create the AllLinkages table for this partition
        if len(linkage_members_partition) > 0:

            all_linkages_partition = AllLinkages.create(
                observations,
                partition,
                linkage_members_partition,
                min_obs=min_obs,
                contamination_percentage=contamination_percentage,
            )

        else:
            all_linkages_partition = AllLinkages.empty()

        all_linkages = qv.concatenate([all_linkages, all_linkages_partition])

        # Update the AllObjects table for this partition
        all_objects_partition = update_all_objects(
            all_objects.select("partition_id", partition_id),
            observations,
            linkage_members_partition,
            all_linkages,
            min_obs=min_obs,
        )
        all_objects_updated = qv.concatenate([all_objects_updated, all_objects_partition])

        # Update the partion summary with the number of pure, pure unknown, contaminated, and mixed linkages
        # and the completeness
        pure_known = len(
            all_linkages_partition.apply_mask(
                pc.and_(
                    pc.equal(all_linkages_partition.pure, True),
                    pc.invert(pc.is_null(all_linkages_partition.linked_object_id)),
                )
            )
        )
        pure_unknown = len(
            all_linkages_partition.apply_mask(
                pc.and_(
                    pc.equal(all_linkages_partition.pure, True),
                    pc.is_null(all_linkages_partition.linked_object_id),
                )
            )
        )
        contaminated = len(all_linkages_partition.select("contaminated", True))
        mixed = len(all_linkages_partition.select("mixed", True))

        found = len(
            all_linkages_partition.apply_mask(
                pc.and_(
                    pc.equal(all_linkages_partition.pure, True),
                    pc.invert(pc.is_null(all_linkages_partition.linked_object_id)),
                )
            ).linked_object_id.unique()
        )
        # findable may be null if not set; treat None as 0
        findable = partition.findable[0].as_py() if not pc.is_null(partition.findable[0]).as_py() else 0

        completeness = found / findable if findable and findable > 0 else float(found)
        completeness *= 100

        # Update the partition summary with the number of pure, pure unknown, contaminated, and mixed linkages
        partition_summary_updated = PartitionSummary.from_kwargs(
            id=partition.id,
            start_night=partition.start_night,
            end_night=partition.end_night,
            observations=partition.observations,
            findable=partition.findable,
            found=[found],
            completeness=[completeness],
            pure_known=[pure_known],
            pure_unknown=[pure_unknown],
            contaminated=[contaminated],
            mixed=[mixed],
        )

        partition_summaries_updated = qv.concatenate([partition_summaries_updated, partition_summary_updated])

    return all_objects_updated, all_linkages, partition_summaries_updated
