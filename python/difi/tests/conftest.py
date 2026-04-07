from importlib.resources import files

import pytest

from ..difi import LinkageMembers
from ..observations import Observations


@pytest.fixture
def test_observations() -> Observations:
    observations_file = files("difi.tests.testdata").joinpath("observations.parquet")
    return Observations.from_parquet(observations_file)


@pytest.fixture
def test_linkage_members() -> LinkageMembers:
    orbit_members_file = files("difi.tests.testdata").joinpath("linkage_members.parquet")
    return LinkageMembers.from_parquet(orbit_members_file)
