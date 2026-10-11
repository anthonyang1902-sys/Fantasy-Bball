import pytest

from ninecat.projections import project
from ninecat.synthetic import SyntheticConfig, generate_season


@pytest.fixture(scope="session")
def world():
    season, league, truth = generate_season(SyntheticConfig(seed=42))
    return season, league, truth


@pytest.fixture(scope="session")
def proj(world):
    season, _, _ = world
    return project(season, 10 * 7)
