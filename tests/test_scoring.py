import numpy as np

from ninecat.categories import CATEGORIES
from ninecat.scoring import category_scores, suggest_punts, team_profile


def test_z_scores_are_standardised_on_the_pool(proj):
    z = category_scores(proj, "z", iterations=1)
    pool = (proj.table["min"] * proj.table["availability"]).nlargest(156).index
    assert np.allclose(z.loc[pool, CATEGORIES].std(), 1.0, atol=1e-9)
    assert np.allclose(z.loc[pool, CATEGORIES].mean(), 0.0, atol=1e-9)


def test_g_scores_shrink_every_category(proj):
    g, z = category_scores(proj, "g"), category_scores(proj, "z")
    pool = z["total"].nlargest(156).index
    ratio = g.loc[pool, CATEGORIES].std() / z.loc[pool, CATEGORIES].std()
    assert (ratio < 1).all()
    # Steals are noisier week to week than points, so G discounts them more.
    assert ratio["stl"] < ratio["pts"]


def test_turnovers_count_against_a_player(proj):
    z = category_scores(proj, "z")
    most_tov = proj.table["tov"].idxmax()
    assert z.loc[most_tov, "tov"] < 0


def test_punting_removes_a_category_from_the_total(proj):
    s = category_scores(proj, "g", weights={"ft_pct": 0.0}, iterations=1)
    assert np.allclose(s["total"], s[[c for c in CATEGORIES if c != "ft_pct"]].sum(axis=1))


def test_team_profile_and_punt_suggestions(proj, world):
    league = world[1]
    s = category_scores(proj, "g")
    profile = team_profile(s, league.rosters)
    assert profile.shape == (12, 9)
    for team in league.teams:
        assert set(suggest_punts(profile, team)) <= set(CATEGORIES)
