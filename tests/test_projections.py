import numpy as np
import pandas as pd

from ninecat.data import Season
from ninecat.projections import ProjectionConfig, project

BOX = dict(fgm=5, fga=10, fg3m=1, ftm=2, fta=3, pts=13, reb=5, ast=3, stl=1, blk=0, tov=2)


def _toy_season(n_games_star=60, n_games_rookie=2, star_reb=12):
    """Thirty ordinary forwards, one rebounding star, one barely-seen rookie."""
    rows, players = [], []
    rng = np.random.default_rng(0)
    for pid in range(32):
        players.append({"player_id": pid, "name": f"P{pid}", "nba_team": "T00", "position": "F"})
        n = n_games_star if pid == 30 else n_games_rookie if pid == 31 else 60
        for day in range(n):
            reb = star_reb if pid == 30 else 30 if pid == 31 else int(rng.poisson(5))
            rows.append({"player_id": pid, "day": day, "nba_team": "T00", "min": 30.0, **(BOX | {"reb": reb})})
    sched = pd.DataFrame({"day": range(70), "nba_team": "T00"})
    return Season(pd.DataFrame(players), pd.DataFrame(rows), sched, n_weeks=10)


def test_small_samples_shrink_toward_the_prior():
    proj = project(_toy_season(), 70).table
    # The rookie averaged 30 rebounds in two games; the projection should not believe it.
    assert proj.loc[31, "reb"] < 15
    # The star's 60-game 12-rebound average is mostly kept.
    assert 10 < proj.loc[30, "reb"] <= 12


def test_no_shrinkage_reproduces_raw_rates():
    proj = project(_toy_season(), 70, ProjectionConfig(prior_scale=0.0)).table
    assert np.isclose(proj.loc[31, "reb"], 30, rtol=1e-6)


def test_recency_weighting_follows_a_change():
    season = _toy_season()
    logs = season.logs
    late = (logs["player_id"] == 30) & (logs["day"] >= 50)
    logs.loc[late, "reb"] = 2
    flat = project(season, 70, ProjectionConfig(half_life_days=float("inf"))).table.loc[30, "reb"]
    recent = project(season, 70, ProjectionConfig(half_life_days=5)).table.loc[30, "reb"]
    assert recent < flat


def test_percentages_are_posterior_means(proj):
    t = proj.table
    assert t["fg_pct"].between(0.3, 0.75).all()
    assert np.allclose(t["fg_pct"], t["fg_alpha"] / (t["fg_alpha"] + t["fg_beta"]))
    assert t["availability"].between(0, 1).all()
