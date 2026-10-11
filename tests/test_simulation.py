import numpy as np
import pandas as pd

from ninecat.simulation import WeekSimulator, category_outcomes


def test_identical_teams_are_a_coin_flip(proj, world):
    league = world[1]
    roster = league.rosters["Team A"]
    games = pd.Series(3.0, index=roster)
    sim = WeekSimulator(proj, games, n_sims=4000, seed=1)
    res = sim.matchup(roster, roster)
    assert res.win_prob == res.loss_prob == 0
    assert res.tie_prob == 1
    assert np.allclose(res.category_win_prob, 0.5)


def test_stars_beat_scrubs(proj):
    by_value = (proj.table["pts"] * proj.table["availability"]).sort_values()
    stars, scrubs = by_value.index[-13:].tolist(), by_value.index[50:63].tolist()
    sim = WeekSimulator(proj, pd.Series(3.5, index=stars + scrubs), n_sims=3000, seed=2)
    res = sim.matchup(stars, scrubs)
    assert res.win_prob > 0.95
    assert res.category_win_prob["pts"] > 0.99


def test_weekly_means_match_projection(proj):
    pid = proj.table["min"].idxmax()
    games = pd.Series([4.0], index=[pid])
    sim = WeekSimulator(proj, games, n_sims=40_000, seed=3)
    expected = 4 * proj.table.loc[pid, "availability"] * proj.table.loc[pid, "pts"]
    assert abs(sim.draws["pts"].mean() - expected) / expected < 0.03


def test_more_variance_with_overdispersion_and_uncertainty(proj):
    pid = proj.table["min"].idxmax()
    games = pd.Series([4.0], index=[pid])
    plain = WeekSimulator(proj, games, n_sims=20_000, seed=4, parameter_uncertainty=False, overdispersion=False)
    full = WeekSimulator(proj, games, n_sims=20_000, seed=4)
    assert full.draws["reb"].var() > plain.draws["reb"].var()


def test_ties_count_half_and_turnovers_are_reversed():
    a = {s: np.array([10.0]) for s in ["fga", "fta", "fg3m", "pts", "reb", "ast", "stl", "blk", "tov", "fgm", "ftm"]}
    b = dict(a)
    b["tov"] = np.array([12.0])  # more turnovers is worse
    out = category_outcomes(a, b)[:, 0]
    assert out[-1] == 1.0
    assert (out[:-1] == 0.5).all()
