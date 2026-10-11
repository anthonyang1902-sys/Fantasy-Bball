import numpy as np

from ninecat.inseason import evaluate_trade, streaming_moves, typical_week_games, week_context, simulate_week
from ninecat.scoring import category_scores


def test_mid_week_context_banks_played_games(world):
    season, league, _ = world
    ctx = week_context(season, league, week=10, day=10 * 7 + 4)
    banked = season.logs_between(70, 74)
    rows = banked[banked["player_id"].isin(league.rosters["Team A"])]
    assert ctx.accrued["Team A"]["pts"] == rows["pts"].sum()
    assert (ctx.games_left <= 3).all()


def test_simulate_week_probabilities_sum_to_one(world, proj):
    season, league, _ = world
    ctx = week_context(season, league, 10)
    res = simulate_week(proj, league, ctx, "Team A", n_sims=1000)
    assert np.isclose(res.win_prob + res.tie_prob + res.loss_prob, 1)


def test_streaming_finds_upgrades_and_respects_protection(world, proj):
    season, league, _ = world
    scores = category_scores(proj, "g")
    ctx = week_context(season, league, 10)
    protect = league.rosters["Team A"][:5]
    moves = streaming_moves(proj, league, ctx, "Team A", scores, n_candidates=15, n_sims=1500, protect=protect)
    assert not moves["drop"].isin(protect).any()
    assert not moves["add"].isin(league.rostered()).any()
    assert moves["delta_win_prob"].is_monotonic_decreasing


def test_empty_trade_changes_nothing(world, proj):
    season, league, _ = world
    scores = category_scores(proj, "g")
    games = typical_week_games(season, 70)
    a, b = evaluate_trade(proj, league, scores, games, "Team A", [], "Team B", [], n_sims=500)
    assert a.delta_win_rate == 0 and a.delta_value == 0
    assert b.delta_win_rate == 0


def test_uneven_trade_keeps_rosters_full(world, proj):
    season, league, _ = world
    scores = category_scores(proj, "g")
    games = typical_week_games(season, 70)
    ra, rb = league.rosters["Team A"], league.rosters["Team B"]
    a, b = evaluate_trade(proj, league, scores, games, "Team A", [ra[0]], "Team B", rb[:2], n_sims=500)
    assert len(a.auto_drops) == 1 and not a.auto_adds
    assert len(b.auto_adds) == 1 and not b.auto_drops


def test_fleecing_trade_helps_the_receiver(world, proj):
    season, league, _ = world
    scores = category_scores(proj, "g")
    games = typical_week_games(season, 70)
    ra, rb = league.rosters["Team A"], league.rosters["Team B"]
    best_b = scores.loc[rb, "total"].idxmax()
    worst_a = scores.loc[ra, "total"].idxmin()
    a, b = evaluate_trade(proj, league, scores, games, "Team A", [worst_a], "Team B", [best_b], n_sims=800)
    assert a.delta_win_rate > 0 > b.delta_win_rate
