"""In-season decision tools: weekly matchups, streaming/waiver moves, and trades."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .categories import CATEGORIES, RAW_STATS_FOR_ACCRUAL
from .data import DAYS_PER_WEEK, League, Season
from .projections import Projections
from .simulation import WeekSimulator, category_outcomes, evaluate_matchup, MatchupResult


# ---------------------------------------------------------------------------
# Weekly matchup
# ---------------------------------------------------------------------------

@dataclass
class WeekContext:
    week: int
    day: int                      # "now": games before this day are already banked
    games_left: pd.Series         # remaining scheduled games this week, by player
    accrued: dict[str, dict[str, float]]  # team -> raw stat totals already banked


def week_context(season: Season, league: League, week: int, day: int | None = None) -> WeekContext:
    start, end = week * DAYS_PER_WEEK, (week + 1) * DAYS_PER_WEEK
    day = start if day is None else int(np.clip(day, start, end))
    games_left = season.player_games(day, end)
    banked = season.logs_between(start, day)
    accrued = {}
    for team, roster in league.rosters.items():
        rows = banked[banked["player_id"].isin(roster)]
        accrued[team] = {s: float(rows[s].sum()) for s in RAW_STATS_FOR_ACCRUAL}
    return WeekContext(week, day, games_left, accrued)


def simulate_week(proj: Projections, league: League, ctx: WeekContext, team: str, opponent: str | None = None,
                  n_sims: int = 5000, seed: int = 0) -> MatchupResult:
    opponent = opponent or league.opponent(team, ctx.week)
    ids = league.rosters[team] + league.rosters[opponent]
    sim = WeekSimulator(proj, ctx.games_left.loc[ids], n_sims=n_sims, seed=seed)
    return sim.matchup(league.rosters[team], league.rosters[opponent],
                       ctx.accrued[team], ctx.accrued[opponent])


# ---------------------------------------------------------------------------
# Streaming / waiver moves for the current week
# ---------------------------------------------------------------------------

def streaming_moves(
    proj: Projections,
    league: League,
    ctx: WeekContext,
    team: str,
    scores: pd.DataFrame,
    opponent: str | None = None,
    n_candidates: int = 40,
    n_sims: int = 4000,
    seed: int = 0,
    protect: list[int] | None = None,
) -> pd.DataFrame:
    """Rank every (add, drop) pair by how much it raises this week's win probability.

    Candidates are the free agents with the most projected value *this week*
    (season value x games remaining). Dropped players keep the stats they have
    already banked, as on most platforms. ``protect`` lists players you will
    never drop (your stars do not belong on the wire for one week of streaming).
    """
    opponent = opponent or league.opponent(team, ctx.week)
    mine, theirs = league.rosters[team], league.rosters[opponent]
    fas = pd.Index(league.free_agents(proj.table.index))
    weekly_value = scores.loc[fas, "total"] * ctx.games_left.loc[fas] / 3.5
    adds = weekly_value[ctx.games_left.loc[fas] > 0].nlargest(n_candidates).index.tolist()
    protect = set(protect or [])
    drops = [p for p in mine if p not in protect]

    sim = WeekSimulator(proj, ctx.games_left.loc[mine + theirs + adds], n_sims=n_sims, seed=seed)
    base_mine = sim.totals(mine, ctx.accrued[team])
    base_opp = sim.totals(theirs, ctx.accrued[opponent])
    base = evaluate_matchup(base_mine, base_opp)
    add_draws = {s: d[:, [sim.col[p] for p in adds]].T for s, d in sim.draws.items()}  # (K, S)

    rows = []
    for d in drops:
        dd = sim.player_draws(d)
        totals = {s: base_mine[s][None, :] - dd[s][None, :] + add_draws[s] for s in base_mine}
        out = category_outcomes(totals, {s: v[None, :] for s, v in base_opp.items()})  # (9, K, S)
        won = (out == 1).sum(0)
        lost = (out == 0).sum(0)
        win = (won > lost).mean(1) + 0.5 * (won == lost).mean(1)
        cats = out.sum(0).mean(1)
        for k, a in enumerate(adds):
            rows.append({"add": a, "drop": d, "win_prob": win[k], "exp_categories": cats[k]})
    base_win = base.win_prob + 0.5 * base.tie_prob
    df = pd.DataFrame(rows)
    df["delta_win_prob"] = df["win_prob"] - base_win
    df["delta_categories"] = df["exp_categories"] - base.expected_categories
    names = proj.table["name"]
    df["add_name"], df["drop_name"] = names.loc[df["add"]].to_numpy(), names.loc[df["drop"]].to_numpy()
    df["add_games"] = ctx.games_left.loc[df["add"]].to_numpy()
    df["drop_games"] = ctx.games_left.loc[df["drop"]].to_numpy()
    df = df.sort_values(["delta_win_prob", "delta_categories"], ascending=False)
    df.attrs["baseline_win_prob"] = base_win
    df.attrs["baseline_categories"] = base.expected_categories
    return df.reset_index(drop=True)


def best_pickups(scores: pd.DataFrame, league: League, team: str, top: int = 10) -> pd.DataFrame:
    """Rest-of-season upgrades: free agents whose value beats your weakest player."""
    roster = league.rosters[team]
    worst = scores.loc[roster, "total"].idxmin()
    fas = scores.loc[league.free_agents(scores.index), "total"].nlargest(top)
    return pd.DataFrame({"add": fas.index, "add_value": fas.to_numpy(), "drop": worst,
                         "gain": fas.to_numpy() - scores.loc[worst, "total"]})


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------

def typical_week_games(season: Season, as_of: int) -> pd.Series:
    """Average scheduled games per remaining week, by player (may be fractional)."""
    weeks_left = max(season.n_weeks - as_of // DAYS_PER_WEEK, 1)
    return season.player_games(as_of, season.n_weeks * DAYS_PER_WEEK) / weeks_left


@dataclass
class TradeSide:
    team: str
    value_before: float
    value_after: float
    win_rate_before: float        # average P(win) against the rest of the league in a typical week
    win_rate_after: float
    categories_before: pd.Series  # average P(win category) against the field
    categories_after: pd.Series
    auto_drops: list[int]
    auto_adds: list[int]

    @property
    def delta_win_rate(self) -> float:
        return self.win_rate_after - self.win_rate_before

    @property
    def delta_value(self) -> float:
        return self.value_after - self.value_before


def _balance(roster: list[int], size: int, scores: pd.DataFrame, pool: list[int]) -> tuple[list[int], list[int], list[int]]:
    roster = list(roster)
    drops, adds = [], []
    while len(roster) > size:
        worst = scores.loc[roster, "total"].idxmin()
        roster.remove(worst)
        drops.append(int(worst))
    while len(roster) < size and pool:
        best = scores.loc[pool, "total"].idxmax()
        pool.remove(best)
        roster.append(int(best))
        adds.append(int(best))
    return roster, drops, adds


def _field_strength(sim: WeekSimulator, roster: list[int], field: list[list[int]]) -> tuple[float, pd.Series]:
    mine = sim.totals(roster)
    wins, cats = [], []
    for opp in field:
        res = evaluate_matchup(mine, sim.totals(opp))
        wins.append(res.win_prob + 0.5 * res.tie_prob)
        cats.append(res.category_win_prob)
    return float(np.mean(wins)), pd.concat(cats, axis=1).mean(axis=1)


def evaluate_trade(
    proj: Projections,
    league: League,
    scores: pd.DataFrame,
    games: pd.Series,
    team_a: str,
    send_a: list[int],
    team_b: str,
    send_b: list[int],
    n_sims: int = 3000,
    seed: int = 0,
) -> tuple[TradeSide, TradeSide]:
    """Evaluate a trade for both teams.

    Each team is measured before and after against the same field (every
    other team in the league), in a typical rest-of-season week. Uneven
    trades are balanced the way a manager would: the side with too many
    players drops its least valuable one, and the side with an open spot
    picks up the best free agent.
    """
    after = league.with_trade(team_a, send_a, team_b, send_b)
    pool = league.free_agents(scores.sort_values("total", ascending=False).index)[:25]
    ra, drops_a, adds_a = _balance(after.rosters[team_a], league.roster_size, scores, pool)
    rb, drops_b, adds_b = _balance(after.rosters[team_b], league.roster_size, scores, pool)

    field = [r for t, r in league.rosters.items() if t not in (team_a, team_b)]
    ids = list(dict.fromkeys([p for r in league.rosters.values() for p in r] + adds_a + adds_b))
    sim = WeekSimulator(proj, games.loc[ids], n_sims=n_sims, seed=seed)

    sides = []
    for team, new_roster, drops, adds in ((team_a, ra, drops_a, adds_a), (team_b, rb, drops_b, adds_b)):
        old_roster = league.rosters[team]
        wr0, c0 = _field_strength(sim, old_roster, field)
        wr1, c1 = _field_strength(sim, new_roster, field)
        sides.append(TradeSide(
            team=team,
            value_before=float(scores.loc[old_roster, "total"].sum()),
            value_after=float(scores.loc[new_roster, "total"].sum()),
            win_rate_before=wr0, win_rate_after=wr1,
            categories_before=c0[CATEGORIES], categories_after=c1[CATEGORIES],
            auto_drops=drops, auto_adds=adds,
        ))
    return sides[0], sides[1]
