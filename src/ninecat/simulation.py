"""Monte Carlo simulation of head-to-head category matchups.

For each player and each simulated week we draw:

1. games played ~ Binomial(scheduled games, availability);
2. a per-week "true rate" from the projection posterior (Gamma for
   per-minute rates, Beta for shooting percentages, plus a small minutes
   shock), so the simulator knows a 10-game sample is less certain than a
   60-game one;
3. weekly counts from a Gamma-Poisson (negative binomial) sum over the games
   played, so hot and cold nights add the right amount of extra variance;
4. makes ~ Binomial(attempts, pct), so FG% and FT% follow from volume.

All player-weeks are drawn once up front ("common random numbers"). A team
total is then a sum over columns, and comparing lineups that differ by one
player compares the same simulated worlds, which makes small differences
in win probability far less noisy than independent runs would.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .categories import CATEGORIES, COUNT_STATS, NEGATIVE, PERCENT_STATS
from .projections import Projections

RAW = [*COUNT_STATS, "fgm", "ftm"]


@dataclass
class MatchupResult:
    category_win_prob: pd.Series   # P(team A wins category), ties count half
    win_prob: float
    tie_prob: float
    loss_prob: float
    expected_categories: float     # expected categories won by A (ties count half)
    totals_a: pd.Series            # mean simulated weekly totals
    totals_b: pd.Series

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({"team_a": self.totals_a, "team_b": self.totals_b,
                             "p_team_a_wins": self.category_win_prob})


class WeekSimulator:
    """Pre-simulated weekly box-score totals for a set of players."""

    def __init__(
        self,
        proj: Projections,
        games: pd.Series,
        n_sims: int = 4000,
        seed: int = 0,
        parameter_uncertainty: bool = True,
        overdispersion: bool = True,
        minutes_cv: float = 0.08,
    ) -> None:
        rng = np.random.default_rng(seed)
        games = games[~games.index.duplicated()]
        self.ids = list(games.index)
        self.col = {pid: i for i, pid in enumerate(self.ids)}
        self.n_sims = n_sims
        t = proj.table.loc[self.ids]
        S, P = n_sims, len(self.ids)

        g = games.to_numpy(float)
        g_int = np.floor(g) + (rng.random((S, P)) < (g - np.floor(g)))
        played = rng.binomial(g_int.astype(int), t["availability"].clip(0, 1).to_numpy())

        if parameter_uncertainty:
            sd = minutes_cv
            mins_mult = rng.lognormal(-sd ** 2 / 2, sd, (S, P))
        else:
            mins_mult = np.ones((S, P))

        self.draws: dict[str, np.ndarray] = {}
        for stat in COUNT_STATS:
            mu = t[stat].to_numpy(float) * mins_mult
            if parameter_uncertainty:
                shape = t[f"{stat}_shape"].to_numpy(float)
                mu = mu * rng.gamma(shape, 1.0 / shape, (S, P))
            if overdispersion:
                r = proj.dispersion[stat]
                lam = rng.gamma(np.maximum(played * r, 1e-12), np.maximum(mu, 0) / r)
                lam = np.where(played > 0, lam, 0.0)
            else:
                lam = played * mu
            self.draws[stat] = rng.poisson(lam).astype(np.int32)

        for _cat, (makes, att) in PERCENT_STATS.items():
            key = makes[:2]
            if parameter_uncertainty:
                p = rng.beta(t[f"{key}_alpha"].to_numpy(), t[f"{key}_beta"].to_numpy(), (S, P))
            else:
                p = np.broadcast_to((t[f"{key}_alpha"] / (t[f"{key}_alpha"] + t[f"{key}_beta"])).to_numpy(), (S, P))
            self.draws[makes] = rng.binomial(self.draws[att], p).astype(np.int32)

    def player_draws(self, pid: int) -> dict[str, np.ndarray]:
        i = self.col[pid]
        return {s: d[:, i] for s, d in self.draws.items()}

    def totals(self, roster, accrued: dict[str, float] | None = None) -> dict[str, np.ndarray]:
        idx = [self.col[p] for p in roster if p in self.col]
        out = {s: d[:, idx].sum(axis=1).astype(float) for s, d in self.draws.items()}
        for s, v in (accrued or {}).items():
            out[s] = out[s] + v
        return out

    def matchup(self, roster_a, roster_b, accrued_a=None, accrued_b=None) -> MatchupResult:
        return evaluate_matchup(self.totals(roster_a, accrued_a), self.totals(roster_b, accrued_b))


def category_values(totals: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Turn raw totals into the nine category values."""
    out = {}
    for cat in CATEGORIES:
        if cat in PERCENT_STATS:
            makes, att = PERCENT_STATS[cat]
            with np.errstate(invalid="ignore", divide="ignore"):
                out[cat] = np.where(totals[att] > 0, totals[makes] / totals[att], 0.0)
        else:
            out[cat] = totals[cat]
    return out


def category_outcomes(totals_a, totals_b) -> np.ndarray:
    """Array of shape (9, ...) with 1 = A wins, 0.5 = tie, 0 = B wins."""
    va, vb = category_values(totals_a), category_values(totals_b)
    rows = []
    for cat in CATEGORIES:
        a, b = (vb[cat], va[cat]) if cat in NEGATIVE else (va[cat], vb[cat])
        rows.append((a > b) + 0.5 * (a == b))
    return np.stack(rows)


def evaluate_matchup(totals_a, totals_b) -> MatchupResult:
    out = category_outcomes(totals_a, totals_b)
    won_a = (out == 1).sum(axis=0)
    won_b = (out == 0).sum(axis=0)
    mean_a = {c: v.mean() for c, v in category_values(totals_a).items()}
    mean_b = {c: v.mean() for c, v in category_values(totals_b).items()}
    return MatchupResult(
        category_win_prob=pd.Series(out.mean(axis=1), index=CATEGORIES),
        win_prob=float((won_a > won_b).mean()),
        tie_prob=float((won_a == won_b).mean()),
        loss_prob=float((won_a < won_b).mean()),
        expected_categories=float(out.sum(axis=0).mean()),
        totals_a=pd.Series(mean_a)[CATEGORIES],
        totals_b=pd.Series(mean_b)[CATEGORIES],
    )


def swing_categories(result: MatchupResult, low: float = 0.25, high: float = 0.75) -> list[str]:
    """Categories that are still genuinely in play this week."""
    p = result.category_win_prob
    return [c for c in CATEGORIES if low <= p[c] <= high]
