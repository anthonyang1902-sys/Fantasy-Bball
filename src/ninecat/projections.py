"""Rest-of-season projections from game logs.

Each player's per-game line is decomposed as

    stat per game = minutes per game x stat per minute
    expected stat per scheduled game = availability x stat per game

and each piece gets its own model:

* **Per-minute rates** use a Gamma-Poisson model. The prior is the player's
  position group, and its strength is estimated from the data (empirical
  Bayes): stats with a lot of real between-player spread (blocks) shrink
  less than stats that are mostly noise. Observations are down-weighted
  exponentially by age, so recent form counts more, and the posterior keeps
  a shape parameter that the simulator uses for parameter uncertainty.
* **Shooting percentages** use the Beta-Binomial analogue on makes/attempts.
* **Minutes** use an exponentially weighted average over recent games,
  regressed toward the season average, which reacts quickly to role changes.
* **Availability** is the recency-weighted share of team games played,
  shrunk toward a league prior, which picks up current injury spells.

Count stats are overdispersed relative to Poisson (hot and cold nights), so
we also estimate a negative-binomial dispersion per stat for the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .categories import COUNT_STATS, PERCENT_STATS
from .data import Season


@dataclass
class ProjectionConfig:
    # Recency decay for per-minute rates. Tuned on held-out synthetic seasons
    # (see reports/results.md): decaying rates added noise faster than it
    # tracked real change, so the default keeps the whole season. Minutes,
    # which do shift with role changes, keep their own recency weighting.
    half_life_days: float = float("inf")
    prior_scale: float = 1.0           # multiplier on the empirical-Bayes prior strength (0 = no shrinkage)
    minutes_half_life_games: float = 6.0
    minutes_prior_games: float = 4.0   # pull of the season-average minutes
    availability_half_life_games: float = 8.0
    availability_prior_games: float = 6.0
    default_availability: float = 0.88
    replacement_minutes: float = 12.0  # minutes assumed for players with no games yet
    min_minutes_for_prior: float = 150.0


@dataclass
class Projections:
    table: pd.DataFrame                 # indexed by player_id
    dispersion: dict[str, float]        # negative-binomial size parameter per count stat
    as_of: int
    config: ProjectionConfig = field(default_factory=ProjectionConfig)

    def per_game(self, stat: str) -> pd.Series:
        return self.table[stat]


def _decay(age: np.ndarray, half_life: float) -> np.ndarray:
    if not np.isfinite(half_life):
        return np.ones_like(age, dtype=float)
    return 0.5 ** (age / half_life)


def _gamma_prior(per_player: pd.DataFrame, stat: str, overdispersion: float) -> tuple[float, float]:
    """Method-of-moments Gamma prior (shape a, rate b in minutes) for a per-minute rate."""
    m = per_player["min"].to_numpy()
    x = per_player[stat].to_numpy()
    rate = x / m
    mean = np.average(rate, weights=m)
    total_var = np.average((rate - mean) ** 2, weights=m)
    noise_var = np.average(overdispersion * rate / m, weights=m)
    between = max(total_var - noise_var, (0.05 * mean) ** 2, 1e-12)
    b = mean / between
    return mean * b, b


def _beta_prior(per_player: pd.DataFrame, makes: str, att: str) -> tuple[float, float]:
    a_ = per_player[att].to_numpy()
    keep = a_ > 0
    p_i = per_player[makes].to_numpy()[keep] / a_[keep]
    w = a_[keep]
    p = np.average(p_i, weights=w)
    total_var = np.average((p_i - p) ** 2, weights=w)
    noise_var = np.average(p_i * (1 - p_i) / w, weights=w)
    between = max(total_var - noise_var, 1e-4)
    strength = max(p * (1 - p) / between - 1, 1.0)
    return p * strength, (1 - p) * strength


def estimate_dispersion(logs: pd.DataFrame) -> dict[str, float]:
    """Pooled negative-binomial size ``r`` per stat (variance = mu + mu^2 / r).

    The mean for each game is the player's season rate at his *average*
    minutes, so game-to-game minutes variation is part of the dispersion,
    which is what the weekly simulator needs.
    """
    g = logs.groupby("player_id")
    n = g["min"].transform("size")
    out = {}
    for stat in COUNT_STATS:
        mu = g[stat].transform("mean")
        keep = (n >= 5) & (mu > 0)
        x, m = logs.loc[keep, stat].to_numpy(float), mu[keep].to_numpy()
        excess = ((x - m) ** 2).sum() - m.sum()
        out[stat] = float(np.clip((m ** 2).sum() / excess, 1.0, 1000.0)) if excess > 0 else 1000.0
    return out


def _availability(season: Season, logs: pd.DataFrame, as_of: int, cfg: ProjectionConfig) -> pd.Series:
    sched = season.schedule[season.schedule["day"] < as_of]
    team_days = {t: np.sort(g["day"].to_numpy()) for t, g in sched.groupby("nba_team")}
    played = logs.groupby("player_id")["day"].apply(lambda d: set(d.to_numpy()))
    out = {}
    for pid, team in season.players["nba_team"].items():
        days = team_days.get(team, np.array([], dtype=int))
        played_days = played.get(pid, set())
        if days.size == 0:
            out[pid] = cfg.default_availability
            continue
        hit = np.fromiter((d in played_days for d in days), dtype=float, count=days.size)
        w = _decay(np.arange(days.size)[::-1].astype(float), cfg.availability_half_life_games)
        k = cfg.availability_prior_games
        out[pid] = (np.dot(w, hit) + k * cfg.default_availability) / (w.sum() + k)
    return pd.Series(out)


def _minutes(logs: pd.DataFrame, cfg: ProjectionConfig) -> pd.Series:
    df = logs[["player_id", "day", "min"]].sort_values(["player_id", "day"], ascending=[True, False])
    back = df.groupby("player_id").cumcount().to_numpy(float)
    df = df.assign(w=_decay(back, cfg.minutes_half_life_games))
    df["wm"] = df["w"] * df["min"]
    g = df.groupby("player_id")
    season_avg = g["min"].mean()
    k = cfg.minutes_prior_games
    return (g["wm"].sum() + k * season_avg) / (g["w"].sum() + k)


def project(season: Season, as_of: int, config: ProjectionConfig | None = None) -> Projections:
    """Project every player's per-game production for games after ``as_of``."""
    cfg = config or ProjectionConfig()
    logs = season.logs_before(as_of)
    players = season.players
    table = players[["name", "nba_team", "position"]].copy()
    table["gp"] = logs.groupby("player_id").size().reindex(table.index).fillna(0).astype(int)

    dispersion = estimate_dispersion(logs) if len(logs) else {s: 10.0 for s in COUNT_STATS}

    # Decay-weighted totals per player.
    w = _decay((as_of - logs["day"]).to_numpy(float), cfg.half_life_days)
    raw_cols = ["min", *COUNT_STATS, "fgm", "ftm"]
    weighted = logs[raw_cols].mul(w, axis=0).assign(player_id=logs["player_id"].to_numpy())
    wsum = weighted.groupby("player_id").sum().reindex(table.index).fillna(0.0)
    totals = logs.groupby("player_id")[raw_cols].sum()
    totals = totals.assign(position=players["position"].reindex(totals.index))
    eligible = totals[totals["min"] >= cfg.min_minutes_for_prior]

    minutes = _minutes(logs, cfg).reindex(table.index).fillna(cfg.replacement_minutes)
    table["min"] = minutes
    table["availability"] = _availability(season, logs, as_of, cfg).reindex(table.index)

    groups = table["position"].fillna("UNK")
    for stat in COUNT_STATS:
        overdisp = 1 + (logs[stat].mean() if len(logs) else 0.0) / dispersion[stat]
        shape = pd.Series(np.nan, index=table.index)
        rate = pd.Series(np.nan, index=table.index)
        for pos in groups.unique():
            pool = eligible[eligible["position"] == pos] if (eligible["position"] == pos).sum() >= 15 else eligible
            a, b = _gamma_prior(pool, stat, overdisp)
            a, b = a * cfg.prior_scale, b * cfg.prior_scale
            idx = groups.index[groups == pos]
            shape[idx] = a + wsum.loc[idx, stat]
            rate[idx] = b + wsum.loc[idx, "min"]
        no_info = rate <= 0  # no prior and no data
        rate[no_info] = 1.0
        per_min = shape / rate
        table[stat] = per_min * table["min"]
        table[f"{stat}_shape"] = shape.clip(lower=1e-3)

    for cat, (makes, att) in PERCENT_STATS.items():
        a0, b0 = _beta_prior(eligible, makes, att)
        a0, b0 = a0 * cfg.prior_scale, b0 * cfg.prior_scale
        alpha = a0 + wsum[makes]
        beta = b0 + (wsum[att] - wsum[makes])
        fallback = (alpha + beta) <= 0
        alpha[fallback], beta[fallback] = 1.0, 1.0
        key = cat.split("_")[0]
        table[f"{key}_alpha"] = alpha
        table[f"{key}_beta"] = beta
        table[cat] = alpha / (alpha + beta)
        table[makes] = table[cat] * table[att]

    return Projections(table, dispersion, as_of, cfg)
