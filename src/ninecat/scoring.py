"""Player valuation with Z-scores and G-scores.

Z-scores standardise each category by the spread of player *averages*:

    Z = (mu_p - mu_pool) / sigma_pool

That ignores the fact that in head-to-head play you win a category based on
one week's noisy total. Rosenof (2024, "Improving on Z-scores for
head-to-head fantasy basketball", arXiv:2307.02188) derives the G-score,
which adds the average within-player week-to-week variance tau^2:

    G = (mu_p - mu_pool) / sqrt(sigma_pool^2 + kappa * tau_pool^2)

with kappa = 2N / (2N - 1) for N players per team. Volatile categories
(steals, blocks, 3PM) are worth less than Z-scores suggest, because an edge
in them converts less reliably into category wins.

Everything here is on a "per scheduled game" scale (production x
availability) and turned into weekly terms with ``games_per_week``: the
weekly between-player spread grows like g while the within-player noise
grows like sqrt(g).

Percentage stats are volume weighted: a player's FG% impact is
``attempts * (pct - pool_pct) / pool_mean_attempts``, so a 60% shooter on
four attempts helps less than a 55% shooter on fifteen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .categories import CATEGORIES, COUNTING_CATEGORIES, NEGATIVE, PERCENT_STATS
from .projections import Projections


def _mixture_var(mu: pd.Series, var: pd.Series, avail: pd.Series) -> pd.Series:
    """Variance of B*Y with B ~ Bernoulli(avail) and Y having mean mu, variance var."""
    return avail * var + avail * (1 - avail) * mu ** 2


def category_moments(proj: Projections) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-scheduled-game mean and within-player variance of each category's contribution.

    Percentage categories are expressed as ``makes - pool_pct * attempts``
    later, so here we return makes/attempts moments instead.
    """
    t = proj.table
    a = t["availability"]
    mean, var = {}, {}
    for stat in [*COUNTING_CATEGORIES, "fga", "fta"]:
        mu = t[stat]
        nb_var = mu + mu ** 2 / proj.dispersion[stat]
        mean[stat] = a * mu
        var[stat] = _mixture_var(mu, nb_var, a)
    for _cat, (makes, att) in PERCENT_STATS.items():
        mean[makes] = a * t[makes]
    return pd.DataFrame(mean), pd.DataFrame(var)


def _scores_for_pool(proj, mean, var, pool, method, games_per_week, kappa) -> pd.DataFrame:
    t = proj.table
    out = {}
    for cat in COUNTING_CATEGORIES:
        mu, v = mean[cat], var[cat]
        diff = mu - mu[pool].mean()
        sigma2 = mu[pool].var()
        tau2 = v[pool].mean() / games_per_week
        denom = np.sqrt(sigma2 + (kappa * tau2 if method == "g" else 0.0))
        out[cat] = (-diff if cat in NEGATIVE else diff) / denom
    for cat, (makes, att) in PERCENT_STATS.items():
        m, A = mean[makes], mean[att]
        p0 = m[pool].sum() / A[pool].sum()
        a_bar = A[pool].mean()
        impact = (m - p0 * A) / a_bar
        p = t[cat]
        within = (A * p * (1 - p) + (p - p0) ** 2 * var[att]) / a_bar ** 2
        sigma2 = impact[pool].var()
        tau2 = within[pool].mean() / games_per_week
        denom = np.sqrt(sigma2 + (kappa * tau2 if method == "g" else 0.0))
        out[cat] = (impact - impact[pool].mean()) / denom
    return pd.DataFrame(out)[CATEGORIES]


def category_scores(
    proj: Projections,
    method: str = "g",
    pool_size: int = 156,
    players_per_team: int = 13,
    games_per_week: float = 3.5,
    weights: dict[str, float] | None = None,
    iterations: int = 3,
) -> pd.DataFrame:
    """Score every player in every category.

    ``pool_size`` is the number of rostered players in the league (teams x
    roster size); the reference distribution is re-estimated a few times
    from the top ``pool_size`` players. ``weights`` lets you punt a category
    (weight 0) or emphasise one; the ``total`` column applies them.
    """
    if method not in {"g", "z"}:
        raise ValueError("method must be 'g' or 'z'")
    weights = {c: 1.0 for c in CATEGORIES} | (weights or {})
    kappa = 2 * players_per_team / (2 * players_per_team - 1)
    mean, var = category_moments(proj)
    pool = (proj.table["min"] * proj.table["availability"]).nlargest(pool_size).index
    w = pd.Series(weights)[CATEGORIES]
    for _ in range(iterations):
        scores = _scores_for_pool(proj, mean, var, pool, method, games_per_week, kappa)
        total = scores.mul(w, axis=1).sum(axis=1)
        pool = total.nlargest(pool_size).index
    scores["total"] = total
    return scores


def rankings(proj: Projections, method: str = "g", weights=None, **kwargs) -> pd.DataFrame:
    """Projection table joined with category scores, sorted by total value."""
    scores = category_scores(proj, method=method, weights=weights, **kwargs)
    cols = ["name", "nba_team", "position", "gp", "min", "availability", *COUNTING_CATEGORIES, "fg_pct", "ft_pct"]
    out = proj.table[cols].join(scores.add_prefix("v_"))
    out = out.sort_values("v_total", ascending=False)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def team_profile(scores: pd.DataFrame, rosters: dict[str, list[int]]) -> pd.DataFrame:
    """Sum of category scores for each fantasy team (rows) by category (columns)."""
    return pd.DataFrame({t: scores.loc[r, CATEGORIES].sum() for t, r in rosters.items()}).T


def suggest_punts(profile: pd.DataFrame, team: str, max_punts: int = 2, threshold: float = -0.75) -> list[str]:
    """Categories where a team is far below the league (in league standard deviations)."""
    z = (profile - profile.mean()) / profile.std(ddof=0)
    row = z.loc[team].sort_values()
    return [c for c, v in row.items() if v <= threshold][:max_punts]
