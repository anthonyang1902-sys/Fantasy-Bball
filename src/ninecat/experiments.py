"""Experiments that check the models instead of trusting them.

1. ``projection_backtest`` - rest-of-season projection error versus simple
   baselines, at several points in the season.
2. ``tune_projections``   - grid search of the recency half-life and the
   prior strength, on seeds the evaluation never sees.
3. ``calibration``        - do simulated matchup probabilities come true at
   the stated rate? Compared against simpler simulators.
4. ``g_vs_z``             - which ranking better predicts how much a player
   actually raises a team's expected category wins?

All experiments run on synthetic seasons (``ninecat.synthetic``), where the
future is known and nothing leaks from it into the models.
"""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pandas as pd

from .categories import CATEGORIES, COUNT_STATS, PERCENT_STATS
from .data import DAYS_PER_WEEK, Season
from .projections import ProjectionConfig, project
from .scoring import category_scores
from .simulation import WeekSimulator, category_outcomes, evaluate_matchup
from .synthetic import SyntheticConfig, generate_season

EVAL_STATS = ["min", *COUNT_STATS, "fg_pct", "ft_pct"]


# ---------------------------------------------------------------------------
# 1-2. Projection accuracy
# ---------------------------------------------------------------------------

def _per_game(logs: pd.DataFrame) -> pd.DataFrame:
    g = logs.groupby("player_id")
    out = g[["min", *COUNT_STATS]].mean()
    for cat, (m, a) in PERCENT_STATS.items():
        out[cat] = g[m].sum() / g[a].sum().replace(0, np.nan)
    out["gp"] = g.size()
    for _cat, (_m, a) in PERCENT_STATS.items():
        out[a + "_total"] = g[a].sum()
    return out


def _baseline_season_avg(season: Season, as_of: int) -> pd.DataFrame:
    return _per_game(season.logs_before(as_of))


def _baseline_last_n(season: Season, as_of: int, n: int = 15) -> pd.DataFrame:
    logs = season.logs_before(as_of).sort_values("day")
    return _per_game(logs.groupby("player_id").tail(n))


def _model(config: ProjectionConfig):
    def run(season: Season, as_of: int) -> pd.DataFrame:
        t = project(season, as_of, config).table
        return t[["min", *COUNT_STATS, "fg_pct", "ft_pct"]]
    return run


def default_models(config: ProjectionConfig | None = None) -> dict:
    cfg = config or ProjectionConfig()
    return {
        "Season average": _baseline_season_avg,
        "Last 15 games": _baseline_last_n,
        "No shrinkage": _model(replace(cfg, prior_scale=0.0)),
        "Shrinkage + 35-day rate decay": _model(replace(cfg, half_life_days=35.0)),
        "ninecat (tuned)": _model(cfg),
    }


def _errors(pred: pd.DataFrame, truth: pd.DataFrame) -> dict[str, float]:
    """Weighted MSE per stat (weights: future games, or future attempts for %)."""
    out = {}
    for stat in EVAL_STATS:
        if stat in PERCENT_STATS:
            w = truth[PERCENT_STATS[stat][1] + "_total"]
        else:
            w = truth["gp"]
        p, y = pred[stat].reindex(truth.index), truth[stat]
        ok = p.notna() & y.notna() & (w > 0)
        out[stat] = float(np.average((p[ok] - y[ok]) ** 2, weights=w[ok]))
    return out


def projection_backtest(
    seeds=range(1, 6),
    checkpoint_weeks=(4, 8, 12, 16),
    models: dict | None = None,
    min_games_before: int = 5,
    min_games_after: int = 10,
) -> pd.DataFrame:
    """Long table of (seed, week, model, stat, mse)."""
    models = models or default_models()
    rows = []
    for seed in seeds:
        season, _league, _ = generate_season(SyntheticConfig(seed=seed))
        end = season.n_weeks * DAYS_PER_WEEK
        for week in checkpoint_weeks:
            as_of = week * DAYS_PER_WEEK
            before = season.logs_before(as_of).groupby("player_id").size()
            truth = _per_game(season.logs_between(as_of, end))
            keep = truth.index[(truth["gp"] >= min_games_after)].intersection(
                before.index[before >= min_games_before])
            truth = truth.loc[keep]
            for name, fn in models.items():
                for stat, mse in _errors(fn(season, as_of), truth).items():
                    rows.append({"seed": seed, "week": week, "model": name, "stat": stat, "mse": mse})
    return pd.DataFrame(rows)


def summarize_backtest(results: pd.DataFrame, baseline: str = "Season average") -> pd.DataFrame:
    """RMSE per stat and model, plus the average error reduction vs. the baseline."""
    mse = results.groupby(["model", "stat"])["mse"].mean().unstack("stat")[EVAL_STATS]
    rmse = np.sqrt(mse)
    skill = 1 - mse / mse.loc[baseline]
    rmse["MSE reduction vs season avg"] = skill.mean(axis=1)
    return rmse.sort_values("MSE reduction vs season avg")


def tune_projections(
    seeds=(101, 102, 103),
    half_lives=(10.0, 20.0, 35.0, 60.0, float("inf")),
    prior_scales=(0.0, 0.5, 1.0, 2.0, 4.0),
    checkpoint_weeks=(4, 8, 12, 16),
) -> pd.DataFrame:
    """Mean MSE reduction vs. season average for each (half-life, prior scale)."""
    models = {"Season average": _baseline_season_avg}
    for hl, ps in itertools.product(half_lives, prior_scales):
        models[f"{hl}|{ps}"] = _model(ProjectionConfig(half_life_days=hl, prior_scale=ps))
    res = projection_backtest(seeds, checkpoint_weeks, models)
    summary = summarize_backtest(res)["MSE reduction vs season avg"].drop("Season average")
    idx = pd.MultiIndex.from_tuples([tuple(float(x) for x in k.split("|")) for k in summary.index],
                                    names=["half_life_days", "prior_scale"])
    return pd.Series(summary.to_numpy(), index=idx).unstack("prior_scale")


# ---------------------------------------------------------------------------
# 3. Calibration of the matchup simulator
# ---------------------------------------------------------------------------

SIM_VARIANTS = {
    "ninecat (NB + parameter uncertainty)": dict(parameter_uncertainty=True, overdispersion=True),
    "Negative binomial only": dict(parameter_uncertainty=False, overdispersion=True),
    "Poisson, point estimates": dict(parameter_uncertainty=False, overdispersion=False),
}


def calibration(seeds=range(1, 4), weeks=range(3, 24), n_sims: int = 2000) -> pd.DataFrame:
    """Predicted vs. realised outcomes for every category and matchup.

    Projections are made at the start of each week using only earlier games;
    the realised result comes from that week's actual box scores.
    """
    rows = []
    for seed in seeds:
        season, league, _ = generate_season(SyntheticConfig(seed=seed))
        for week in weeks:
            start, end = week * DAYS_PER_WEEK, (week + 1) * DAYS_PER_WEEK
            proj = project(season, start)
            games = season.player_games(start, end)
            week_logs = season.logs_between(start, end)
            ids = sorted(league.rostered())
            sims = {name: WeekSimulator(proj, games.loc[ids], n_sims=n_sims, seed=seed * 1000 + week, **kw)
                    for name, kw in SIM_VARIANTS.items()}
            for _, m in league.matchups[league.matchups["week"] == week].iterrows():
                ra, rb = league.rosters[m["team_a"]], league.rosters[m["team_b"]]
                actual = {}
                for key, roster in (("a", ra), ("b", rb)):
                    rows_ = week_logs[week_logs["player_id"].isin(roster)]
                    actual[key] = {s: np.array([float(rows_[s].sum())]) for s in [*COUNT_STATS, "fgm", "ftm"]}
                real = category_outcomes(actual["a"], actual["b"])[:, 0]
                won, lost = (real == 1).sum(), (real == 0).sum()
                real_match = 1.0 if won > lost else 0.5 if won == lost else 0.0
                for name, sim in sims.items():
                    res = sim.matchup(ra, rb)
                    for cat, p in res.category_win_prob.items():
                        rows.append({"seed": seed, "week": week, "variant": name, "kind": "category",
                                     "target": cat, "p": p, "y": real[CATEGORIES.index(cat)]})
                    rows.append({"seed": seed, "week": week, "variant": name, "kind": "matchup",
                                 "target": "matchup", "p": res.win_prob + 0.5 * res.tie_prob, "y": real_match})
    return pd.DataFrame(rows)


def calibration_metrics(cal: pd.DataFrame) -> pd.DataFrame:
    def metrics(g):
        p = g["p"].clip(1e-3, 1 - 1e-3)
        y = g["y"]
        return pd.Series({
            "Brier": float(((p - y) ** 2).mean()),
            "Log loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
            "ECE": _ece(p, y),
            "n": len(g),
        })
    return cal.groupby(["kind", "variant"]).apply(metrics, include_groups=False)


def _ece(p: pd.Series, y: pd.Series, bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    b = np.minimum((p * bins).astype(int), bins - 1)
    df = pd.DataFrame({"b": b, "p": p, "y": y}).groupby("b").agg(p=("p", "mean"), y=("y", "mean"), n=("p", "size"))
    return float((df["n"] * (df["p"] - df["y"]).abs()).sum() / df["n"].sum())


def reliability_table(cal: pd.DataFrame, kind: str = "category", bins: int = 10) -> pd.DataFrame:
    d = cal[cal["kind"] == kind].copy()
    d["bin"] = np.minimum((d["p"] * bins).astype(int), bins - 1)
    return d.groupby(["variant", "bin"]).agg(p=("p", "mean"), y=("y", "mean"), n=("p", "size")).reset_index()


# ---------------------------------------------------------------------------
# 4. G-scores vs Z-scores
# ---------------------------------------------------------------------------

def g_vs_z(seeds=range(1, 6), week: int = 10, n_pairs: int = 300, n_sims: int = 1000,
           n_candidates: int = 156, roster_size: int = 13) -> pd.DataFrame:
    """Simulated marginal value of each player, next to his G and Z totals.

    For each candidate we add him to many random 12-man teams drawn from the
    rostered pool and simulate a typical week (3.5 games) against random full
    teams. His value is the average gain in expected categories won and in
    matchup win probability. The ranking that tracks this value better is
    the better guide to in-season decisions such as trades and pickups.
    """
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        season, _league, _ = generate_season(SyntheticConfig(seed=seed))
        proj = project(season, week * DAYS_PER_WEEK)
        g = category_scores(proj, "g")["total"]
        z = category_scores(proj, "z")["total"]
        pool_size = 12 * roster_size
        pool = ((g.rank(ascending=False) + z.rank(ascending=False)) / 2).nsmallest(pool_size).index.to_numpy()
        candidates = pool[:n_candidates] if n_candidates <= pool_size else pool
        games = pd.Series(3.5, index=pool)
        sim = WeekSimulator(proj, games, n_sims=n_sims, seed=seed)

        teams_a = [rng.choice(pool, roster_size - 1, replace=False) for _ in range(n_pairs)]
        teams_b = [rng.choice(pool, roster_size, replace=False) for _ in range(n_pairs)]
        tot_a = {s: np.stack([sim.totals(t)[s] for t in teams_a]) for s in sim.draws}  # (pairs, S)
        tot_b = {s: np.stack([sim.totals(t)[s] for t in teams_b]) for s in sim.draws}
        members = [set(t) for t in teams_a]
        for pid in candidates:
            d = sim.player_draws(pid)
            ok = np.array([pid not in m for m in members])  # never add a player to his own team
            out = category_outcomes({s: tot_a[s][ok] + d[s][None, :] for s in tot_a},
                                    {s: v[ok] for s, v in tot_b.items()})  # (9, pairs, S)
            won, lost = (out == 1).sum(0), (out == 0).sum(0)
            rows.append({
                "seed": seed, "player_id": pid, "g": g[pid], "z": z[pid],
                "exp_categories": float(out.sum(0).mean()),
                "win_prob": float(((won > lost) + 0.5 * (won == lost)).mean()),
            })
    return pd.DataFrame(rows)


def g_vs_z_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for target in ("exp_categories", "win_prob"):
        for score in ("g", "z"):
            rho = df.groupby("seed").apply(
                lambda d: d[score].rank().corr(d[target].rank()), include_groups=False)  # Spearman
            out.append({"value measured by": target, "score": score.upper(),
                        "Spearman rho (mean over seeds)": rho.mean(), "min": rho.min(), "max": rho.max()})
    return pd.DataFrame(out)
