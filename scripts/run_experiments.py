"""Run every experiment and write reports/results.md plus figures.

    python scripts/run_experiments.py           # full run (several minutes)
    python scripts/run_experiments.py --quick   # smaller run for a smoke test
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from ninecat import experiments as ex  # noqa: E402
from ninecat.categories import CATEGORIES, LABELS  # noqa: E402
from ninecat.projections import ProjectionConfig, project  # noqa: E402
from ninecat.scoring import category_scores  # noqa: E402
from ninecat.synthetic import SyntheticConfig, generate_season  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"

# Reference data-viz palette (light mode).
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
DIVERGING = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def md_table(df: pd.DataFrame, fmt: str = "{:.3f}", index: bool = True) -> str:
    d = df.reset_index() if index else df
    def cell(col, v):
        if col == "n":
            return f"{int(v):,}"
        return fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)

    cells = [[cell(c, v) for c, v in zip(d.columns, row)] for row in d.itertuples(index=False)]
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    return "\n".join([head, sep] + ["| " + " | ".join(r) + " |" for r in cells])


def fig_projection_skill(summary: pd.DataFrame) -> str:
    s = summary["MSE reduction vs season avg"].sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(s.index, s.to_numpy() * 100, color=SERIES[0], height=0.55)
    ax.axvline(0, color=MUTED, linewidth=1)
    for y, v in enumerate(s.to_numpy() * 100):
        ax.text(v + (0.6 if v >= 0 else -0.6), y, f"{v:+.1f}%", va="center",
                ha="left" if v >= 0 else "right", color=INK_2, fontsize=9)
    ax.set_xlabel("Mean squared error reduction vs. season average (avg. over 12 stats)")
    ax.set_title("Rest-of-season projection accuracy")
    ax.grid(axis="y", visible=False)
    lo, hi = min(s.min() * 100, 0), max(s.max() * 100, 0)
    ax.set_xlim(lo - 8, hi + 8)
    fig.tight_layout()
    path = FIGS / "projection_skill.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def fig_tuning(grid: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    v = grid.to_numpy() * 100
    lim = np.nanmax(np.abs(v))
    ax.imshow(v, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(grid.shape[1]), [f"{c:g}" for c in grid.columns])
    ax.set_yticks(range(grid.shape[0]), ["none" if not np.isfinite(i) else f"{i:g} d" for i in grid.index])
    ax.set_xlabel("Prior strength (x empirical-Bayes estimate)")
    ax.set_ylabel("Recency half-life of rates")
    best = np.unravel_index(np.nanargmax(v), v.shape)
    for (i, j), val in np.ndenumerate(v):
        ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=9,
                color=INK, fontweight="bold" if (i, j) == best else "normal")
    ax.grid(False)
    ax.set_title("Tuning: % MSE reduction vs. season average (held-out seeds)")
    fig.tight_layout()
    path = FIGS / "tuning_grid.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def fig_reliability(rel: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="Perfect calibration")
    for color, (name, g) in zip(SERIES, rel.groupby("variant", sort=False)):
        g = g[g["n"] >= 20]
        ax.plot(g["p"], g["y"], color=color, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
    ax.set_xlabel("Predicted probability of winning the category")
    ax.set_ylabel("Observed win rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Category win probabilities: reliability")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.tight_layout()
    path = FIGS / "reliability.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def fig_g_vs_z(df: pd.DataFrame, seed: int) -> str:
    d = df[df["seed"] == seed]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, score, color in ((axes[0], "z", SERIES[1]), (axes[1], "g", SERIES[0])):
        rho = d[score].rank().corr(d["exp_categories"].rank())
        ax.scatter(d[score], d["exp_categories"], s=14, color=color, alpha=0.8, edgecolors=SURFACE, linewidths=0.5)
        ax.set_title(f"{score.upper()}-score total  (Spearman {rho:.3f})")
        ax.set_xlabel(f"{score.upper()}-score total")
    axes[0].set_ylabel("Simulated expected categories won\nafter adding the player")
    fig.tight_layout()
    path = FIGS / "g_vs_z.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def fig_category_discount(seed: int = 1) -> tuple[str, pd.Series]:
    season, _, _ = generate_season(SyntheticConfig(seed=seed))
    proj = project(season, 10 * 7)
    g, z = category_scores(proj, "g"), category_scores(proj, "z")
    pool = z["total"].nlargest(156).index
    ratio = (g.loc[pool, CATEGORIES].std() / z.loc[pool, CATEGORIES].std()).sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar([LABELS[c] for c in ratio.index], ratio.to_numpy(), color=SERIES[0], width=0.55)
    for x, v in enumerate(ratio.to_numpy()):
        ax.text(x, v + 0.015, f"{v:.2f}", ha="center", color=INK_2, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Spread of G relative to Z")
    ax.set_title("How much G-scores discount each category")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    path = FIGS / "category_discount.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name, ratio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="fewer seeds and simulations")
    args = parser.parse_args()
    q = args.quick
    FIGS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    tune_seeds = (101, 102) if q else (101, 102, 103)
    eval_seeds = range(1, 3) if q else range(1, 6)
    print("tuning ...")
    grid = ex.tune_projections(seeds=tune_seeds)
    best_hl, best_ps = grid.stack().idxmax()
    tuned = ProjectionConfig(half_life_days=best_hl, prior_scale=best_ps)

    print("backtesting ...")
    bt = ex.projection_backtest(seeds=eval_seeds, models=ex.default_models(tuned))
    summary = ex.summarize_backtest(bt)

    print("calibration ...")
    cal = ex.calibration(seeds=range(1, 2) if q else range(1, 4), n_sims=1000 if q else 2000)
    cal_metrics = ex.calibration_metrics(cal)
    rel = ex.reliability_table(cal)

    print("G vs Z ...")
    gz = ex.g_vs_z(seeds=eval_seeds, n_pairs=150 if q else 300, n_sims=500 if q else 1000)
    gz_summary = ex.g_vs_z_summary(gz)

    figs = {
        "skill": fig_projection_skill(summary),
        "tuning": fig_tuning(grid),
        "reliability": fig_reliability(rel),
        "gz": fig_g_vs_z(gz, seed=1),
    }
    figs["discount"], discount = fig_category_discount()

    best_model = summary["MSE reduction vs season avg"].idxmax()
    gz_cat = gz_summary[gz_summary["value measured by"] == "exp_categories"].set_index("score")
    cat_m = cal_metrics.loc["category"]
    match_m = cal_metrics.loc["matchup"]

    naive, nb, full = list(ex.SIM_VARIANTS)[2], list(ex.SIM_VARIANTS)[1], list(ex.SIM_VARIANTS)[0]
    hl_text = "no decay" if not np.isfinite(best_hl) else f"a {best_hl:g}-day half-life"
    lines = [
        "# Results",
        "",
        f"_Generated by `scripts/run_experiments.py{' --quick' if q else ''}` in {time.time() - t0:.0f}s. "
        "All numbers come from synthetic seasons (see `docs/methodology.md`); tuning seeds "
        f"{list(tune_seeds)} never overlap evaluation seeds {list(eval_seeds)}._",
        "",
        "## 1. Projection tuning",
        "",
        f"The grid search picked **{hl_text}** on per-minute rates and a prior strength of "
        f"**{best_ps:g}x** the empirical-Bayes estimate. That the best prior scale sits near 1x is a check "
        "that the method-of-moments prior is about right: shrinking harder or softer both lose accuracy.",
        "",
        f"![tuning grid](figures/{figs['tuning']})",
        "",
        "## 2. Projection backtest",
        "",
        "Rest-of-season per-game projections made at weeks 4, 8, 12 and 16, scored against what each "
        "player actually averaged afterwards (players with at least 5 games before and 10 after). "
        "Error is weighted by games played (attempts for FG%/FT%).",
        "",
        f"![projection skill](figures/{figs['skill']})",
        "",
        "RMSE by stat (lower is better):",
        "",
        md_table(summary.rename(columns=LABELS | {"min": "MIN", "fga": "FGA", "fta": "FTA"})),
        "",
        f"Best model: **{best_model}**.",
        "",
        "## 3. Matchup simulator calibration",
        "",
        "Every league matchup from week 3 on is simulated at the start of the week, using only earlier "
        "games, then compared with the real result. A calibrated simulator's 70% calls come true 70% of "
        "the time.",
        "",
        f"![reliability](figures/{figs['reliability']})",
        "",
        "Per category:",
        "",
        md_table(cat_m, fmt="{:.4f}"),
        "",
        "Per matchup (overall win/loss):",
        "",
        md_table(match_m, fmt="{:.4f}"),
        "",
        f"Overdispersion is what matters most: moving from Poisson to negative-binomial counts cuts "
        f"category log loss from {cat_m.loc[naive, 'Log loss']:.4f} to {cat_m.loc[nb, 'Log loss']:.4f}. "
        f"Adding parameter uncertainty changes it to {cat_m.loc[full, 'Log loss']:.4f} per category and "
        f"{match_m.loc[nb, 'Log loss']:.4f} -> {match_m.loc[full, 'Log loss']:.4f} per matchup, a small "
        "effect at this point in the season because most rostered players already have large samples.",
        "",
        "## 4. G-scores vs Z-scores",
        "",
        "Each player in the rostered pool is added to hundreds of random teams and simulated against "
        "random opponents. His true marginal value is the average gain in categories won. The better "
        "ranking is the one that orders players the same way.",
        "",
        md_table(gz_summary, index=False),
        "",
        f"G-scores track simulated value better (Spearman {gz_cat.loc['G', 'Spearman rho (mean over seeds)']:.3f} "
        f"vs {gz_cat.loc['Z', 'Spearman rho (mean over seeds)']:.3f}). The gain comes from discounting "
        "volatile categories:",
        "",
        f"![g vs z](figures/{figs['gz']})",
        "",
        f"![category discount](figures/{figs['discount']})",
        "",
        "Caveat: the simulator's model of weekly variance is the same one G-scores assume, so this "
        "experiment shows G is the right summary *under that model*. Section 3 is the evidence that "
        "the model itself matches the data.",
        "",
    ]
    (REPORTS / "results.md").write_text("\n".join(lines))
    print(f"wrote {REPORTS / 'results.md'} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
