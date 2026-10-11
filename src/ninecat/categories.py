"""The nine standard head-to-head categories and the box-score stats behind them."""

from __future__ import annotations

# Box-score columns every game log must have.
BOX_STATS = ["min", "fgm", "fga", "fg3m", "ftm", "fta", "pts", "reb", "ast", "stl", "blk", "tov"]

# Stats modelled as overdispersed counts (rate per minute x minutes).
COUNT_STATS = ["fga", "fta", "fg3m", "pts", "reb", "ast", "stl", "blk", "tov"]

# Percentage categories: name -> (makes column, attempts column).
PERCENT_STATS = {"fg_pct": ("fgm", "fga"), "ft_pct": ("ftm", "fta")}

CATEGORIES = ["fg_pct", "ft_pct", "fg3m", "pts", "reb", "ast", "stl", "blk", "tov"]

# Categories where the lower total wins.
NEGATIVE = {"tov"}

LABELS = {
    "fg_pct": "FG%",
    "ft_pct": "FT%",
    "fg3m": "3PM",
    "pts": "PTS",
    "reb": "REB",
    "ast": "AST",
    "stl": "STL",
    "blk": "BLK",
    "tov": "TO",
}

COUNTING_CATEGORIES = [c for c in CATEGORIES if c not in PERCENT_STATS]

# Raw totals tracked during a week (enough to rebuild every category).
RAW_STATS_FOR_ACCRUAL = [*COUNT_STATS, "fgm", "ftm"]
