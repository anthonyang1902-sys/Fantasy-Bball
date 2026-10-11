"""Core data containers and CSV input/output.

A season is three tables:

* ``players``  - one row per player: ``player_id, name, nba_team, position``
* ``logs``     - one row per game *played*: ``player_id, day, nba_team`` + box stats
* ``schedule`` - one row per NBA team game: ``day, nba_team``

Days are integers counted from the season's opening day; fantasy weeks are
``day // 7``. Real data can be converted into this format with
``scripts/fetch_nba_logs.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .categories import BOX_STATS

DAYS_PER_WEEK = 7


@dataclass
class Season:
    players: pd.DataFrame
    logs: pd.DataFrame
    schedule: pd.DataFrame
    n_weeks: int

    def __post_init__(self) -> None:
        validate_logs(self.logs)
        self.players = self.players.set_index("player_id", drop=False)

    def logs_before(self, day: int) -> pd.DataFrame:
        return self.logs[self.logs["day"] < day]

    def logs_between(self, start: int, end: int) -> pd.DataFrame:
        return self.logs[(self.logs["day"] >= start) & (self.logs["day"] < end)]

    def team_games(self, start: int, end: int) -> pd.Series:
        """Number of games each NBA team plays in ``[start, end)``."""
        sched = self.schedule[(self.schedule["day"] >= start) & (self.schedule["day"] < end)]
        return sched.groupby("nba_team").size()

    def player_games(self, start: int, end: int, player_ids=None) -> pd.Series:
        """Scheduled games for each player's NBA team in ``[start, end)``."""
        players = self.players if player_ids is None else self.players.loc[list(player_ids)]
        counts = self.team_games(start, end)
        return players["nba_team"].map(counts).fillna(0).astype(float)

    @property
    def last_day(self) -> int:
        return int(self.schedule["day"].max())


@dataclass
class League:
    """A fantasy league: rosters plus a head-to-head schedule."""

    rosters: dict[str, list[int]]
    matchups: pd.DataFrame  # columns: week, team_a, team_b
    roster_size: int = 13

    @property
    def teams(self) -> list[str]:
        return list(self.rosters)

    def rostered(self) -> set[int]:
        return {p for r in self.rosters.values() for p in r}

    def free_agents(self, all_player_ids) -> list[int]:
        taken = self.rostered()
        return [p for p in all_player_ids if p not in taken]

    def opponent(self, team: str, week: int) -> str | None:
        m = self.matchups[self.matchups["week"] == week]
        for _, row in m.iterrows():
            if row["team_a"] == team:
                return row["team_b"]
            if row["team_b"] == team:
                return row["team_a"]
        return None

    def with_trade(self, team_a: str, send_a: list[int], team_b: str, send_b: list[int]) -> "League":
        rosters = {t: list(r) for t, r in self.rosters.items()}
        rosters[team_a] = [p for p in rosters[team_a] if p not in send_a] + list(send_b)
        rosters[team_b] = [p for p in rosters[team_b] if p not in send_b] + list(send_a)
        return League(rosters, self.matchups, self.roster_size)


def validate_logs(logs: pd.DataFrame) -> None:
    missing = {"player_id", "day", *BOX_STATS} - set(logs.columns)
    if missing:
        raise ValueError(f"game logs are missing columns: {sorted(missing)}")
    if (logs["fgm"] > logs["fga"]).any() or (logs["ftm"] > logs["fta"]).any():
        raise ValueError("game logs contain makes greater than attempts")


def round_robin(teams: list[str], n_weeks: int) -> pd.DataFrame:
    """Circle-method round robin, repeated until ``n_weeks`` weeks are filled."""
    teams = list(teams)
    if len(teams) % 2:
        teams.append("BYE")
    n = len(teams)
    rows = []
    order = teams[:]
    for week in range(n_weeks):
        for i in range(n // 2):
            a, b = order[i], order[n - 1 - i]
            if "BYE" not in (a, b):
                rows.append({"week": week, "team_a": a, "team_b": b})
        order = [order[0]] + [order[-1]] + order[1:-1]
    return pd.DataFrame(rows)


def save_season(season: Season, league: League | None, directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    season.players.to_csv(directory / "players.csv", index=False)
    season.logs.to_csv(directory / "logs.csv", index=False)
    season.schedule.to_csv(directory / "schedule.csv", index=False)
    if league is not None:
        rows = [{"team": t, "player_id": p} for t, r in league.rosters.items() for p in r]
        pd.DataFrame(rows).to_csv(directory / "rosters.csv", index=False)
        league.matchups.to_csv(directory / "matchups.csv", index=False)


def load_season(directory: str | Path) -> tuple[Season, League | None]:
    directory = Path(directory)
    players = pd.read_csv(directory / "players.csv")
    logs = pd.read_csv(directory / "logs.csv")
    schedule = pd.read_csv(directory / "schedule.csv")
    n_weeks = int(np.ceil((schedule["day"].max() + 1) / DAYS_PER_WEEK))
    season = Season(players, logs, schedule, n_weeks)
    league = None
    if (directory / "rosters.csv").exists():
        r = pd.read_csv(directory / "rosters.csv")
        rosters = {t: g["player_id"].tolist() for t, g in r.groupby("team", sort=False)}
        matchups_path = directory / "matchups.csv"
        matchups = pd.read_csv(matchups_path) if matchups_path.exists() else round_robin(list(rosters), n_weeks)
        league = League(rosters, matchups)
    return season, league
