"""Download real NBA game logs into ninecat's CSV format.

    pip install -e ".[nba]"
    python scripts/fetch_nba_logs.py --season 2025-26 --out data/real \
        [--schedule schedule.csv] [--rosters my_league.csv]

Writes ``players.csv``, ``logs.csv`` and ``schedule.csv`` (plus
``rosters.csv`` if you pass your league), which ``ninecat.load_season``
reads. Uses the public stats.nba.com endpoints through ``nba_api``, so it
needs network access and is not run in CI.

* ``--schedule``: CSV with ``date,team`` rows for games that have not been
  played yet (needed for weekly matchups and streaming; past games are
  taken from the logs automatically).
* ``--rosters``: CSV with ``team,player_name`` rows for your fantasy league.

stats.nba.com does not report positions in game logs, so every player gets
position ``UNK`` and projections shrink toward the league-wide prior.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ninecat.data import League, Season, round_robin, save_season, DAYS_PER_WEEK


def fetch_logs(season: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import playergamelogs

    df = playergamelogs.PlayerGameLogs(season_nullable=season, season_type_nullable="Regular Season")
    return df.get_data_frames()[0]


def to_ninecat(raw: pd.DataFrame, future: pd.DataFrame | None) -> Season:
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["GAME_DATE"]).dt.normalize()
    start = raw["date"].min()
    # Align day 0 with a Monday so fantasy weeks run Monday-Sunday.
    start = start - pd.Timedelta(days=start.weekday())
    raw["day"] = (raw["date"] - start).dt.days
    raw = raw[raw["MIN"].astype(float) > 0]

    logs = pd.DataFrame({
        "player_id": raw["PLAYER_ID"].astype(int),
        "day": raw["day"].astype(int),
        "nba_team": raw["TEAM_ABBREVIATION"],
        "min": raw["MIN"].astype(float),
        "fgm": raw["FGM"], "fga": raw["FGA"], "fg3m": raw["FG3M"], "ftm": raw["FTM"], "fta": raw["FTA"],
        "pts": raw["PTS"], "reb": raw["REB"], "ast": raw["AST"], "stl": raw["STL"], "blk": raw["BLK"],
        "tov": raw["TOV"],
    })

    latest = raw.sort_values("date").groupby("PLAYER_ID").tail(1)  # current team after trades
    players = pd.DataFrame({
        "player_id": latest["PLAYER_ID"].astype(int),
        "name": latest["PLAYER_NAME"],
        "nba_team": latest["TEAM_ABBREVIATION"],
        "position": "UNK",
    })

    schedule = raw[["day", "TEAM_ABBREVIATION"]].drop_duplicates().rename(columns={"TEAM_ABBREVIATION": "nba_team"})
    if future is not None:
        f = future.assign(day=(pd.to_datetime(future["date"]).dt.normalize() - start).dt.days)
        f = f[f["day"] > schedule["day"].max()].rename(columns={"team": "nba_team"})[["day", "nba_team"]]
        schedule = pd.concat([schedule, f], ignore_index=True)
    n_weeks = int(schedule["day"].max() // DAYS_PER_WEEK + 1)
    return Season(players, logs, schedule.sort_values(["day", "nba_team"]).reset_index(drop=True), n_weeks)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--season", default="2025-26")
    p.add_argument("--out", default="data/real")
    p.add_argument("--schedule", help="CSV of future games: date,team")
    p.add_argument("--rosters", help="CSV of your league: team,player_name")
    args = p.parse_args()

    future = pd.read_csv(args.schedule) if args.schedule else None
    season = to_ninecat(fetch_logs(args.season), future)

    league = None
    if args.rosters:
        r = pd.read_csv(args.rosters)
        ids = season.players.set_index("name")["player_id"]
        missing = sorted(set(r["player_name"]) - set(ids.index))
        if missing:
            print(f"warning: {len(missing)} roster names not found in game logs: {missing[:10]}")
        r = r[r["player_name"].isin(ids.index)]
        rosters = {t: [int(ids[n]) for n in g["player_name"]] for t, g in r.groupby("team", sort=False)}
        league = League(rosters, round_robin(list(rosters), season.n_weeks))

    save_season(season, league, Path(args.out))
    print(f"wrote {len(season.logs)} game logs for {len(season.players)} players to {args.out}")


if __name__ == "__main__":
    main()
