"""A seeded generator of realistic NBA seasons for testing and backtesting.

Real game logs only show one realisation of each player's season, so it is
hard to tell whether a projection was good or lucky. The generator gives us
seasons whose data-generating process we control, while keeping the
features that make in-season projection hard:

* talent varies by position and is correlated across stats (usage drives
  shots, points, assists and turnovers together);
* players change *during* the season: young players improve, old players
  decline, form drifts week to week, and some players gain or lose a role;
* injuries come in multi-game spells and some players are more prone to them;
* box scores are internally consistent: 3PM <= FGM <= FGA and
  PTS = 2*FG2M + 3*FG3M + FTM, with game-level "hot nights" that make counts
  overdispersed relative to Poisson.

Every number here is a modelling assumption chosen to look like a modern NBA
season, not an estimate from real data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import DAYS_PER_WEEK, League, Season, round_robin

POSITIONS = np.array(["G", "F", "C"])

# Per-36-minute league baselines by position.
BASE_RATES = {
    #        fga   fta  reb   ast  stl  blk  tov
    "G": [16.0, 4.2, 4.6, 6.2, 1.30, 0.35, 2.4],
    "F": [14.5, 4.0, 7.2, 3.2, 1.10, 0.75, 1.8],
    "C": [12.5, 4.6, 11.8, 2.6, 0.90, 1.70, 2.0],
}
RATE_STATS = ["fga", "fta", "reb", "ast", "stl", "blk", "tov"]
THREE_RATE = {"G": 0.43, "F": 0.38, "C": 0.12}
TWO_PCT = {"G": 0.51, "F": 0.53, "C": 0.60}
FT_PCT = {"G": 0.82, "F": 0.77, "C": 0.70}

FIRST = ["Jalen", "Marcus", "Tyrese", "Devin", "Andre", "Luka", "Nikola", "Jaylen", "Darius", "Malik",
         "Cade", "Evan", "Scottie", "Paolo", "Franz", "Trey", "Mikal", "Jordan", "Derrick", "Kevin",
         "Anthony", "Bam", "Jarrett", "Isaiah", "Keegan", "Desmond", "Aaron", "Zion", "Brandon", "Kyle",
         "Josh", "Victor", "Chet", "Alperen", "Dyson", "Coby", "Herb", "Immanuel", "Walker", "Ausar"]
LAST = ["Walker", "Brooks", "Haliburton", "Banks", "Carter", "Ivanov", "Petrovic", "Brown", "Garland",
        "Monk", "Barrett", "Mobley", "Holmes", "Banchero", "Wagner", "Murphy", "Bridges", "Poole", "White",
        "Durant", "Edwards", "Adebayo", "Allen", "Stewart", "Murray", "Bane", "Gordon", "Williams", "Miller",
        "Lowry", "Hart", "Wembley", "Holmgren", "Sengun", "Daniels", "Jones", "Jones", "Quickley", "Kessler",
        "Thompson", "Grant", "Okafor", "Nembhard", "Reid", "Vassell", "Suggs", "Mathurin", "Kuminga"]


@dataclass
class SyntheticConfig:
    seed: int = 0
    n_nba_teams: int = 30
    players_per_team: int = 14
    n_weeks: int = 24
    n_fantasy_teams: int = 12
    roster_size: int = 13
    draft_noise: float = 1.5  # how far managers' draft boards stray from true value


def _week_schedule(rng: np.random.Generator, n_teams: int, n_weeks: int) -> pd.DataFrame:
    rows = []
    for t in range(n_teams):
        for w in range(n_weeks):
            n_games = rng.choice([2, 3, 4], p=[0.12, 0.44, 0.44])
            days = rng.choice(DAYS_PER_WEEK, size=n_games, replace=False)
            rows += [{"day": w * DAYS_PER_WEEK + int(d), "nba_team": f"T{t:02d}"} for d in sorted(days)]
    return pd.DataFrame(rows).sort_values(["day", "nba_team"]).reset_index(drop=True)


def _players(rng: np.random.Generator, cfg: SyntheticConfig) -> pd.DataFrame:
    n = cfg.n_nba_teams * cfg.players_per_team
    pos = rng.choice(POSITIONS, size=n, p=[0.40, 0.35, 0.25])
    names, seen = [], set()
    while len(names) < n:
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        if name not in seen:
            seen.add(name)
            names.append(name)
    return pd.DataFrame({
        "player_id": np.arange(n),
        "name": names,
        "nba_team": [f"T{i // cfg.players_per_team:02d}" for i in range(n)],
        "position": pos,
        "age": rng.integers(20, 36, size=n),
    })


def _latent(rng: np.random.Generator, players: pd.DataFrame) -> dict[str, np.ndarray]:
    """Draw each player's true (hidden) talent parameters."""
    n = len(players)
    pos = players["position"].to_numpy()
    age = players["age"].to_numpy()
    talent = rng.normal(size=n)
    usage = np.clip(np.exp(0.15 * talent + rng.normal(0, 0.17, n)), 0.65, 1.55)

    base = np.array([BASE_RATES[p] for p in pos])  # (n, 7)
    idio = np.exp(rng.normal(0, 0.22, size=base.shape))
    skill = np.exp(0.10 * talent)[:, None]
    rates = base * idio * skill
    rates[:, 0] = base[:, 0] * usage * np.exp(rng.normal(0, 0.08, n))   # fga
    rates[:, 1] *= usage                                                 # fta
    rates[:, 3] *= np.sqrt(usage)                                        # ast
    rates[:, 6] = base[:, 6] * usage ** 0.8 * np.exp(rng.normal(0, 0.15, n))  # tov
    rates = np.clip(rates, 0.35 * base, 2.2 * base)

    three_rate = np.clip(np.array([THREE_RATE[p] for p in pos]) + rng.normal(0, 0.10, n), 0.0, 0.75)
    p3 = np.clip(rng.normal(0.355, 0.035, n), 0.25, 0.45)
    p2 = np.clip(np.array([TWO_PCT[p] for p in pos]) + 0.012 * talent + rng.normal(0, 0.035, n), 0.40, 0.72)
    ftp = np.clip(np.array([FT_PCT[p] for p in pos]) + rng.normal(0, 0.07, n), 0.45, 0.93)

    minutes = np.clip(21 + 6.5 * talent + rng.normal(0, 3, n), 6, 37)
    growth = np.where(age <= 23, rng.normal(0.10, 0.06, n),
                      np.where(age >= 32, rng.normal(-0.06, 0.04, n), rng.normal(0, 0.03, n)))
    role_change = rng.random(n) < 0.25
    role_day = rng.integers(20, 140, n)
    role_shift = np.where(role_change, rng.normal(0, 6, n), 0.0)
    injury_prone = rng.beta(1.2, 60, n)
    return dict(talent=talent, rates=rates, three_rate=three_rate, p3=p3, p2=p2, ftp=ftp,
                minutes=minutes, growth=growth, role_day=role_day, role_shift=role_shift,
                injury_prone=injury_prone)


def _simulate_logs(rng, players, schedule, lat, n_days) -> pd.DataFrame:
    n = len(players)
    team_of = players["nba_team"].to_numpy()
    games_by_day = schedule.groupby("day")["nba_team"].apply(set).to_dict()
    n_weeks = n_days // DAYS_PER_WEEK + 1
    form = np.zeros((n_weeks, n))
    for w in range(1, n_weeks):
        form[w] = 0.8 * form[w - 1] + rng.normal(0, 0.06, n)

    injured_until = np.full(n, -1)
    frames = []
    for day in range(n_days):
        teams_today = games_by_day.get(day)
        if not teams_today:
            continue
        plays = np.isin(team_of, list(teams_today))
        # injuries start as multi-game spells
        new_injury = plays & (injured_until < day) & (rng.random(n) < lat["injury_prone"])
        injured_until[new_injury] = day + 2 * rng.geometric(1 / 6, new_injury.sum())
        rest = rng.random(n) < 0.015
        active = plays & (injured_until < day) & ~rest
        idx = np.flatnonzero(active)
        if idx.size == 0:
            continue

        frac = day / n_days
        mult = np.exp(lat["growth"][idx] * frac + form[day // DAYS_PER_WEEK, idx])
        mean_min = lat["minutes"][idx] + np.where(day >= lat["role_day"][idx], lat["role_shift"][idx], 0)
        mins = np.clip(rng.normal(np.clip(mean_min, 4, 38), 4.5), 4, 44)
        hot = rng.gamma(14, 1 / 14, idx.size)
        lam = lat["rates"][idx] / 36 * (mins * mult * hot)[:, None]
        counts = rng.poisson(lam)
        fga, fta, reb, ast, stl, blk, tov = counts.T
        fg3a = rng.binomial(fga, lat["three_rate"][idx])
        fg3m = rng.binomial(fg3a, lat["p3"][idx])
        fg2m = rng.binomial(fga - fg3a, lat["p2"][idx])
        ftm = rng.binomial(fta, lat["ftp"][idx])
        frames.append(pd.DataFrame({
            "player_id": idx, "day": day, "nba_team": team_of[idx], "min": np.round(mins, 1),
            "fgm": fg2m + fg3m, "fga": fga, "fg3m": fg3m, "ftm": ftm, "fta": fta,
            "pts": 2 * fg2m + 3 * fg3m + ftm, "reb": reb, "ast": ast, "stl": stl, "blk": blk, "tov": tov,
        }))
    return pd.concat(frames, ignore_index=True)


def _preseason_value(lat: dict[str, np.ndarray]) -> np.ndarray:
    """Rough 9-cat value of each player's true talent, used to seed the draft."""
    m = lat["minutes"] / 36
    r = lat["rates"] * m[:, None]
    fga, fta = r[:, 0], r[:, 1]
    fg3m = fga * lat["three_rate"] * lat["p3"]
    fgm = fg3m + fga * (1 - lat["three_rate"]) * lat["p2"]
    ftm = fta * lat["ftp"]
    pts = 2 * fgm + fg3m + ftm
    cols = [fg3m, pts, r[:, 2], r[:, 3], r[:, 4], r[:, 5], -r[:, 6],
            fga * (fgm / np.maximum(fga, 1e-9) - 0.47), fta * (ftm / np.maximum(fta, 1e-9) - 0.77)]
    x = np.column_stack(cols)
    top = np.argsort(-lat["minutes"])[:200]
    return ((x - x[top].mean(0)) / x[top].std(0)).sum(1)


def _draft(rng, value: np.ndarray, cfg: SyntheticConfig) -> dict[str, list[int]]:
    board = list(np.argsort(-(value + rng.normal(0, cfg.draft_noise, value.size))))
    teams = [f"Team {chr(ord('A') + i)}" for i in range(cfg.n_fantasy_teams)]
    rosters: dict[str, list[int]] = {t: [] for t in teams}
    for rnd in range(cfg.roster_size):
        order = teams if rnd % 2 == 0 else teams[::-1]
        for t in order:
            rosters[t].append(int(board.pop(0)))
    return rosters


def generate_season(cfg: SyntheticConfig | None = None) -> tuple[Season, League, dict]:
    """Generate a season, a drafted fantasy league, and the hidden truth.

    The third return value holds the latent player parameters; use it only for
    evaluation, never as an input to a model.
    """
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    players = _players(rng, cfg)
    schedule = _week_schedule(rng, cfg.n_nba_teams, cfg.n_weeks)
    lat = _latent(rng, players)
    n_days = cfg.n_weeks * DAYS_PER_WEEK
    logs = _simulate_logs(rng, players, schedule, lat, n_days)
    rosters = _draft(rng, _preseason_value(lat), cfg)
    league = League(rosters, round_robin(list(rosters), cfg.n_weeks), cfg.roster_size)
    season = Season(players, logs, schedule, cfg.n_weeks)
    return season, league, lat
