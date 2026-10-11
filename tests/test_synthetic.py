import numpy as np

from ninecat.data import load_season, save_season


def test_box_scores_are_internally_consistent(world):
    logs = world[0].logs
    assert (logs["fgm"] <= logs["fga"]).all()
    assert (logs["fg3m"] <= logs["fgm"]).all()
    assert (logs["ftm"] <= logs["fta"]).all()
    # PTS = 2 * FGM + 3PM + FTM  (2 * two-pointers + 3 * threes + free throws)
    assert (logs["pts"] == 2 * logs["fgm"] + logs["fg3m"] + logs["ftm"]).all()


def test_players_only_log_games_their_team_played(world):
    season = world[0]
    sched = set(map(tuple, season.schedule[["day", "nba_team"]].to_numpy()))
    played = set(map(tuple, season.logs[["day", "nba_team"]].to_numpy()))
    assert played <= sched


def test_schedule_and_league_shape(world):
    season, league, _ = world
    per_team = season.schedule.groupby("nba_team").size()
    assert per_team.between(70, 90).all()
    assert len(league.rosters) == 12
    assert all(len(r) == 13 for r in league.rosters.values())
    assert len(league.rostered()) == 12 * 13
    # every team plays exactly once per week
    for _, wk in league.matchups.groupby("week"):
        teams = np.concatenate([wk["team_a"], wk["team_b"]])
        assert len(teams) == len(set(teams)) == 12


def test_round_trip_csv(world, tmp_path):
    season, league, _ = world
    save_season(season, league, tmp_path)
    s2, l2 = load_season(tmp_path)
    assert len(s2.logs) == len(season.logs)
    assert l2.rosters == league.rosters
