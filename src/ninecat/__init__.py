"""ninecat: in-season analytics for 9-category head-to-head fantasy basketball."""

from .categories import CATEGORIES, LABELS
from .data import League, Season, load_season, save_season
from .inseason import evaluate_trade, simulate_week, streaming_moves, typical_week_games, week_context
from .projections import ProjectionConfig, Projections, project
from .scoring import category_scores, rankings, team_profile
from .simulation import WeekSimulator, evaluate_matchup
from .synthetic import SyntheticConfig, generate_season

__all__ = [
    "CATEGORIES", "LABELS", "League", "Season", "load_season", "save_season",
    "evaluate_trade", "simulate_week", "streaming_moves", "typical_week_games", "week_context",
    "ProjectionConfig", "Projections", "project", "category_scores", "rankings", "team_profile",
    "WeekSimulator", "evaluate_matchup", "SyntheticConfig", "generate_season",
]
__version__ = "0.1.0"
