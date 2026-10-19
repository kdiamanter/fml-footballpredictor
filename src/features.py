"""Pre-match feature engineering without data leakage.

The rule: features for a match played on day D only use matches played
strictly before day D. In practice:

- every match is split into two "team rows" (one per team);
- each team's rows are sorted by date and *shifted by one* before the rolling
  window is applied, so a match never sees its own goals or xG;
- a team can only play once per day (checked below), so "previous matches of
  this team" is the same thing as "this team's matches before day D".

Two ways to get features, which must always agree (a test checks this):

- ``build_match_features``: all historical matches at once (fast, for training)
- ``build_fixture_features``: one fixture on any date, using only matches
  before that date (for predicting upcoming matches)
"""

from __future__ import annotations

import difflib

import numpy as np
import pandas as pd

WINDOW = 5          # last 5 matches for overall form
VENUE_WINDOW = 5    # last 5 home (or away) matches for venue form
VENUE_MIN = 3       # need at least 3 home (or away) matches for venue form

# Statistics averaged over the last WINDOW matches.
AVG_STATS = ["xg_for", "xg_against", "goals_for", "goals_against"]

# Per-team feature names (prefixed with home_/away_ in the match table).
TEAM_FEATURES = [
    "roll_xg_for",         # average xG created
    "roll_xg_against",     # average xG conceded (xGA)
    "roll_goals_for",      # average goals scored
    "roll_goals_against",  # average goals conceded
    "roll_points",         # points from the last 5 matches (0-15)
    "roll_win_rate",       # share of the last 5 matches won
    "venue_ppg",           # points per game in recent home (or away) matches
    "venue_xgd",           # xG minus xGA per game in recent home (or away) matches
]

# Home-minus-away differences, the main model inputs in Stage 2.
DIFF_FEATURES = [
    "xg_for_diff",
    "xg_against_diff",
    "goals_for_diff",
    "goals_against_diff",
    "points_diff",
    "win_rate_diff",
    "venue_ppg_diff",
    "venue_xgd_diff",
]

FEATURE_COLUMNS = (
    [f"home_{f}" for f in TEAM_FEATURES]
    + [f"away_{f}" for f in TEAM_FEATURES]
    + DIFF_FEATURES
)


class UnknownTeamError(ValueError):
    """The team does not appear anywhere in the data."""


class InsufficientHistoryError(ValueError):
    """The team exists but has too few earlier matches to build features."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def matches_before(matches: pd.DataFrame, date) -> pd.DataFrame:
    """Matches played strictly before the calendar day of ``date``."""
    day = pd.Timestamp(date).normalize()
    return matches[pd.to_datetime(matches["date"]).dt.normalize() < day]


def to_team_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Turn one row per match into two rows per match: one per team."""
    base = matches.reset_index(drop=True)
    home = pd.DataFrame({
        "match_idx": base.index,
        "date": pd.to_datetime(base["date"]),
        "team": base["home_team"],
        "opponent": base["away_team"],
        "venue": "home",
        "xg_for": base["home_xg"],
        "xg_against": base["away_xg"],
        "goals_for": base["home_goals"],
        "goals_against": base["away_goals"],
    })
    away = pd.DataFrame({
        "match_idx": base.index,
        "date": pd.to_datetime(base["date"]),
        "team": base["away_team"],
        "opponent": base["home_team"],
        "venue": "away",
        "xg_for": base["away_xg"],
        "xg_against": base["home_xg"],
        "goals_for": base["away_goals"],
        "goals_against": base["home_goals"],
    })
    rows = pd.concat([home, away], ignore_index=True)
    rows["points"] = np.select(
        [rows["goals_for"] > rows["goals_against"], rows["goals_for"] == rows["goals_against"]],
        [3, 1],
        default=0,
    )
    rows["win"] = (rows["goals_for"] > rows["goals_against"]).astype(float)

    # A team playing twice on one day would break the "strictly before D" rule.
    same_day = rows.assign(_day=rows["date"].dt.normalize()).duplicated(["team", "_day"], keep=False)
    clashes = rows[same_day]
    if not clashes.empty:
        raise ValueError(
            "A team plays more than once on the same day:\n"
            f"{clashes[['date', 'team', 'opponent']].head()}"
        )

    return rows.sort_values(["team", "date", "match_idx"], kind="mergesort").reset_index(drop=True)


def _past_rolling(series: pd.Series, groups, window: int, min_periods: int, how: str) -> pd.Series:
    """Rolling statistic of *previous* values only (shift by 1 first)."""
    def calc(s: pd.Series) -> pd.Series:
        rolled = s.shift(1).rolling(window, min_periods=min_periods)
        return rolled.sum() if how == "sum" else rolled.mean()

    return series.groupby(groups, sort=False).transform(calc)


def add_difference_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add home-minus-away difference columns."""
    df = df.copy()
    df["xg_for_diff"] = df["home_roll_xg_for"] - df["away_roll_xg_for"]
    df["xg_against_diff"] = df["home_roll_xg_against"] - df["away_roll_xg_against"]
    df["goals_for_diff"] = df["home_roll_goals_for"] - df["away_roll_goals_for"]
    df["goals_against_diff"] = df["home_roll_goals_against"] - df["away_roll_goals_against"]
    df["points_diff"] = df["home_roll_points"] - df["away_roll_points"]
    df["win_rate_diff"] = df["home_roll_win_rate"] - df["away_roll_win_rate"]
    df["venue_ppg_diff"] = df["home_venue_ppg"] - df["away_venue_ppg"]
    df["venue_xgd_diff"] = df["home_venue_xgd"] - df["away_venue_xgd"]
    return df


# ---------------------------------------------------------------------------
# All historical matches at once
# ---------------------------------------------------------------------------
def build_match_features(
    matches: pd.DataFrame,
    window: int = WINDOW,
    venue_window: int = VENUE_WINDOW,
    venue_min: int = VENUE_MIN,
) -> pd.DataFrame:
    """Return ``matches`` (sorted by date) with pre-match features added.

    Rows where a team lacks enough earlier matches get NaN features. Use
    ``drop_incomplete`` before modelling.
    """
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.sort_values(["date", "home_team", "away_team"], kind="mergesort").reset_index(drop=True)

    rows = to_team_matches(matches)

    # Overall form: this team's previous `window` matches.
    for stat in AVG_STATS:
        rows[f"roll_{stat}"] = _past_rolling(rows[stat], rows["team"], window, window, "mean")
    rows["roll_points"] = _past_rolling(rows["points"], rows["team"], window, window, "sum")
    rows["roll_win_rate"] = _past_rolling(rows["win"], rows["team"], window, window, "mean")

    # Venue form: this team's previous home matches (for home rows) or
    # previous away matches (for away rows).
    venue_groups = [rows["team"], rows["venue"]]
    rows["venue_ppg"] = _past_rolling(rows["points"], venue_groups, venue_window, venue_min, "mean")
    rows["xgd"] = rows["xg_for"] - rows["xg_against"]
    rows["venue_xgd"] = _past_rolling(rows["xgd"], venue_groups, venue_window, venue_min, "mean")

    # Back to one row per match.
    for side in ["home", "away"]:
        side_rows = rows[rows["venue"] == side].set_index("match_idx")[TEAM_FEATURES]
        side_rows = side_rows.add_prefix(f"{side}_")
        matches = matches.join(side_rows)

    return add_difference_features(matches)


def drop_incomplete(features: pd.DataFrame) -> pd.DataFrame:
    """Keep only matches where every feature could be calculated."""
    return features.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


# ---------------------------------------------------------------------------
# One fixture on any date (used for upcoming matches)
# ---------------------------------------------------------------------------
def _check_team_known(matches: pd.DataFrame, team: str) -> None:
    known = set(matches["home_team"]) | set(matches["away_team"])
    if team not in known:
        hint = difflib.get_close_matches(team, sorted(known), n=3)
        extra = f" Did you mean: {', '.join(hint)}?" if hint else ""
        raise UnknownTeamError(f"Unknown team '{team}'.{extra}")


def team_features_before(
    matches: pd.DataFrame,
    team: str,
    date,
    venue: str,
    window: int = WINDOW,
    venue_window: int = VENUE_WINDOW,
    venue_min: int = VENUE_MIN,
) -> dict[str, float]:
    """Features for one team using only its matches before ``date``."""
    _check_team_known(matches, team)
    history = matches_before(matches, date)
    rows = to_team_matches(history) if len(history) else pd.DataFrame(columns=["team"])
    rows = rows[rows["team"] == team]

    if len(rows) < window:
        raise InsufficientHistoryError(
            f"'{team}' has {len(rows)} matches before {pd.Timestamp(date).date()}; need {window}."
        )
    venue_rows = rows[rows["venue"] == venue]
    if len(venue_rows) < venue_min:
        raise InsufficientHistoryError(
            f"'{team}' has {len(venue_rows)} {venue} matches before "
            f"{pd.Timestamp(date).date()}; need {venue_min}."
        )

    last = rows.tail(window)
    last_venue = venue_rows.tail(venue_window)
    return {
        "roll_xg_for": last["xg_for"].mean(),
        "roll_xg_against": last["xg_against"].mean(),
        "roll_goals_for": last["goals_for"].mean(),
        "roll_goals_against": last["goals_against"].mean(),
        "roll_points": float(last["points"].sum()),
        "roll_win_rate": last["win"].mean(),
        "venue_ppg": last_venue["points"].mean(),
        "venue_xgd": (last_venue["xg_for"] - last_venue["xg_against"]).mean(),
    }


def build_fixture_features(
    matches: pd.DataFrame,
    home_team: str,
    away_team: str,
    match_date,
    window: int = WINDOW,
    venue_window: int = VENUE_WINDOW,
    venue_min: int = VENUE_MIN,
) -> pd.DataFrame:
    """One-row table of features for a fixture, using only earlier matches.

    Raises ``UnknownTeamError`` or ``InsufficientHistoryError`` instead of
    silently returning a weak prediction.
    """
    if home_team == away_team:
        raise ValueError("Home and away team must be different.")
    kwargs = dict(window=window, venue_window=venue_window, venue_min=venue_min)
    home = team_features_before(matches, home_team, match_date, "home", **kwargs)
    away = team_features_before(matches, away_team, match_date, "away", **kwargs)

    row = {"date": pd.Timestamp(match_date), "home_team": home_team, "away_team": away_team}
    row.update({f"home_{k}": v for k, v in home.items()})
    row.update({f"away_{k}": v for k, v in away.items()})
    return add_difference_features(pd.DataFrame([row]))
