"""Rolling feature calculations, checked against values worked out by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import match_outcome
from src.features import (
    FEATURE_COLUMNS,
    InsufficientHistoryError,
    UnknownTeamError,
    build_fixture_features,
    build_match_features,
    drop_incomplete,
    to_team_matches,
)


@pytest.fixture
def arsenal_home_run() -> pd.DataFrame:
    """Arsenal play six home matches. Results: W W D L W, then the 6th match."""
    df = pd.DataFrame({
        "date": pd.date_range("2021-08-01", periods=6, freq="7D"),
        "home_team": ["Arsenal"] * 6,
        "away_team": ["Brentford", "Chelsea", "Everton", "Brentford", "Chelsea", "Everton"],
        "home_goals": [2, 1, 1, 0, 3, 0],
        "away_goals": [0, 0, 1, 2, 1, 0],
        "home_xg": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "away_xg": [0.5] * 6,
    })
    df["result"] = match_outcome(df["home_goals"], df["away_goals"])
    return df


def test_rolling_values_match_hand_calculation(arsenal_home_run):
    feats = build_match_features(arsenal_home_run)
    sixth = feats.iloc[5]
    # Previous five matches: xG 1,2,3,4,5 -> mean 3.0
    assert sixth["home_roll_xg_for"] == pytest.approx(3.0)
    assert sixth["home_roll_xg_against"] == pytest.approx(0.5)
    # Goals 2,1,1,0,3 -> 1.4 ; conceded 0,0,1,2,1 -> 0.8
    assert sixth["home_roll_goals_for"] == pytest.approx(1.4)
    assert sixth["home_roll_goals_against"] == pytest.approx(0.8)
    # W W D L W -> 3+3+1+0+3 = 10 points, 3 wins out of 5
    assert sixth["home_roll_points"] == pytest.approx(10.0)
    assert sixth["home_roll_win_rate"] == pytest.approx(0.6)
    # All five were home matches: 10/5 = 2.0 ppg; xG diff 0.5,1.5,...,4.5 -> 2.5
    assert sixth["home_venue_ppg"] == pytest.approx(2.0)
    assert sixth["home_venue_xgd"] == pytest.approx(2.5)


def test_rolling_needs_full_window(arsenal_home_run):
    feats = build_match_features(arsenal_home_run)
    fifth = feats.iloc[4]
    # Only 4 earlier matches: overall form is not available yet...
    assert np.isnan(fifth["home_roll_xg_for"])
    # ...but venue form needs only 3 home matches: (3+3+1+0)/4 = 1.75
    assert fifth["home_venue_ppg"] == pytest.approx(1.75)
    # First match has no history at all.
    assert feats.iloc[0][[c for c in FEATURE_COLUMNS if c.startswith("home_")]].isna().all()


def test_venue_form_uses_only_home_or_away_matches():
    """Home venue form must ignore away matches and vice versa."""
    dates = pd.date_range("2021-08-01", periods=8, freq="7D")
    df = pd.DataFrame({
        "date": dates,
        # Arsenal alternate: home, away, home, away, ...
        "home_team": ["Arsenal", "Chelsea"] * 4,
        "away_team": ["Chelsea", "Arsenal"] * 4,
        # Arsenal win every home match 1-0 and lose every away match 0-1
        "home_goals": [1, 1] * 4,
        "away_goals": [0, 0] * 4,
        "home_xg": [1.0] * 8,
        "away_xg": [0.0] * 8,
    })
    df["result"] = match_outcome(df["home_goals"], df["away_goals"])
    feats = build_match_features(df)
    last_home = feats[feats["home_team"] == "Arsenal"].iloc[-1]
    last_away = feats[feats["away_team"] == "Arsenal"].iloc[-1]
    assert last_home["home_venue_ppg"] == pytest.approx(3.0)   # home: all wins
    assert last_away["away_venue_ppg"] == pytest.approx(0.0)   # away: all losses


def test_difference_features(league):
    feats = drop_incomplete(build_match_features(league))
    assert len(feats) > 0
    np.testing.assert_allclose(
        feats["xg_for_diff"], feats["home_roll_xg_for"] - feats["away_roll_xg_for"]
    )
    assert not feats[FEATURE_COLUMNS].isna().any().any()


def test_team_rows_are_two_per_match(league):
    rows = to_team_matches(league)
    assert len(rows) == 2 * len(league)
    assert set(rows["points"]) <= {0, 1, 3}


def test_team_playing_twice_on_same_day_is_rejected():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2021-08-01 12:30", "2021-08-01 17:30"]),
        "home_team": ["Arsenal", "Chelsea"],
        "away_team": ["Brentford", "Arsenal"],
        "home_goals": [1, 1], "away_goals": [0, 0],
        "home_xg": [1.0, 1.0], "away_xg": [0.5, 0.5],
    })
    with pytest.raises(ValueError, match="more than once on the same day"):
        to_team_matches(df)


def test_unknown_team_raises_clear_error(league):
    with pytest.raises(UnknownTeamError, match="Unknown team 'Arsenall'.*Arsenal"):
        build_fixture_features(league, "Arsenall", "Chelsea", "2021-06-01")


def test_insufficient_history_raises(league):
    first_date = league["date"].min()
    with pytest.raises(InsufficientHistoryError):
        build_fixture_features(league, "Arsenal", "Chelsea", first_date)


def test_same_team_twice_raises(league):
    with pytest.raises(ValueError):
        build_fixture_features(league, "Arsenal", "Arsenal", "2021-06-01")


def test_fixture_after_last_match_uses_all_history(league):
    """An upcoming fixture after the data ends gets features (no error)."""
    upcoming = league["date"].max() + pd.Timedelta(days=7)
    feats = build_fixture_features(league, "Arsenal", "Chelsea", upcoming)
    assert not feats[FEATURE_COLUMNS].isna().any().any()
