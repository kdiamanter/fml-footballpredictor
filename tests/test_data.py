"""Cleaning, date ordering, missing values and outcome labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import REQUIRED_COLUMNS, clean_matches, match_outcome, standardise_team_name


@pytest.fixture
def messy_raw() -> pd.DataFrame:
    """Raw data shaped like soccerdata's Understat schedule, with problems added."""
    return pd.DataFrame({
        "league": ["ENG-Premier League"] * 7,
        "season": [2122] * 7,
        "game_id": [5, 2, 3, 3, 4, 6, 7],
        "date": [
            "2021-08-21 15:00:00",  # out of order
            "2021-08-13 20:00:00",
            "2021-08-14 12:30:00",
            "2021-08-14 12:30:00",  # exact duplicate
            "not a date",           # unparseable date
            "2021-08-22 16:30:00",  # missing xG
            "2021-08-28 15:00:00",  # not played yet
        ],
        "home_team": ["Man Utd", "Brentford", " Manchester  United ", " Manchester  United ", "Chelsea", "Everton", "Arsenal"],
        "away_team": ["Arsenal", "Arsenal", "Leeds", "Leeds", "Everton", "Chelsea", "Chelsea"],
        "home_goals": ["1", 2, 5, 5, 1, 0, np.nan],
        "away_goals": [1, 0, 1, 1, 0, 2, np.nan],
        "home_xg": [1.2, 1.3, 1.5, 1.5, 0.9, np.nan, np.nan],
        "away_xg": ["0.8", 1.4, 0.7, 0.7, 0.4, 1.1, np.nan],
        "is_result": [True, True, True, True, True, True, False],
    })


def test_clean_sorts_chronologically(messy_raw):
    clean = clean_matches(messy_raw, verbose=False)
    assert clean["date"].is_monotonic_increasing
    assert clean.iloc[0]["home_team"] == "Brentford"


def test_clean_removes_duplicates_missing_and_unplayed(messy_raw):
    clean = clean_matches(messy_raw, verbose=False)
    # Kept: Brentford-Arsenal, Man United-Leeds (once), Man United-Arsenal
    assert len(clean) == 3
    assert not clean[REQUIRED_COLUMNS].isna().any().any()


def test_clean_types(messy_raw):
    clean = clean_matches(messy_raw, verbose=False)
    assert pd.api.types.is_datetime64_any_dtype(clean["date"])
    assert pd.api.types.is_integer_dtype(clean["home_goals"])
    assert pd.api.types.is_integer_dtype(clean["away_goals"])
    assert pd.api.types.is_float_dtype(clean["home_xg"])
    assert pd.api.types.is_float_dtype(clean["away_xg"])


def test_team_names_standardised(messy_raw):
    clean = clean_matches(messy_raw, verbose=False)
    assert set(clean["home_team"]) == {"Brentford", "Manchester United"}
    assert standardise_team_name("  Spurs ") == "Tottenham"


def test_missing_required_column_raises(messy_raw):
    with pytest.raises(ValueError, match="home_xg"):
        clean_matches(messy_raw.drop(columns="home_xg"), verbose=False)


def test_is_result_read_from_csv_as_text(messy_raw):
    """After a CSV round trip True/False may be strings; cleaning must still work."""
    raw = messy_raw.copy()
    raw["is_result"] = raw["is_result"].astype(str)
    assert len(clean_matches(raw, verbose=False)) == 3


def test_invalid_rows_removed():
    raw = pd.DataFrame({
        "date": ["2021-08-13", "2021-08-14"],
        "home_team": ["Arsenal", "Chelsea"],
        "away_team": ["Arsenal", "Everton"],   # team vs itself
        "home_goals": [1, -1],                  # negative goals
        "away_goals": [0, 0],
        "home_xg": [1.0, 1.0],
        "away_xg": [1.0, 1.0],
    })
    assert len(clean_matches(raw, verbose=False)) == 0


@pytest.mark.parametrize("home, away, expected", [(2, 1, "H"), (1, 1, "D"), (0, 3, "A"), (0, 0, "D")])
def test_match_outcome_scalar(home, away, expected):
    assert match_outcome(home, away) == expected


def test_match_outcome_series():
    out = match_outcome(pd.Series([3, 0, 2]), pd.Series([1, 0, 4]))
    assert out.tolist() == ["H", "D", "A"]


def test_clean_adds_result(messy_raw):
    clean = clean_matches(messy_raw, verbose=False)
    expected = match_outcome(clean["home_goals"], clean["away_goals"])
    assert clean["result"].tolist() == expected.tolist()
