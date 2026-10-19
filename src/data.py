"""Data collection and cleaning.

This module does four things:

1. Downloads English Premier League match data from Understat through the
   ``soccerdata`` library and caches it in ``data/raw/``.
2. Cleans the raw data (types, dates, duplicates, team names, missing values,
   chronological order).
3. Derives the match outcome label (H = home win, D = draw, A = away win).
4. Extracts the upcoming (not yet played) fixtures for predictions.

Column names come from soccerdata's ``Understat.read_schedule()``. The
notebook prints the raw columns so you can check them yourself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to the project root, so the project can live anywhere)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_MATCHES_PATH = PROCESSED_DIR / "processed_matches.csv"

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
LEAGUE = "ENG-Premier League"

# Ten completed seasons (2016/17 to 2025/26) plus the current 2026/27 season.
# soccerdata's season code "1617" means 2016/17.
SEASONS: list[str] = [
    "1617", "1718", "1819", "1920", "2021",
    "2122", "2223", "2324", "2425", "2526",
    "2627",
]

# The columns every later step relies on.
REQUIRED_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "home_xg",
    "away_xg",
]

# Extra columns we keep if soccerdata provides them.
OPTIONAL_COLUMNS = ["season", "game_id", "is_result"]

# Understat already uses one consistent name per club. This map catches
# common alternative spellings in case data from another source is mixed in.
# If the notebook's team list shows two names for one club, add it here.
TEAM_NAME_MAP: dict[str, str] = {
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Man Utd": "Manchester United",
    "Manchester Utd": "Manchester United",
    "Newcastle": "Newcastle United",
    "Spurs": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "Wolves": "Wolverhampton Wanderers",
    "West Ham United": "West Ham",
    "Brighton & Hove Albion": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "Leicester City": "Leicester",
    "Leeds United": "Leeds",
    "Nott'm Forest": "Nottingham Forest",
    "Sheffield Utd": "Sheffield United",
}


# ---------------------------------------------------------------------------
# 1. Download with local caching
# ---------------------------------------------------------------------------
def raw_cache_path(league: str = LEAGUE, seasons: list[str] = SEASONS) -> Path:
    """Path of the cached raw CSV for a league and list of seasons."""
    league_slug = league.replace(" ", "-")
    return RAW_DIR / f"understat_{league_slug}_{seasons[0]}-{seasons[-1]}.csv"


def download_understat_matches(
    league: str = LEAGUE,
    seasons: list[str] = SEASONS,
    force_download: bool = False,
) -> pd.DataFrame:
    """Return Understat match data, downloading it only when needed.

    The first call downloads from Understat and saves a CSV in ``data/raw/``.
    Later calls read that CSV, so no internet is needed. Use
    ``force_download=True`` to refresh (for example to get the latest
    results of the current season).
    """
    cache_file = raw_cache_path(league, seasons)
    if cache_file.exists() and not force_download:
        print(f"Loading cached raw data: {cache_file.relative_to(PROJECT_ROOT)}")
        return pd.read_csv(cache_file)

    # Imported here so the rest of the project (and the tests) work without
    # soccerdata installed.
    import soccerdata as sd

    print(f"Downloading {league}, seasons {seasons[0]}-{seasons[-1]} from Understat...")
    reader = sd.Understat(
        leagues=league,
        seasons=seasons,
        data_dir=RAW_DIR / "soccerdata_cache",  # soccerdata's own JSON cache
    )
    schedule = reader.read_schedule().reset_index()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(cache_file, index=False)
    print(f"Saved {len(schedule)} rows to {cache_file.relative_to(PROJECT_ROOT)}")
    return schedule


# ---------------------------------------------------------------------------
# 2. Cleaning
# ---------------------------------------------------------------------------
def standardise_team_name(name: str) -> str:
    """Trim spaces, collapse double spaces and apply ``TEAM_NAME_MAP``."""
    cleaned = " ".join(str(name).split())
    return TEAM_NAME_MAP.get(cleaned, cleaned)


def _to_bool(series: pd.Series) -> pd.Series:
    """Convert True/False values that may have been read from CSV as text."""
    mapping = {"true": True, "false": False, "1": True, "0": False}
    return series.map(lambda v: v if isinstance(v, (bool, np.bool_)) else mapping.get(str(v).strip().lower()))


def clean_matches(raw: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Clean raw match data and return one row per played match.

    Steps:
    - check that the required columns exist
    - keep only matches that have been played (``is_result``), if available
    - parse dates and convert goals/xG to numbers
    - standardise team names
    - drop rows with missing required values
    - drop impossible rows (team playing itself, negative goals or xG)
    - remove duplicate matches
    - sort chronologically and add the outcome label
    """
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"Raw data is missing required columns: {missing_cols}")

    keep = REQUIRED_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in raw.columns]
    df = raw[keep].copy()
    report: dict[str, int] = {"raw rows": len(df)}

    # Only keep matches that have a final result (drops upcoming fixtures).
    if "is_result" in df.columns:
        df["is_result"] = _to_bool(df["is_result"])
        df = df[df["is_result"] == True]  # noqa: E712  (also drops missing)
        report["after removing unplayed fixtures"] = len(df)

    # Types: dates and numbers. Bad values become NaT/NaN and are dropped below.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["home_goals", "away_goals", "home_xg", "away_xg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Team names.
    for col in ["home_team", "away_team"]:
        df[col] = df[col].where(df[col].isna(), df[col].map(standardise_team_name))

    # Missing values: a match without date, teams, goals or xG cannot be used.
    before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    report["dropped (missing values)"] = before - len(df)

    # Impossible rows.
    before = len(df)
    valid = (
        (df["home_team"] != df["away_team"])
        & (df["home_goals"] >= 0) & (df["away_goals"] >= 0)
        & (df["home_xg"] >= 0) & (df["away_xg"] >= 0)
    )
    df = df[valid]
    report["dropped (invalid values)"] = before - len(df)

    # Goals are whole numbers.
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    df["home_xg"] = df["home_xg"].astype(float)
    df["away_xg"] = df["away_xg"].astype(float)

    # Duplicates: the same fixture on the same calendar day counts once.
    before = len(df)
    df["_day"] = df["date"].dt.normalize()
    df = df.drop_duplicates(subset=["_day", "home_team", "away_team"], keep="first")
    df = df.drop(columns="_day")
    report["dropped (duplicates)"] = before - len(df)

    # Chronological order with a fixed tie-break so results are reproducible.
    df = df.sort_values(["date", "home_team", "away_team"], kind="mergesort")
    df = df.reset_index(drop=True)

    df["result"] = match_outcome(df["home_goals"], df["away_goals"])
    if "is_result" in df.columns:
        df = df.drop(columns="is_result")

    report["clean rows"] = len(df)
    if verbose:
        print("Cleaning report")
        for step, n in report.items():
            print(f"  {step:<35} {n:>6}")
    return df


def upcoming_fixtures(raw: pd.DataFrame) -> pd.DataFrame:
    """Fixtures that have not been played yet, cleaned like ``clean_matches``.

    Returns date, home_team and away_team (plus season/game_id if present),
    sorted by date. Needs the ``is_result`` column to tell played from unplayed.
    """
    if "is_result" not in raw.columns:
        raise ValueError("Raw data has no 'is_result' column, so upcoming fixtures cannot be found.")
    cols = ["date", "home_team", "away_team"] + [c for c in ["season", "game_id"] if c in raw.columns]
    df = raw.loc[_to_bool(raw["is_result"]) == False, cols].copy()  # noqa: E712
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["home_team", "away_team"]:
        df[col] = df[col].where(df[col].isna(), df[col].map(standardise_team_name))
    df = df.dropna(subset=["date", "home_team", "away_team"])
    df = df[df["home_team"] != df["away_team"]]
    df["_day"] = df["date"].dt.normalize()
    df = df.drop_duplicates(subset=["_day", "home_team", "away_team"]).drop(columns="_day")
    return df.sort_values(["date", "home_team", "away_team"], kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Outcome label
# ---------------------------------------------------------------------------
def match_outcome(home_goals, away_goals):
    """Return 'H', 'D' or 'A' for each match.

    Works for single numbers and for pandas Series.
    """
    if isinstance(home_goals, pd.Series):
        hg = home_goals.to_numpy()
        ag = pd.Series(away_goals).to_numpy()
        labels = np.select([hg > ag, hg < ag], ["H", "A"], default="D")
        return pd.Series(labels, index=home_goals.index, name="result")
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def save_processed(df: pd.DataFrame, path: Path = PROCESSED_MATCHES_PATH) -> Path:
    """Save the processed dataset as CSV and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_processed(path: Path = PROCESSED_MATCHES_PATH) -> pd.DataFrame:
    """Load the processed dataset with the date column parsed."""
    return pd.read_csv(path, parse_dates=["date"])
