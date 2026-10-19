"""Predicting upcoming matches, and checking those predictions later.

Every time predictions are made they can be saved to
``data/predictions/predictions_made_<date>.csv``. Once those matches have been
played, ``score_saved_predictions`` compares the saved probabilities with the
real results. This is a true out-of-sample test: the predictions were written
down before the matches happened.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.evaluation import CLASSES, check_probabilities
from src.features import InsufficientHistoryError, UnknownTeamError, build_fixture_features
from src.models import MAX_GOALS, PoissonModel

PREDICTIONS_DIR = Path(__file__).resolve().parents[1] / "data" / "predictions"
PROB_COLS = ["p_H", "p_D", "p_A"]


def most_likely_score(home_mu: float, away_mu: float,
                      max_goals: int = MAX_GOALS) -> tuple[int, int, float]:
    """The single most likely scoreline under independent Poisson goals.

    Returns (home goals, away goals, probability of that exact score). The
    probability is usually only 10-15%: even the likeliest score rarely happens.
    """
    goals = np.arange(max_goals + 1)
    grid = np.outer(poisson.pmf(goals, home_mu), poisson.pmf(goals, away_mu))
    home, away = np.unravel_index(grid.argmax(), grid.shape)
    return int(home), int(away), float(grid[home, away] / grid.sum())


def split_upcoming(fixtures: pd.DataFrame, last_played, gap_days: int = 3):
    """Split unplayed fixtures into (next round, postponed).

    - postponed: dated before the last played match's day; their real date is
      unknown, so they are not predicted.
    - next round: from the first remaining fixture, keep adding match days
      until there is a gap of ``gap_days`` or more days without matches.
    """
    fixtures = fixtures.sort_values("date").reset_index(drop=True)
    day = pd.to_datetime(fixtures["date"]).dt.normalize()
    last_day = pd.Timestamp(last_played).normalize()
    postponed = fixtures[day < last_day]
    future = fixtures[day >= last_day]
    if future.empty:
        return future, postponed
    days = pd.Series(sorted(day[day >= last_day].unique()))
    gaps = days.diff().dt.days.fillna(0)
    end = days[(gaps >= gap_days).cumsum() == 0].max()
    return future[day[day >= last_day] <= end].reset_index(drop=True), postponed.reset_index(drop=True)


def predict_fixtures(matches: pd.DataFrame, fixtures: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Predict each fixture with each model.

    ``matches`` are the played matches (cleaned); features for a fixture on
    day D use only matches before D. ``models`` maps a name to a fitted model.

    Returns one row per (fixture, model) with p_H, p_D, p_A. Poisson rows also
    get expected goals and the most likely score. A fixture that cannot be
    predicted (unknown team, too little history) gets one row with a ``note``
    and no probabilities, instead of a silent bad guess.
    """
    rows = []
    for _, fx in fixtures.iterrows():
        base = {"date": fx["date"], "home_team": fx["home_team"], "away_team": fx["away_team"]}
        try:
            feats = build_fixture_features(matches, fx["home_team"], fx["away_team"], fx["date"])
        except (UnknownTeamError, InsufficientHistoryError) as err:
            rows.append({**base, "model": None, "note": str(err)})
            continue
        for name, model in models.items():
            probs = model.predict_proba(feats)
            check_probabilities(probs)
            row = {**base, "model": name, **dict(zip(PROB_COLS, probs[0])), "note": ""}
            if isinstance(model, PoissonModel):
                mu = model.predict_goals(feats).iloc[0]
                row["home_xg_pred"], row["away_xg_pred"] = mu["home_mu"], mu["away_mu"]
                h, a, prob = most_likely_score(mu["home_mu"], mu["away_mu"])
                row["likely_score"], row["likely_score_prob"] = f"{h}-{a}", prob
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Prediction log
# ---------------------------------------------------------------------------
def save_predictions(predictions: pd.DataFrame, made_on, directory: Path = PREDICTIONS_DIR) -> Path:
    """Save predictions with the date they were made. Same day = overwritten."""
    made_on = pd.Timestamp(made_on).normalize()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out = predictions.dropna(subset=["model"]).assign(made_on=made_on)
    path = directory / f"predictions_made_{made_on.date()}.csv"
    out.to_csv(path, index=False)
    return path


def load_saved_predictions(directory: Path = PREDICTIONS_DIR) -> pd.DataFrame:
    """All saved prediction files combined (empty table if none)."""
    files = sorted(Path(directory).glob("predictions_made_*.csv"))
    if not files:
        return pd.DataFrame(columns=["date", "home_team", "away_team", "model", *PROB_COLS, "made_on"])
    return pd.concat([pd.read_csv(f, parse_dates=["date", "made_on"]) for f in files], ignore_index=True)


def score_saved_predictions(saved: pd.DataFrame, played: pd.DataFrame) -> pd.DataFrame:
    """Join saved predictions to real results.

    Only predictions made *before* the match day count. If a fixture was
    predicted several times, the most recent prediction before kickoff is used
    (it knew the most recent form). Returns one row per (match, model) with
    the result attached.
    """
    if saved.empty:
        return saved.assign(result=pd.Series(dtype=str))
    saved = saved.copy()
    saved["_day"] = pd.to_datetime(saved["date"]).dt.normalize()
    saved = saved[pd.to_datetime(saved["made_on"]).dt.normalize() < saved["_day"]]
    saved = (saved.sort_values("made_on")
             .drop_duplicates(["_day", "home_team", "away_team", "model"], keep="last"))
    results = played[["date", "home_team", "away_team", "home_goals", "away_goals", "result"]].copy()
    results["_day"] = pd.to_datetime(results["date"]).dt.normalize()
    merged = saved.merge(results.drop(columns="date"), on=["_day", "home_team", "away_team"], how="inner")
    merged = merged[merged["result"].isin(CLASSES)]
    return merged.drop(columns="_day").sort_values(["date", "home_team", "model"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Any match on any date, without hindsight
# ---------------------------------------------------------------------------
def predict_match_as_of(matches: pd.DataFrame, home_team: str, away_team: str, date) -> pd.DataFrame:
    """Predict one match exactly as it could have been predicted before ``date``.

    Both the models AND the features use only matches played before ``date``,
    so a past match gets a fair prediction (the model has not seen its result
    or anything later). If the match has been played, its real score and
    result are attached.

    Returns one row per model, like ``predict_fixtures``.
    """
    from src.features import build_match_features
    from src.models import fit_logistic_before, fit_poisson_before

    date = pd.Timestamp(date)
    history = matches[pd.to_datetime(matches["date"]).dt.normalize() < date.normalize()]
    features = build_match_features(history)
    models = {
        "Poisson": fit_poisson_before(features, date),
        "Logistic": fit_logistic_before(features, date),
    }
    fixture = pd.DataFrame({"date": [date], "home_team": [home_team], "away_team": [away_team]})
    preds = predict_fixtures(history, fixture, models)
    preds["trained_on"] = models["Poisson"].n_matches_

    day = pd.to_datetime(matches["date"]).dt.normalize()
    played = matches[(day == date.normalize()) & (matches["home_team"] == home_team)
                     & (matches["away_team"] == away_team)]
    if len(played):
        m = played.iloc[0]
        preds["actual_score"] = f"{int(m['home_goals'])}-{int(m['away_goals'])}"
        preds["actual_result"] = m["result"]
    return preds


# ---------------------------------------------------------------------------
# Season replay: one frozen model for a whole past season
# ---------------------------------------------------------------------------
def season_replay(features: pd.DataFrame, season) -> tuple[pd.DataFrame, dict]:
    """Predict every match of a past season with models frozen before it began.

    ``features`` is the full feature table (``build_match_features`` output).
    The models are trained once on all complete matches before the season's
    first match and never updated. Each match's features still come from the
    two teams' previous matches (which can be earlier games of the same
    season), exactly as they were known before kickoff.

    Returns (predictions, info). Predictions have one row per (match, model)
    with p_H, p_D, p_A, like the backtest table. ``info`` holds the training
    period and the matches that could not be predicted.
    """
    from src.features import FEATURE_COLUMNS
    from src.models import fit_baseline_before, fit_logistic_before, fit_poisson_before

    features = features.copy()
    features["date"] = pd.to_datetime(features["date"])
    in_season = features[features["season"].astype(str) == str(season)]
    if in_season.empty:
        raise ValueError(f"No matches for season {season}.")
    start = in_season["date"].min().normalize()
    if (features["date"] < start).sum() == 0:
        raise ValueError(f"Season {season} is the first in the data; nothing to train on.")

    models = {
        "Baseline": fit_baseline_before(features, start),
        "Logistic": fit_logistic_before(features, start),
        "Poisson": fit_poisson_before(features, start),
    }
    complete = in_season.dropna(subset=FEATURE_COLUMNS)
    complete = complete[complete["result"].isin(CLASSES)]
    keep = ["date", "season", "home_team", "away_team", "home_goals", "away_goals", "result"]
    rows = []
    for name, model in models.items():
        block = complete[keep].copy()
        block["model"] = name
        block[PROB_COLS] = model.predict_proba(complete)
        if isinstance(model, PoissonModel):
            mu = model.predict_goals(complete)
            block["home_xg_pred"], block["away_xg_pred"] = mu["home_mu"].values, mu["away_mu"].values
        rows.append(block)

    info = {
        "season_start": start,
        "trained_from": features["date"].min(),
        "trained_to": features.loc[features["date"] < start, "date"].max(),
        "trained_on": models["Logistic"].n_matches_,
        "season_matches": len(in_season),
        "not_predicted": in_season.drop(complete.index)[["date", "home_team", "away_team"]],
    }
    return pd.concat(rows).reset_index(drop=True), info
