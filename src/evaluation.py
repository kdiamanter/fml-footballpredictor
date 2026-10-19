"""Evaluation: probability checks, scoring rules, calibration and the backtest.

Scoring rules (all: lower is better, computed per match then averaged):

- **log loss**: -log(probability given to what happened). Punishes confident
  mistakes very hard. Always-1/3 scores log(3) = 1.099.
- **Brier score**: squared distance between the probabilities and the
  outcome written as (1, 0, 0), (0, 1, 0) or (0, 0, 1). Range 0 to 2.
  Always-1/3 scores 0.667.
- **RPS** (ranked probability score): like Brier but on *cumulative*
  probabilities, so it respects the order home win > draw > away win.
  Predicting a draw when the home team won is "less wrong" than predicting an
  away win. Range 0 to 1. The standard score for football forecasts.

The walk-forward backtest refits every model before each matchday, using
only matches played strictly before that day, then predicts that day's
matches. This mimics using the model live through past seasons.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

# Column order for every probability array in the project.
# RPS needs this exact order: home win, draw, away win.
CLASSES = ["H", "D", "A"]


def check_probabilities(probs, atol: float = 1e-9) -> None:
    """Raise ValueError unless every row is three probabilities summing to 1."""
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != len(CLASSES):
        raise ValueError(f"Expected shape (n, 3), got {probs.shape}")
    if np.isnan(probs).any():
        raise ValueError("Probabilities contain NaN")
    if (probs < 0).any() or (probs > 1).any():
        raise ValueError("Probabilities must be between 0 and 1")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=atol):
        raise ValueError("Each row of probabilities must sum to 1")


def _prepare(results, probs) -> tuple[np.ndarray, np.ndarray]:
    """Validate inputs; return probs and a one-hot (n, 3) matrix of outcomes."""
    probs = np.asarray(probs, dtype=float)
    check_probabilities(probs)
    results = np.asarray(results)
    if len(results) != len(probs):
        raise ValueError("results and probs must have the same length")
    unexpected = set(results) - set(CLASSES)
    if unexpected:
        raise ValueError(f"Unexpected result labels: {unexpected}")
    onehot = (results[:, None] == np.array(CLASSES)[None, :]).astype(float)
    return probs, onehot


# ---------------------------------------------------------------------------
# Per-match scores (used for averages and for paired model comparisons)
# ---------------------------------------------------------------------------
def log_loss_per_match(results, probs, eps: float = 1e-15) -> np.ndarray:
    probs, onehot = _prepare(results, probs)
    picked = np.clip((probs * onehot).sum(axis=1), eps, 1.0)
    return -np.log(picked)


def brier_per_match(results, probs) -> np.ndarray:
    probs, onehot = _prepare(results, probs)
    return ((probs - onehot) ** 2).sum(axis=1)


def rps_per_match(results, probs) -> np.ndarray:
    probs, onehot = _prepare(results, probs)
    cum_p = np.cumsum(probs, axis=1)[:, :-1]     # P(H), P(H or D)
    cum_o = np.cumsum(onehot, axis=1)[:, :-1]
    return ((cum_p - cum_o) ** 2).sum(axis=1) / (len(CLASSES) - 1)


def log_loss(results, probs, eps: float = 1e-15) -> float:
    """Average negative log probability given to the outcome that happened.

    Lower is better. Uses the project's column order (H, D, A). Do NOT use
    ``sklearn.metrics.log_loss`` with string labels: it silently assumes the
    columns are in alphabetical order (A, D, H) and gives wrong numbers.
    """
    return float(log_loss_per_match(results, probs, eps).mean())


def brier_score(results, probs) -> float:
    """Average multi-class Brier score (0 = perfect, 2 = worst)."""
    return float(brier_per_match(results, probs).mean())


def rps(results, probs) -> float:
    """Average ranked probability score (0 = perfect, 1 = worst)."""
    return float(rps_per_match(results, probs).mean())


def accuracy(results, probs) -> float:
    """Share of matches where the most likely outcome happened (for intuition only)."""
    probs, onehot = _prepare(results, probs)
    return float(onehot[np.arange(len(probs)), probs.argmax(axis=1)].mean())


METRICS: dict[str, Callable] = {
    "log loss": log_loss,
    "Brier": brier_score,
    "RPS": rps,
    "accuracy": accuracy,
}
PER_MATCH: dict[str, Callable] = {
    "log loss": log_loss_per_match,
    "Brier": brier_per_match,
    "RPS": rps_per_match,
}


def score_table(results, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    """One row per model, one column per metric."""
    return pd.DataFrame({
        name: {metric: fn(results, probs) for metric, fn in METRICS.items()}
        for name, probs in predictions.items()
    }).T


def paired_difference(results, probs_a, probs_b, metric: str = "RPS") -> dict[str, float]:
    """Mean per-match score of model A minus model B, with a 95% interval.

    Negative = A is better. Both models are scored on the *same* matches, so
    the comparison is paired: match difficulty cancels out. The interval uses
    the normal approximation (mean ± 1.96 standard errors), treating matches
    as independent. If the interval contains 0, the data cannot tell the two
    models apart.
    """
    diff = PER_MATCH[metric](results, probs_a) - PER_MATCH[metric](results, probs_b)
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    mean = float(diff.mean())
    return {"mean diff": mean, "ci low": mean - 1.96 * se, "ci high": mean + 1.96 * se,
            "n": len(diff)}


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibration_table(results, probs, outcome: str, n_bins: int = 10) -> pd.DataFrame:
    """For one outcome: group matches by predicted probability and compare.

    A well-calibrated model that says "30% draw" should see a draw in about
    30% of those matches. Bins are equal-width (0-0.1, 0.1-0.2, ...); empty
    bins are dropped.
    """
    probs, onehot = _prepare(results, probs)
    j = CLASSES.index(outcome)
    p, hit = probs[:, j], onehot[:, j]
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    table = pd.DataFrame({"bin": bins, "predicted": p, "observed": hit}).groupby("bin").agg(
        predicted=("predicted", "mean"),
        observed=("observed", "mean"),
        matches=("observed", "size"),
    )
    table.index = [f"{b / n_bins:.1f}-{(b + 1) / n_bins:.1f}" for b in table.index]
    return table


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------
def walk_forward_backtest(
    matches: pd.DataFrame,
    fitters: dict[str, Callable],
    start,
    end=None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Predict every complete match in [start, end) as if live.

    ``matches`` is the full feature table (one row per played match).
    ``fitters`` maps a model name to ``fit(matches, date) -> model``, for
    example ``fit_logistic_before``; each fitter must only use matches before
    ``date``. For every matchday D the models are refitted on matches before
    D, then predict the matches on D.

    Returns one row per (match, model) with columns p_H, p_D, p_A and
    n_train (number of matches the model was trained on).
    """
    # Imported here to avoid a circular import (models imports CLASSES).
    from src.features import FEATURE_COLUMNS

    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"])
    day = matches["date"].dt.normalize()
    in_window = day >= pd.Timestamp(start).normalize()
    if end is not None:
        in_window &= day < pd.Timestamp(end).normalize()
    targets = matches[in_window & matches[FEATURE_COLUMNS].notna().all(axis=1)
                      & matches["result"].isin(CLASSES)]
    if targets.empty:
        raise ValueError("No complete matches in the backtest window.")

    keep = [c for c in ["date", "season", "home_team", "away_team",
                        "home_goals", "away_goals", "result"] if c in targets.columns]
    out = []
    days = sorted(targets["date"].dt.normalize().unique())
    for i, d in enumerate(days):
        todays = targets[targets["date"].dt.normalize() == d]
        for name, fit in fitters.items():
            model = fit(matches, d)
            probs = model.predict_proba(todays)
            check_probabilities(probs)
            block = todays[keep].copy()
            block["model"] = name
            block[["p_H", "p_D", "p_A"]] = probs
            block["n_train"] = model.n_matches_
            out.append(block)
        if verbose and (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(days)} matchdays done")
    if verbose:
        print(f"Backtest done: {len(targets)} matches on {len(days)} matchdays, {len(fitters)} models.")
    return pd.concat(out).reset_index(drop=True)


def probs_of(backtest: pd.DataFrame, model: str) -> tuple[pd.Series, np.ndarray]:
    """Results and (n, 3) probabilities of one model from a backtest table."""
    rows = backtest[backtest["model"] == model]
    return rows["result"], rows[["p_H", "p_D", "p_A"]].to_numpy()
