"""Scoring rules, calibration and the walk-forward backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    accuracy, brier_score, calibration_table, log_loss, paired_difference, probs_of, rps,
    rps_per_match, score_table, walk_forward_backtest,
)
from src.features import build_match_features
from src.models import fit_baseline_before, fit_logistic_before, fit_poisson_before
from tests.conftest import make_league

PERFECT_H = [[1.0, 0.0, 0.0]]
UNIFORM = [[1 / 3, 1 / 3, 1 / 3]]


def test_brier_hand_checked():
    assert brier_score(["H"], PERFECT_H) == 0.0
    assert brier_score(["A"], PERFECT_H) == pytest.approx(2.0)          # (1-0)^2 + (0-1)^2
    assert brier_score(["D"], UNIFORM) == pytest.approx(2 / 3)
    assert brier_score(["H"], [[0.5, 0.3, 0.2]]) == pytest.approx(0.25 + 0.09 + 0.04)


def test_rps_hand_checked():
    assert rps(["H"], PERFECT_H) == 0.0
    assert rps(["A"], PERFECT_H) == pytest.approx(1.0)                  # worst possible
    # Cumulative probs (0.5, 0.8) vs outcome H (1, 1): ((0.5)^2 + (0.2)^2) / 2
    assert rps(["H"], [[0.5, 0.3, 0.2]]) == pytest.approx((0.25 + 0.04) / 2)


def test_rps_respects_outcome_order():
    """Home win happened: betting on a draw is less wrong than betting on an away win."""
    draw_bet = rps_per_match(["H"], [[0.0, 1.0, 0.0]])[0]
    away_bet = rps_per_match(["H"], [[0.0, 0.0, 1.0]])[0]
    assert draw_bet < away_bet
    # Brier cannot see the difference; that is why RPS is used for football.
    assert brier_score(["H"], [[0.0, 1.0, 0.0]]) == brier_score(["H"], [[0.0, 0.0, 1.0]])


def test_accuracy_and_score_table():
    probs = np.array([[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]])
    assert accuracy(["H", "H"], probs) == 0.5
    table = score_table(["H", "H"], {"m": probs})
    assert list(table.columns) == ["log loss", "Brier", "RPS", "accuracy"]
    assert table.loc["m", "log loss"] == pytest.approx(log_loss(["H", "H"], probs))


def test_paired_difference_sign():
    results = ["H"] * 50 + ["A"] * 50
    good = np.array([[0.7, 0.2, 0.1]] * 50 + [[0.1, 0.2, 0.7]] * 50)
    bad = np.array([[1 / 3] * 3] * 100)
    d = paired_difference(results, good, bad, "RPS")
    assert d["mean diff"] < 0 and d["ci high"] < 0 and d["n"] == 100


def test_calibration_table():
    probs = np.array([[0.05, 0.25, 0.70], [0.15, 0.25, 0.60], [0.95, 0.03, 0.02]])
    table = calibration_table(["A", "H", "H"], probs, "H", n_bins=10)
    assert table["matches"].sum() == 3
    assert list(table.index) == ["0.0-0.1", "0.1-0.2", "0.9-1.0"]
    assert table.loc["0.1-0.2", "observed"] == 1.0
    assert table.loc["0.9-1.0", "predicted"] == pytest.approx(0.95)


@pytest.mark.parametrize("fn", [log_loss, brier_score, rps])
def test_metrics_reject_bad_input(fn):
    with pytest.raises(ValueError):
        fn(["H", "D"], PERFECT_H)
    with pytest.raises(ValueError):
        fn(["X"], PERFECT_H)


FITTERS = {"baseline": fit_baseline_before, "logistic": fit_logistic_before,
           "poisson": fit_poisson_before}


@pytest.fixture
def features():
    return build_match_features(make_league(n_cycles=6))


def test_backtest_output(features):
    start = features["date"].iloc[len(features) // 2]
    bt = walk_forward_backtest(features, FITTERS, start, verbose=False)
    n_targets = (features["date"].dt.normalize() >= start.normalize()).sum()
    assert len(bt) == 3 * n_targets
    assert set(bt["model"]) == set(FITTERS)
    results, probs = probs_of(bt, "logistic")
    assert len(results) == n_targets
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)


def test_backtest_models_only_train_on_earlier_days(features):
    """Every prediction's model was trained on exactly the complete matches before that day."""
    start = features["date"].iloc[len(features) // 2]
    bt = walk_forward_backtest(features, {"logistic": fit_logistic_before}, start, verbose=False)
    complete = features.dropna()
    for _, row in bt.iterrows():
        earlier = (complete["date"].dt.normalize() < row["date"].normalize()).sum()
        assert row["n_train"] == earlier
