"""Baseline model and probability checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import CLASSES, check_probabilities
from src.models import HistoricalBaseline, fit_baseline_before


def test_baseline_frequencies():
    model = HistoricalBaseline().fit(pd.Series(["H", "H", "D", "A"]))
    np.testing.assert_allclose(model.probabilities_, [0.5, 0.25, 0.25])
    assert CLASSES == ["H", "D", "A"]


def test_baseline_probabilities_sum_to_one(league):
    probs = HistoricalBaseline().fit(league["result"]).predict_proba(10)
    assert probs.shape == (10, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    check_probabilities(probs)


def test_baseline_handles_missing_class():
    """No draws in the data -> draw probability 0, still sums to 1."""
    probs = HistoricalBaseline().fit(pd.Series(["H", "A", "H"])).predict_proba(1)
    np.testing.assert_allclose(probs, [[2 / 3, 0.0, 1 / 3]])


def test_baseline_rejects_bad_input():
    with pytest.raises(ValueError):
        HistoricalBaseline().fit(pd.Series([], dtype=str))
    with pytest.raises(ValueError):
        HistoricalBaseline().fit(pd.Series(["H", "X"]))
    with pytest.raises(RuntimeError):
        HistoricalBaseline().predict_proba(1)


def test_fit_before_uses_only_past(league):
    cutoff = league["date"].iloc[10]
    model = fit_baseline_before(league, cutoff)
    assert model.n_matches_ == (league["date"].dt.normalize() < cutoff.normalize()).sum()


@pytest.mark.parametrize("bad", [
    [[0.5, 0.5, 0.5]],          # sums to 1.5
    [[1.2, -0.1, -0.1]],        # outside [0, 1]
    [[0.5, 0.5]],               # wrong shape
    [[np.nan, 0.5, 0.5]],       # NaN
])
def test_check_probabilities_rejects(bad):
    with pytest.raises(ValueError):
        check_probabilities(bad)


# ---------------------------------------------------------------------------
# Stage 2: logistic and Poisson models
# ---------------------------------------------------------------------------
from src.features import build_match_features, drop_incomplete  # noqa: E402
from src.models import (  # noqa: E402
    LOGISTIC_FEATURES, POISSON_FEATURES, LogisticModel, PoissonModel, fit_logistic_before,
    fit_poisson_before, load_model, outcome_probs_from_goals, save_model,
)


@pytest.fixture
def train_rows():
    from tests.conftest import make_league
    return drop_incomplete(build_match_features(make_league(n_cycles=6)))


def _signal_data(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Synthetic matches where a bigger xG difference means more home wins."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, len(POISSON_FEATURES))), columns=POISSON_FEATURES)
    strength = df["xg_for_diff"].to_numpy()
    df["home_goals"] = rng.poisson(np.exp(0.3 + 0.4 * strength))
    df["away_goals"] = rng.poisson(np.exp(0.1 - 0.4 * strength))
    df["result"] = np.select([df["home_goals"] > df["away_goals"],
                              df["home_goals"] == df["away_goals"]], ["H", "D"], "A")
    return df


@pytest.mark.parametrize("model_cls", [LogisticModel, PoissonModel])
def test_model_probabilities_valid(model_cls, train_rows):
    probs = model_cls().fit(train_rows).predict_proba(train_rows)
    assert probs.shape == (len(train_rows), 3)
    check_probabilities(probs)


def test_logistic_columns_in_h_d_a_order():
    """sklearn sorts classes A, D, H. Our output must be H, D, A."""
    df = _signal_data()
    model = LogisticModel().fit(df)
    strong_home = df.nlargest(50, "xg_for_diff")
    strong_away = df.nsmallest(50, "xg_for_diff")
    p_home = model.predict_proba(strong_home).mean(axis=0)
    p_away = model.predict_proba(strong_away).mean(axis=0)
    assert p_home[0] > p_home[2]   # column 0 = home win
    assert p_away[2] > p_away[0]   # column 2 = away win


def test_poisson_learns_direction():
    df = _signal_data()
    model = PoissonModel(alpha=0.01).fit(df)
    goals = model.predict_goals(df.nlargest(50, "xg_for_diff")).mean()
    assert goals["home_mu"] > goals["away_mu"]
    probs = model.predict_proba(df.nlargest(50, "xg_for_diff")).mean(axis=0)
    assert probs[0] > probs[2]


def test_outcome_probs_from_goals_hand_checked():
    # Equal teams -> P(home win) == P(away win).
    p = outcome_probs_from_goals([1.3], [1.3])[0]
    assert p[0] == pytest.approx(p[2])
    # 0 vs 0 is the only way to draw when max_goals=0 -> certain draw.
    np.testing.assert_allclose(outcome_probs_from_goals([1.0], [1.0], max_goals=0), [[0, 1, 0]])
    # Draw probability matches the direct sum of P(i, i).
    from scipy.stats import poisson
    draw = sum(poisson.pmf(k, 1.5) * poisson.pmf(k, 1.1) for k in range(11))
    full = sum(poisson.pmf(i, 1.5) * poisson.pmf(j, 1.1) for i in range(11) for j in range(11))
    assert outcome_probs_from_goals([1.5], [1.1])[0][1] == pytest.approx(draw / full)
    with pytest.raises(ValueError):
        outcome_probs_from_goals([0.0], [1.0])


@pytest.mark.parametrize("model_cls", [LogisticModel, PoissonModel])
def test_models_reject_bad_input(model_cls, train_rows):
    with pytest.raises(RuntimeError):
        model_cls().predict_proba(train_rows)
    with_nan = train_rows.copy()
    with_nan.loc[0, LOGISTIC_FEATURES[0]] = np.nan
    with pytest.raises(ValueError):
        model_cls().fit(with_nan)
    with pytest.raises(ValueError):
        model_cls().fit(train_rows.drop(columns=LOGISTIC_FEATURES[0]))


@pytest.mark.parametrize("fit_before", [fit_logistic_before, fit_poisson_before])
def test_fit_model_before_uses_only_past(fit_before, train_rows):
    cutoff = train_rows["date"].iloc[len(train_rows) // 2]
    model = fit_before(train_rows, cutoff)
    assert model.n_matches_ == (train_rows["date"].dt.normalize() < cutoff.normalize()).sum()
    with pytest.raises(ValueError):
        fit_before(train_rows, train_rows["date"].min())


@pytest.mark.parametrize("model_cls", [LogisticModel, PoissonModel])
def test_save_and_load_round_trip(model_cls, train_rows, tmp_path):
    model = model_cls().fit(train_rows)
    save_model(model, "m", tmp_path)
    loaded = load_model("m", tmp_path)
    np.testing.assert_array_equal(model.predict_proba(train_rows), loaded.predict_proba(train_rows))
    with pytest.raises(FileNotFoundError):
        load_model("missing", tmp_path)


def test_log_loss_hand_checked_and_column_order():
    from src.evaluation import log_loss
    probs = np.array([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])
    # Match 1 home win (0.5), match 2 away win (0.7).
    assert log_loss(["H", "A"], probs) == pytest.approx(-(np.log(0.5) + np.log(0.7)) / 2)
    # Swapping H and A must change the answer (guards against alphabetical ordering).
    assert log_loss(["A", "H"], probs) != pytest.approx(log_loss(["H", "A"], probs))
    with pytest.raises(ValueError):
        log_loss(["H"], probs)
