"""No-future-leakage tests.

These are the most important tests in the project. Never remove or weaken
them. If one fails, a feature or model is using information that would not
have been available before kickoff.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import match_outcome
from src.features import FEATURE_COLUMNS, build_fixture_features, build_match_features
from src.models import fit_baseline_before


def _scramble_results(df: pd.DataFrame, mask: pd.Series, seed: int = 7) -> pd.DataFrame:
    """Replace goals, xG and result of the selected matches with new values."""
    rng = np.random.default_rng(seed)
    changed = df.copy()
    n = int(mask.sum())
    changed.loc[mask, "home_goals"] = rng.integers(0, 7, n)
    changed.loc[mask, "away_goals"] = rng.integers(0, 7, n)
    changed.loc[mask, "home_xg"] = rng.uniform(0, 5, n).round(2)
    changed.loc[mask, "away_xg"] = rng.uniform(0, 5, n).round(2)
    changed["result"] = match_outcome(changed["home_goals"], changed["away_goals"])
    return changed


def _predict_historical_match(matches: pd.DataFrame, match_row: pd.Series):
    """Features + baseline probabilities for one past match, as a model would see them."""
    features = build_fixture_features(
        matches, match_row["home_team"], match_row["away_team"], match_row["date"]
    )
    probs = fit_baseline_before(matches, match_row["date"]).predict_proba(1)
    return features, probs


def test_prediction_unchanged_when_future_results_change(league):
    """1) predict a historical match, 2) change future results,
    3) predict again, 4) the prediction must be exactly the same."""
    target = league.iloc[24]  # a match with enough history for both teams
    future = pd.to_datetime(league["date"]).dt.normalize() > target["date"].normalize()
    assert future.sum() > 0, "test needs future matches to change"

    features_before, probs_before = _predict_historical_match(league, target)
    changed = _scramble_results(league, future)
    assert not changed.equals(league), "future results were not actually changed"
    features_after, probs_after = _predict_historical_match(changed, target)

    pd.testing.assert_frame_equal(features_before, features_after, check_exact=True)
    np.testing.assert_array_equal(probs_before, probs_after)


def test_batch_features_unchanged_when_future_results_change(league):
    """Features for every match up to a cutoff must not move when later results change."""
    cutoff = pd.to_datetime(league["date"]).iloc[24].normalize()
    future = pd.to_datetime(league["date"]) >= cutoff
    past = ~future

    before = build_match_features(league)
    after = build_match_features(_scramble_results(league, future))

    keep = pd.to_datetime(before["date"]) < cutoff
    assert keep.sum() == past.sum()
    pd.testing.assert_frame_equal(
        before.loc[keep, FEATURE_COLUMNS], after.loc[keep, FEATURE_COLUMNS], check_exact=True
    )
    # Sanity check that the test has teeth: later features DID change.
    later = ~keep
    assert not before.loc[later, FEATURE_COLUMNS].equals(after.loc[later, FEATURE_COLUMNS])


def test_match_does_not_see_its_own_result(league):
    """Changing a match's own goals and xG must not change its own features."""
    i = 24
    before = build_match_features(league)
    only_this = pd.Series(False, index=league.index)
    only_this.iloc[i] = True
    after = build_match_features(_scramble_results(league, only_this))

    pd.testing.assert_series_equal(
        before.loc[i, FEATURE_COLUMNS], after.loc[i, FEATURE_COLUMNS], check_exact=True
    )


def test_same_day_results_are_not_used():
    """A result from earlier on the same calendar day must not leak into the baseline."""
    df = pd.DataFrame({
        "date": pd.to_datetime(["2021-01-01 15:00", "2021-01-02 12:30", "2021-01-02 17:30"]),
        "result": ["H", "A", "D"],
    })
    model = fit_baseline_before(df, "2021-01-02 17:30")
    assert model.n_matches_ == 1  # only the 1 January match


def test_batch_and_single_fixture_features_agree(league):
    """The fast all-matches path and the one-fixture path must give the same features."""
    batch = build_match_features(league)
    for i in [20, 25, 30, 35]:
        row = batch.iloc[i]
        single = build_fixture_features(league, row["home_team"], row["away_team"], row["date"])
        np.testing.assert_allclose(
            single[FEATURE_COLUMNS].iloc[0].to_numpy(dtype=float),
            row[FEATURE_COLUMNS].to_numpy(dtype=float),
            rtol=1e-12,
        )


def test_input_order_does_not_matter(league):
    """Shuffled input rows must give the same features (sorting happens inside)."""
    shuffled = league.sample(frac=1.0, random_state=0)
    pd.testing.assert_frame_equal(
        build_match_features(league)[FEATURE_COLUMNS],
        build_match_features(shuffled)[FEATURE_COLUMNS],
    )


def test_trained_models_unchanged_when_future_results_change():
    """Stage 2 models fitted before a date must not move when later results change."""
    from tests.conftest import make_league
    from src.features import drop_incomplete
    from src.models import fit_logistic_before, fit_poisson_before

    league = make_league(n_cycles=6)
    cutoff = pd.to_datetime(league["date"]).iloc[len(league) // 2].normalize()
    future = pd.to_datetime(league["date"]) >= cutoff

    before = build_match_features(league)
    after = build_match_features(_scramble_results(league, future))
    target = drop_incomplete(before)
    target = target[target["date"] >= cutoff].head(5)
    assert len(target) > 0
    target_after = after.loc[target.index]  # same sort order, so the same matches

    for fit_before in [fit_logistic_before, fit_poisson_before]:
        m_before = fit_before(before, cutoff)
        m_after = fit_before(after, cutoff)
        # Same training data -> identical predictions on the same feature rows.
        np.testing.assert_array_equal(m_before.predict_proba(target), m_after.predict_proba(target))
    # Teeth: the scrambled future really does change those later rows' features.
    assert not np.allclose(target[FEATURE_COLUMNS].to_numpy(float),
                           target_after[FEATURE_COLUMNS].to_numpy(float))


def test_backtest_unchanged_when_future_results_change():
    """Backtest predictions up to a cutoff must not move when later results change."""
    from tests.conftest import make_league
    from src.evaluation import walk_forward_backtest
    from src.models import fit_baseline_before, fit_logistic_before, fit_poisson_before

    league = make_league(n_cycles=6)
    dates = pd.to_datetime(league["date"])
    start, cutoff = dates.iloc[len(league) // 3], dates.iloc[2 * len(league) // 3].normalize()
    fitters = {"baseline": fit_baseline_before, "logistic": fit_logistic_before,
               "poisson": fit_poisson_before}

    before = walk_forward_backtest(build_match_features(league), fitters, start, verbose=False)
    scrambled = _scramble_results(league, dates.dt.normalize() >= cutoff)
    after = walk_forward_backtest(build_match_features(scrambled), fitters, start, verbose=False)

    cols = ["p_H", "p_D", "p_A"]
    early_b = before[before["date"] < cutoff]
    early_a = after[after["date"] < cutoff]
    assert len(early_b) > 0
    np.testing.assert_array_equal(early_b[cols].to_numpy(), early_a[cols].to_numpy())
    # Teeth: predictions after the cutoff did change.
    assert not np.allclose(before.loc[before["date"] >= cutoff, cols].to_numpy(),
                           after.loc[after["date"] >= cutoff, cols].to_numpy())
