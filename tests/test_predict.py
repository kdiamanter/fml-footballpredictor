"""Upcoming fixtures, fixture predictions and the prediction log."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import upcoming_fixtures
from src.features import drop_incomplete, build_match_features
from src.models import LogisticModel, PoissonModel
from src.predict import (
    PROB_COLS, load_saved_predictions, most_likely_score, predict_fixtures, save_predictions,
    score_saved_predictions, split_upcoming,
)
from tests.conftest import make_league


def test_upcoming_fixtures_keeps_only_unplayed():
    raw = pd.DataFrame({
        "date": ["2026-10-17 14:00", "2026-10-10 15:00", "2026-10-18 13:00", "2026-10-18 13:00"],
        "home_team": ["Man City", "Arsenal", "Spurs", "Spurs"],
        "away_team": ["Arsenal", "Chelsea", "Leeds", "Leeds"],
        "is_result": [False, True, "False", False],   # CSV can give text booleans
    })
    up = upcoming_fixtures(raw)
    assert list(up["home_team"]) == ["Manchester City", "Tottenham"]   # names standardised, dup dropped
    assert up["date"].is_monotonic_increasing


def test_most_likely_score():
    assert most_likely_score(2.5, 0.3)[:2] == (2, 0)
    h, a, prob = most_likely_score(0.4, 0.4)
    assert (h, a) == (0, 0)
    assert prob == pytest.approx(np.exp(-0.8), rel=1e-6)   # P(0) * P(0)


def test_split_upcoming():
    fixtures = pd.DataFrame({
        "date": pd.to_datetime(["2026-10-01", "2026-10-17", "2026-10-18", "2026-10-19",
                                "2026-10-23", "2026-10-24"]),
        "home_team": list("ABCDEF"), "away_team": list("GHIJKL"),
    })
    nxt, postponed = split_upcoming(fixtures, last_played="2026-10-12 19:00")
    assert list(postponed["home_team"]) == ["A"]           # dated before the last result
    assert list(nxt["home_team"]) == ["B", "C", "D"]       # stops at the 4-day gap


@pytest.fixture
def setup():
    league = make_league(n_cycles=6)
    train = drop_incomplete(build_match_features(league))
    models = {"Logistic": LogisticModel().fit(train), "Poisson": PoissonModel().fit(train)}
    last = league["date"].max()
    fixtures = pd.DataFrame({
        "date": [last + pd.Timedelta(days=7), last + pd.Timedelta(days=7)],
        "home_team": ["Arsenal", "Chelsea"],
        "away_team": ["Brentford", "Newcastle United"],   # Newcastle unknown in this league
    })
    return league, models, fixtures


def test_predict_fixtures(setup):
    league, models, fixtures = setup
    preds = predict_fixtures(league, fixtures, models)
    ok = preds[preds["model"].notna()]
    assert set(ok["model"]) == {"Logistic", "Poisson"}
    np.testing.assert_allclose(ok[PROB_COLS].sum(axis=1), 1.0)
    assert ok.loc[ok["model"] == "Poisson", "likely_score"].str.match(r"^\d+-\d+$").all()
    bad = preds[preds["model"].isna()]
    assert len(bad) == 1 and "Unknown team" in bad.iloc[0]["note"]


def test_prediction_log_scores_only_predictions_made_before_kickoff(setup, tmp_path):
    league, models, _ = setup
    target = league.iloc[[-1]]                       # a match we know the result of
    day = target["date"].iloc[0].normalize()
    preds = predict_fixtures(league, target[["date", "home_team", "away_team"]], models)

    save_predictions(preds, day - pd.Timedelta(days=3), tmp_path)   # early
    later = preds.assign(p_H=1.0, p_D=0.0, p_A=0.0)
    save_predictions(later, day - pd.Timedelta(days=1), tmp_path)   # latest before kickoff
    save_predictions(preds.assign(p_H=0.0, p_D=0.0, p_A=1.0), day, tmp_path)  # on match day: ignored

    saved = load_saved_predictions(tmp_path)
    assert len(saved) == 6
    scored = score_saved_predictions(saved, league)
    assert len(scored) == 2                                   # one per model
    assert (scored["p_H"] == 1.0).all()                       # the day-before version was used
    assert (scored["result"] == target["result"].iloc[0]).all()


def test_empty_prediction_log(tmp_path):
    saved = load_saved_predictions(tmp_path)
    assert saved.empty
    assert score_saved_predictions(saved, make_league()).empty


def test_predict_match_as_of_ignores_later_results():
    """A past match must get the same prediction whatever happened on or after its date."""
    from src.predict import predict_match_as_of
    from tests.test_leakage import _scramble_results

    league = make_league(n_cycles=6)
    target = league.iloc[len(league) * 2 // 3]
    day = target["date"].normalize()
    before = predict_match_as_of(league, target["home_team"], target["away_team"], target["date"])
    changed = _scramble_results(league, league["date"].dt.normalize() >= day)
    after = predict_match_as_of(changed, target["home_team"], target["away_team"], target["date"])

    np.testing.assert_array_equal(before[PROB_COLS].to_numpy(), after[PROB_COLS].to_numpy())
    assert (before["trained_on"] < len(league)).all()
    assert before["actual_result"].iloc[0] == target["result"]   # real result attached


def test_season_replay_freezes_models_before_the_season():
    from src.predict import season_replay
    from tests.test_leakage import _scramble_results

    league = make_league(n_cycles=6)
    # Label the second half of the league as the "season" to replay.
    cut = league["date"].iloc[len(league) // 2].normalize()
    league["season"] = np.where(league["date"] < cut, "A", "B")
    preds, info = season_replay(build_match_features(league), "B")

    assert info["trained_to"] < cut
    assert info["trained_on"] == len(drop_incomplete(build_match_features(league[league["date"] < cut])))
    assert set(preds["model"]) == {"Baseline", "Logistic", "Poisson"}
    np.testing.assert_allclose(preds[PROB_COLS].sum(axis=1), 1.0)
    assert (preds["date"] >= cut).all()

    # Changing the result of the season's LAST match changes nothing: no prediction
    # (and no model) may depend on it.
    last = league["date"] == league["date"].max()
    changed, _ = season_replay(build_match_features(_scramble_results(league, last)), "B")
    np.testing.assert_array_equal(preds[PROB_COLS].to_numpy(), changed[PROB_COLS].to_numpy())

    # Changing an EARLY season result changes later form features but not the frozen
    # model, so the prediction for that early match itself stays the same.
    first_b = league.index[league["date"] >= cut][0]
    mask = pd.Series(False, index=league.index)
    mask[first_b] = True
    changed, _ = season_replay(build_match_features(_scramble_results(league, mask)), "B")
    same_day = preds["date"].dt.normalize() == league.loc[first_b, "date"].normalize()
    np.testing.assert_array_equal(preds.loc[same_day, PROB_COLS].to_numpy(),
                                  changed.loc[same_day, PROB_COLS].to_numpy())

    with pytest.raises(ValueError):
        season_replay(build_match_features(league), "A")   # nothing before it
    with pytest.raises(ValueError):
        season_replay(build_match_features(league), "Z")   # no such season
