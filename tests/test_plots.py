import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from src.plots import plot_match_prediction, plot_next_round, scoreline_grid


def _matches():
    rows = []
    for i in range(12):
        home, away = ("Alpha", "Beta") if i % 2 == 0 else ("Beta", "Alpha")
        rows.append(dict(date=pd.Timestamp("2026-01-01") + pd.Timedelta(days=7 * i),
                         home_team=home, away_team=away, home_xg=1.4, away_xg=1.0,
                         home_goals=2, away_goals=1))
    return pd.DataFrame(rows)


def _one(with_actual=True):
    one = pd.DataFrame([
        dict(model="Poisson", p_H=.5, p_D=.25, p_A=.25, home_xg_pred=1.6, away_xg_pred=1.0),
        dict(model="Logistic", p_H=.48, p_D=.26, p_A=.26, home_xg_pred=np.nan, away_xg_pred=np.nan)])
    if with_actual:
        one["actual_score"], one["actual_result"] = "2-1", "H"
    return one


def test_scoreline_grid_sums_to_one_and_peaks_near_expected_goals():
    grid = scoreline_grid(1.6, 1.3)
    assert abs(grid.sum() - 1) < 1e-9
    assert np.unravel_index(grid.argmax(), grid.shape) == (1, 1)


def test_match_plot_with_and_without_actual_result():
    for with_actual in (True, False):
        fig = plot_match_prediction(_one(with_actual), _matches(), "Alpha", "Beta", "2026-04-20")
        assert len(fig.axes) == 4


def test_next_round_plot():
    preds = pd.DataFrame([
        dict(date=pd.Timestamp("2026-05-01"), home_team="Alpha", away_team="Beta", model="Poisson",
             p_H=.5, p_D=.25, p_A=.25),
        dict(date=pd.Timestamp("2026-05-02"), home_team="Gamma", away_team="Delta", model="Poisson",
             p_H=.3, p_D=.3, p_A=.4)])
    assert len(plot_next_round(preds).axes) == 1
