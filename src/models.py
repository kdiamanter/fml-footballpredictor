"""Prediction models.

- ``HistoricalBaseline``: the share of home wins, draws and away wins in past
  matches. Ignores the teams; every other model must beat it.
- ``LogisticModel``: multinomial logistic regression on the home-minus-away
  difference features. Predicts H/D/A directly.
- ``PoissonModel``: two Poisson regressions, one for home goals and one for
  away goals. The expected goals are turned into a scoreline grid, and H/D/A
  probabilities are summed from that grid.

Every model returns probabilities as an array of shape (n, 3) with columns in
``CLASSES`` order (H, D, A). Each ``fit_*_before(date)`` helper trains only on
matches played strictly before ``date``, so it can be used in the Stage 3
walk-forward backtest without leakage.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation import CLASSES, check_probabilities
from src.features import DIFF_FEATURES, FEATURE_COLUMNS, drop_incomplete, matches_before

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

# Inputs per model. The logistic model uses the differences only (8 features)
# because home and away features carry the same information twice. The
# Poisson model needs the separate home and away values (16 features): home
# goals depend on the home attack AND the away defence, not just their gap.
LOGISTIC_FEATURES = DIFF_FEATURES
POISSON_FEATURES = FEATURE_COLUMNS

# Scorelines up to 10-10 are summed. The chance of 11+ goals for one team is
# negligible; the grid is renormalised so the probabilities still sum to 1.
MAX_GOALS = 10

# Regularisation chosen in notebook 02 (trained before 2024/25, scored on
# 2024/25 by log loss). Results were flat around these values.
DEFAULT_C = 0.1
DEFAULT_ALPHA = 0.1


def _check_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if df[columns].isna().any().any():
        raise ValueError("Features contain NaN. Use drop_incomplete() first.")


def _train_rows(matches: pd.DataFrame, date) -> pd.DataFrame:
    """Complete feature rows for matches strictly before ``date``."""
    train = drop_incomplete(matches_before(matches, date))
    if train.empty:
        raise ValueError(f"No complete matches before {pd.Timestamp(date).date()} to train on.")
    return train


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
class HistoricalBaseline:
    """Always predicts the historical share of home wins, draws and away wins.

    Example: if 45% of past matches were home wins, 25% draws and 30% away
    wins, every match gets the probabilities (0.45, 0.25, 0.30). It ignores
    which teams are playing, so any useful model should beat it.
    """

    def __init__(self) -> None:
        self.probabilities_: np.ndarray | None = None
        self.n_matches_: int = 0

    def fit(self, results: pd.Series) -> "HistoricalBaseline":
        """Learn the frequencies from a Series of 'H'/'D'/'A' labels."""
        results = pd.Series(results).dropna()
        unexpected = set(results) - set(CLASSES)
        if unexpected:
            raise ValueError(f"Unexpected result labels: {unexpected}")
        if results.empty:
            raise ValueError("Cannot fit the baseline on zero matches.")
        counts = results.value_counts().reindex(CLASSES, fill_value=0)
        self.probabilities_ = (counts / counts.sum()).to_numpy(dtype=float)
        self.n_matches_ = len(results)
        return self

    def predict_proba(self, n_matches: int | pd.DataFrame = 1) -> np.ndarray:
        """Return an array of shape (n_matches, 3), columns in ``CLASSES`` order.

        Accepts a number of matches, or a table of matches (its length is used),
        so it can be called the same way as the other models.
        """
        if self.probabilities_ is None:
            raise RuntimeError("Call fit() before predict_proba().")
        if not isinstance(n_matches, (int, np.integer)):
            n_matches = len(n_matches)
        probs = np.tile(self.probabilities_, (n_matches, 1))
        check_probabilities(probs)
        return probs


def fit_baseline_before(matches: pd.DataFrame, date) -> HistoricalBaseline:
    """Fit the baseline on matches played strictly before ``date``."""
    return HistoricalBaseline().fit(matches_before(matches, date)["result"])


# ---------------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------------
class LogisticModel:
    """Multinomial logistic regression: features -> P(H), P(D), P(A).

    Features are standardised first so the regularisation strength ``C``
    treats them equally. Smaller ``C`` means stronger regularisation, which
    keeps the correlated form features (xG, goals, points) from fighting
    each other with large opposite-sign coefficients.
    """

    def __init__(self, C: float = DEFAULT_C, features: list[str] = LOGISTIC_FEATURES) -> None:
        self.C = C
        self.features = list(features)
        self.pipeline_ = None
        self.n_matches_: int = 0

    def fit(self, df: pd.DataFrame) -> "LogisticModel":
        _check_columns(df, self.features)
        if set(df["result"]) - set(CLASSES):
            raise ValueError(f"Unexpected result labels: {set(df['result']) - set(CLASSES)}")
        self.pipeline_ = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=self.C, max_iter=1000),
        )
        self.pipeline_.fit(df[self.features], df["result"])
        self.n_matches_ = len(df)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Shape (n, 3), columns in ``CLASSES`` order."""
        if self.pipeline_ is None:
            raise RuntimeError("Call fit() before predict_proba().")
        _check_columns(df, self.features)
        raw = self.pipeline_.predict_proba(df[self.features])
        # sklearn orders classes alphabetically (A, D, H); reorder to H, D, A.
        # A class missing from the training data gets probability 0.
        learned = list(self.pipeline_.classes_)
        probs = np.zeros((len(df), len(CLASSES)))
        for j, label in enumerate(CLASSES):
            if label in learned:
                probs[:, j] = raw[:, learned.index(label)]
        check_probabilities(probs)
        return probs

    def coefficients(self) -> pd.DataFrame:
        """Coefficients per outcome (on standardised features), for interpretation."""
        if self.pipeline_ is None:
            raise RuntimeError("Call fit() first.")
        lr = self.pipeline_[-1]
        return pd.DataFrame(lr.coef_, index=lr.classes_, columns=self.features).reindex(
            [c for c in CLASSES if c in lr.classes_]
        )


def fit_logistic_before(matches: pd.DataFrame, date, **kwargs) -> LogisticModel:
    """Fit the logistic model on complete matches played strictly before ``date``."""
    return LogisticModel(**kwargs).fit(_train_rows(matches, date))


# ---------------------------------------------------------------------------
# Poisson regression
# ---------------------------------------------------------------------------
def outcome_probs_from_goals(home_mu, away_mu, max_goals: int = MAX_GOALS) -> np.ndarray:
    """Turn expected home and away goals into P(H), P(D), P(A).

    Assumes home and away goals are independent Poisson counts. Builds the
    grid P(home = i) * P(away = j) for 0..max_goals and sums the cells below
    the diagonal (home win), on it (draw) and above it (away win).
    """
    home_mu = np.atleast_1d(np.asarray(home_mu, dtype=float))
    away_mu = np.atleast_1d(np.asarray(away_mu, dtype=float))
    if (home_mu <= 0).any() or (away_mu <= 0).any():
        raise ValueError("Expected goals must be positive.")
    goals = np.arange(max_goals + 1)
    p_home = poisson.pmf(goals[None, :], home_mu[:, None])   # (n, G)
    p_away = poisson.pmf(goals[None, :], away_mu[:, None])   # (n, G)
    grid = p_home[:, :, None] * p_away[:, None, :]            # (n, G, G): [home, away]

    home_win = np.tril(np.ones((len(goals), len(goals))), k=-1)  # home > away
    draw = np.eye(len(goals))
    away_win = np.triu(np.ones((len(goals), len(goals))), k=1)   # away > home
    probs = np.stack([
        (grid * home_win).sum(axis=(1, 2)),
        (grid * draw).sum(axis=(1, 2)),
        (grid * away_win).sum(axis=(1, 2)),
    ], axis=1)
    probs /= probs.sum(axis=1, keepdims=True)
    check_probabilities(probs)
    return probs


class PoissonModel:
    """Two Poisson regressions: one predicts home goals, one away goals.

    ``alpha`` is the L2 regularisation strength (larger = stronger).
    """

    def __init__(self, alpha: float = DEFAULT_ALPHA, features: list[str] = POISSON_FEATURES,
                 max_goals: int = MAX_GOALS) -> None:
        self.alpha = alpha
        self.features = list(features)
        self.max_goals = max_goals
        self.home_ = None
        self.away_ = None
        self.n_matches_: int = 0

    def _new_regressor(self):
        return make_pipeline(StandardScaler(), PoissonRegressor(alpha=self.alpha, max_iter=1000))

    def fit(self, df: pd.DataFrame) -> "PoissonModel":
        _check_columns(df, self.features + ["home_goals", "away_goals"])
        self.home_ = self._new_regressor().fit(df[self.features], df["home_goals"])
        self.away_ = self._new_regressor().fit(df[self.features], df["away_goals"])
        self.n_matches_ = len(df)
        return self

    def predict_goals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Expected home and away goals per match."""
        if self.home_ is None:
            raise RuntimeError("Call fit() before predicting.")
        _check_columns(df, self.features)
        return pd.DataFrame({
            "home_mu": self.home_.predict(df[self.features]),
            "away_mu": self.away_.predict(df[self.features]),
        }, index=df.index)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Shape (n, 3), columns in ``CLASSES`` order."""
        mu = self.predict_goals(df)
        return outcome_probs_from_goals(mu["home_mu"], mu["away_mu"], self.max_goals)


def fit_poisson_before(matches: pd.DataFrame, date, **kwargs) -> PoissonModel:
    """Fit the Poisson model on complete matches played strictly before ``date``."""
    return PoissonModel(**kwargs).fit(_train_rows(matches, date))


# ---------------------------------------------------------------------------
# Saving and loading
# ---------------------------------------------------------------------------
def save_model(model, name: str, directory: Path = MODELS_DIR) -> Path:
    """Save a fitted model to ``models/<name>.joblib``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def load_model(name: str, directory: Path = MODELS_DIR):
    """Load a model saved with ``save_model``."""
    path = Path(directory) / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}. Run notebook 02 first.")
    return joblib.load(path)
