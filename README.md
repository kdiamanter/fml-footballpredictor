# FML Football Predictor

A university project that predicts Premier League matches from historical
performance, especially expected goals (xG).

**Research question:** How accurate are our pre-match predictions compared
with actual match results?

Each match gets a home win, draw and away win probability (summing to 1).
The focus is on the *quality of the probabilities* (log loss, Brier score,
ranked probability score, calibration), not just accuracy.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                             # all tests should pass, no internet needed
jupyter lab
```

Then open `notebooks/01_data_preparation.ipynb` and run all cells. The first
run downloads the data from Understat (a few minutes) and caches it in
`data/raw/`; later runs work offline.

## Structure

| Path | Contents |
|---|---|
| `notebooks/01_data_preparation.ipynb` | download, cleaning, features, baseline |
| `notebooks/02_model_training.ipynb` | logistic and Poisson regression, regularisation, saving |
| `notebooks/03_model_evaluation.ipynb` | walk-forward backtest, scores, calibration, findings |
| `notebooks/04_predictions.ipynb` | refresh data, predict the next round, score past predictions |
| `notebooks/05_season_replay.ipynb` | replay a past season with a model frozen before it (for presentations) |
| `src/data.py` | Understat download with caching, cleaning, outcome labels |
| `src/features.py` | leakage-safe rolling features |
| `src/models.py` | prediction models |
| `src/evaluation.py` | log loss, Brier, RPS, calibration, walk-forward backtest |
| `src/predict.py` | upcoming-fixture predictions and the prediction log |
| `data/predictions/` | dated prediction files, scored once the matches are played |
| `tests/` | pytest tests on small hand-made datasets |
| `STATUS.md` | progress log: done, problems, next steps |

## Data

English Premier League, seasons 2016/17 to 2025/26 plus the current season,
from [Understat](https://understat.com) via the
[`soccerdata`](https://soccerdata.readthedocs.io) library.

## No data leakage

A prediction for a match on day *D* uses only matches played strictly before
day *D*. Rolling features shift each team's history by one match before
averaging, so a match never sees its own result. `tests/test_leakage.py`
predicts a historical match, scrambles all later results, predicts again and
requires an identical answer.
