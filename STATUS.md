# STATUS

## Current stage: 4 (predictions) done. All four stages built

## Stage 1: data, features, baseline (approved 2026-10-13)

- `src/data.py`: cached Understat download via soccerdata (EPL, 2016/17–2025/26 + current
  2026/27), cleaning, H/D/A outcome label.
- `src/features.py`: leakage-safe rolling features (last 5 matches + last 5 home/away matches)
  and home-minus-away differences.
- `src/models.py`: historical-frequency baseline.
- Real data run done: 380 matches and 20 teams in every completed season, 60 matches of
  2026/27 so far, 35 distinct club names with no duplicates. Outcome shares H 44.5% /
  D 23.4% / A 32.1%.
- Accepted as is: rolling windows cross season boundaries; matches where a team has fewer
  than 5 earlier matches in the data are left out of modelling (3,710 of 3,860 kept).

## Stage 2: models (approved 2026-10-14)

### Done
- `src/models.py`:
  - `LogisticModel`: standardised features → multinomial logistic regression on the 8
    difference features. Output reordered from sklearn's alphabetical A/D/H to our H/D/A.
  - `PoissonModel`: two Poisson regressions (home goals, away goals) on all 16 home/away
    features; expected goals → independent-Poisson scoreline grid (0–10 goals) → H/D/A.
  - `fit_logistic_before(date)` / `fit_poisson_before(date)`: train only on matches before a
    date, ready for the Stage 3 walk-forward backtest.
  - `save_model` / `load_model` (joblib, `models/`).
- `src/evaluation.py`: `log_loss` in our own H/D/A order. **Do not use
  `sklearn.metrics.log_loss` with string labels**: it assumes alphabetical column order and
  silently gives wrong numbers (it made both models look worse than the baseline).
- `notebooks/02_model_training.ipynb`: time-based splits, regularisation choice, coefficients,
  expected-goals sanity check, first check on 2025/26, saves final models.
- Tests: 51 passing. New: H/D/A column order, Poisson grid hand-check, models learn the right
  direction, fit-before uses only the past, save/load round trip, log loss hand-check, and a
  leakage test that models fitted before a date are unchanged when later results are scrambled.

### Results so far
- Regularisation (train 2016/17–2023/24, score 2024/25): log loss flat for C ≥ 0.03 and
  alpha 0.01–0.3. Chosen C = 0.1, alpha = 0.1. Logistic 0.990, Poisson 0.990, baseline 1.078.
- First check on 2025/26 (trained on 2016/17–2024/25, one fixed model):

  | Model | Log loss | Accuracy |
  |---|---|---|
  | Uniform 1/3 | 1.099 | – |
  | Baseline | 1.085 | 42.6% |
  | Logistic | 1.047 | 47.9% |
  | Poisson | 1.050 | 48.7% |

  2025/26 had an unusually high draw share (27.4% vs 23.4% overall), which makes it a
  harder season than 2024/25.
- Saved models in `models/` are fitted on all 3,710 complete matches up to 2026-10-11 (for
  Stage 4 predictions). The Stage 3 backtest must refit with `fit_*_before`, not load these.

### Open points
- Poisson independence assumption: checked in Stage 3, draw calibration is fine.
- Features are correlated, so individual logistic coefficients should not be over-interpreted.

## Stage 3: evaluation (approved 2026-10-14)

### Done
- `src/evaluation.py`: log loss, Brier score, RPS (per match and averaged), accuracy,
  `score_table`, `paired_difference` (paired 95% interval), `calibration_table`,
  `walk_forward_backtest` (refits every model before each matchday), `probs_of`.
- `HistoricalBaseline.predict_proba` now also accepts a table of matches, so all models are
  called the same way.
- `notebooks/03_model_evaluation.ipynb`: backtest 2019/20 to today, scores, paired
  comparisons, per-season RPS, calibration plots and tables, draw check, confident
  predictions, written findings and limitations. Saves
  `data/processed/backtest_predictions.csv`.
- Tests: 63 passing. New: Brier/RPS hand checks, RPS respects outcome order, calibration
  table, paired difference, backtest output, every backtest model trains only on earlier
  days, backtest predictions unchanged when later results are scrambled.

### Results (2,666 matches, 2019-08-10 to 2026-10-11, refit every matchday)

| Model | Log loss | Brier | RPS | Accuracy |
|---|---|---|---|---|
| Always 1/3 | 1.099 | 0.667 | 0.238 | – |
| Baseline | 1.071 | 0.649 | 0.234 | 43.3% |
| Logistic | 1.006 | 0.600 | 0.210 | 51.3% |
| Poisson | 1.005 | 0.600 | 0.210 | 51.5% |
| Logistic, xG only | 1.005 | 0.599 | 0.209 | 51.4% |
| Logistic, results only | 1.023 | 0.614 | 0.216 | 50.0% |

- Both models beat the baseline clearly (RPS −0.024, 95% interval −0.029 to −0.019).
- Logistic vs Poisson: no difference (RPS interval −0.0008 to +0.0008).
- xG only beats results only clearly (RPS −0.0066, interval −0.0098 to −0.0034); the full
  model is no better than xG only. Recent xG carries the useful information.
- Calibration good; strong home favourites (70–80%) slightly overconfident.
- Draws: average predicted 23% matches the actual 23.8%, but predictions barely vary
  (mostly 20–30%). Dixon–Coles not needed for calibration, so not added.
- 2020/21 (no crowds): home wins dropped to 37.9%, the baseline's worst season.

## Stage 4: predictions

### Done
- `src/data.py`: `upcoming_fixtures(raw)` reads unplayed fixtures from the raw file (the raw
  download already contains the full 2026/27 fixture list).
- `src/predict.py`: `split_upcoming` (next round = consecutive match days until a 3+ day gap;
  fixtures dated before the latest result are treated as postponed), `predict_fixtures`
  (both models, Poisson expected goals and most likely score with its probability; unknown
  teams / too little history give a note, not a guess), prediction log
  (`save_predictions`, `load_saved_predictions`, `score_saved_predictions`: only predictions
  made before the match day count, latest one wins).
- `notebooks/04_predictions.ipynb`: `FORCE_DOWNLOAD` switch at the top; refreshes data,
  retrains and saves models, predicts the next round (table + chart), saves the predictions,
  scores earlier saved predictions, predicts any chosen fixture.
- `predict_match_as_of(matches, home, away, date)`: fair prediction for any past or future
  match. Models and features use only matches before the date; the real score is attached
  if played. Notebook 04 section 6 uses it, plus a team/season match list for picking dates.
- First prediction file: `data/predictions/predictions_made_2026-10-14.csv` (10 matches,
  17–19 Oct 2026).
- Tests: 70 passing.

## Season replay (notebook 05)

- `season_replay(features, season)` in `src/predict.py`: trains baseline, logistic and Poisson
  once on all matches before the season, freezes them, predicts every match of the season.
  Form features still use each team's last 5 matches before kickoff.
- `notebooks/05_season_replay.ipynb`: settings `SEASON`, `MODEL`, `TEAM`; scores vs baseline,
  running RPS through the season, every match table, biggest upsets, most confident correct
  calls, favourite-won check, one team's season, summary. Saves
  `data/processed/season_replay_<season>.csv`.
- 2025/26 (cleanest season: nothing in it used for building or tuning): RPS Poisson 0.216,
  logistic 0.215, baseline 0.228, always-1/3 0.232; favourite won 49% / 48% / 43%. Poisson vs
  baseline RPS −0.012 (95% interval −0.023 to −0.001). Models trailed the baseline in August
  (stale early-season form), led from mid-September on.
- Tests: 71 passing.

### How to use weekly
After each round: notebook 04 with `FORCE_DOWNLOAD = True`, run all, set it back to False.
Section 5 then scores the previous round's saved predictions.

### Next steps
1. Review notebook 04.
2. After 17–19 Oct: refresh and check the first live scores in notebook 04, section 5.
3. Full plain-language walkthrough of the whole project (user request).
