"""Small, hand-made datasets for the tests. No internet needed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import match_outcome

TEAMS = ["Arsenal", "Brentford", "Chelsea", "Everton"]


def make_league(n_cycles: int = 3, seed: int = 42) -> pd.DataFrame:
    """A tiny 4-team league: double round robin repeated ``n_cycles`` times.

    One round per week, two matches per round (Saturday and Sunday), so every
    team plays at most once per day. Goals and xG are random but fixed by the
    seed, so every test run uses the same data.
    """
    rng = np.random.default_rng(seed)
    a, b, c, d = TEAMS
    first_half = [[(a, b), (c, d)], [(a, c), (b, d)], [(a, d), (b, c)]]
    second_half = [[(h2, h1) for h1, h2 in rnd] for rnd in first_half]
    rounds = (first_half + second_half) * n_cycles

    rows = []
    start = pd.Timestamp("2020-08-01 15:00")
    for week, fixtures in enumerate(rounds):
        for i, (home, away) in enumerate(fixtures):
            home_xg = round(float(rng.uniform(0.3, 2.8)), 2)
            away_xg = round(float(rng.uniform(0.2, 2.2)), 2)
            rows.append({
                "date": start + pd.Timedelta(weeks=week, days=i),
                "home_team": home,
                "away_team": away,
                "home_goals": int(rng.poisson(home_xg)),
                "away_goals": int(rng.poisson(away_xg)),
                "home_xg": home_xg,
                "away_xg": away_xg,
            })
    df = pd.DataFrame(rows)
    df["result"] = match_outcome(df["home_goals"], df["away_goals"])
    return df


@pytest.fixture
def league() -> pd.DataFrame:
    return make_league()
