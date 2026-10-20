"""Pictures for match predictions.

``plot_match_prediction`` draws one match on one page:
  1. the H/D/A probabilities of both models,
  2. the scoreline grid (how likely each exact score is, Poisson model),
  3. each team's last 5 matches (xG created vs xG conceded).

It takes the table returned by ``predict_match_as_of`` (or the rows of
``predict_fixtures`` for one match) plus the played ``matches``.
``plot_next_round`` draws all predicted matches of a round in one chart.
"""
from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.features import WINDOW, matches_before, to_team_matches

HOME_COLOR = "#2b6cb0"   # blue
DRAW_COLOR = "#a0aec0"   # grey
AWAY_COLOR = "#c05621"   # orange
GRID_GOALS = 5           # the grid shows 0..5 goals per team
FULL_GOALS = 10          # probabilities are normalised over 0..10, like the model


def scoreline_grid(home_mu: float, away_mu: float) -> np.ndarray:
    """Probability of every exact score (rows = home goals, columns = away goals)."""
    goals = np.arange(FULL_GOALS + 1)
    grid = np.outer(poisson.pmf(goals, home_mu), poisson.pmf(goals, away_mu))
    return grid / grid.sum()


def _plot_probabilities(ax, one: pd.DataFrame, home: str, away: str) -> None:
    models = list(one["model"])
    labels = [f"{home} win", "Draw", f"{away} win"]
    colors = [HOME_COLOR, DRAW_COLOR, AWAY_COLOR]
    height = 0.8 / max(len(models), 1)
    actual = one["actual_result"].iloc[0] if "actual_result" in one else None
    for i, (_, r) in enumerate(one.iterrows()):
        vals = [r["p_H"], r["p_D"], r["p_A"]]
        y = np.arange(3) + (i - (len(models) - 1) / 2) * height
        bars = ax.barh(y, vals, height=height * 0.9, color=colors, alpha=0.95 - 0.4 * i)
        for bar, v in zip(bars, vals):
            ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.0%}", va="center", fontsize=9)
    if actual in ("H", "D", "A"):
        labels[["H", "D", "A"].index(actual)] += "  \u2605 actual"
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.set_title("Chance of each result", loc="left", fontsize=11, fontweight="bold")
    # Bars of each result share a colour; the lighter bar is the later model listed.
    ax.text(1, -0.12, " / ".join(f"{m}{' (darker)' if i == 0 else ' (lighter)'}" for i, m in enumerate(models)),
            transform=ax.transAxes, ha="right", fontsize=8, color="#4a5568")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _plot_scoreline_grid(ax, one: pd.DataFrame, home: str, away: str) -> None:
    poi = one[one["model"] == "Poisson"]
    if poi.empty or pd.isna(poi["home_xg_pred"].iloc[0]):
        ax.axis("off")
        ax.text(0.5, 0.5, "Scoreline grid needs the Poisson model", ha="center", va="center")
        return
    mu_h, mu_a = float(poi["home_xg_pred"].iloc[0]), float(poi["away_xg_pred"].iloc[0])
    grid = scoreline_grid(mu_h, mu_a)
    shown = grid[:GRID_GOALS + 1, :GRID_GOALS + 1]
    ax.imshow(shown, cmap="Blues", origin="upper")
    for i in range(shown.shape[0]):
        for j in range(shown.shape[1]):
            ax.text(j, i, f"{shown[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if shown[i, j] > shown.max() * 0.6 else "#2d3748")
    bi, bj = np.unravel_index(shown.argmax(), shown.shape)
    ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, edgecolor="#c05621", lw=2.2))
    if "actual_score" in one.columns and isinstance(one["actual_score"].iloc[0], str):
        ah, aa = [int(x) for x in one["actual_score"].iloc[0].split("-")]
        if ah <= GRID_GOALS and aa <= GRID_GOALS:
            ax.plot(aa, ah, marker="*", markersize=16, color="#d69e2e", markeredgecolor="#2d3748")
    ax.set_xticks(range(GRID_GOALS + 1))
    ax.set_yticks(range(GRID_GOALS + 1))
    ax.set_xlabel(f"{away} goals")
    ax.set_ylabel(f"{home} goals")
    ax.set_title(f"Likely scores (expected goals {mu_h:.1f} - {mu_a:.1f})", loc="left",
                 fontsize=11, fontweight="bold")
    note = "orange box = likeliest score"
    if "actual_score" in one.columns and isinstance(one["actual_score"].iloc[0], str):
        note += ", star = actual score"
    ax.text(0, -0.2, note, transform=ax.transAxes, fontsize=8, color="#4a5568")


def _plot_form(ax, matches: pd.DataFrame, team: str, date, venue: str, color: str) -> None:
    rows = to_team_matches(matches_before(matches, date))
    team_rows = rows[rows["team"] == team]
    last = team_rows.tail(5).reset_index(drop=True)
    x = np.arange(len(last))
    ax.bar(x - 0.2, last["xg_for"], width=0.4, color=color, label="xG created")
    ax.bar(x + 0.2, last["xg_against"], width=0.4, color="#cbd5e0", label="xG conceded")
    labels = [f"{textwrap.fill(o, 11, break_long_words=False)}\n({v[0].upper()}) {int(gf)}-{int(ga)}" for o, v, gf, ga
              in zip(last["opponent"], last["venue"], last["goals_for"], last["goals_against"])]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_title(f"{team}: last 5 matches ({'home' if venue == 'home' else 'away'} side)",
                 loc="left", fontsize=10, fontweight="bold")
    ax.set_ylabel("xG")
    title_extra = ""
    if WINDOW > 5:
        # The models average a longer window than the 5 bars shown: draw that average too.
        longer = team_rows.tail(WINDOW)
        ax.axhline(longer["xg_for"].mean(), color=color, ls="--", lw=1.2)
        ax.axhline(longer["xg_against"].mean(), color="#718096", ls="--", lw=1.2)
        title_extra = f"\ndashed = average of last {WINDOW} (what the model uses)"
        ax.set_title(ax.get_title() + title_extra, loc="left", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(last["xg_for"].max(), last["xg_against"].max()) * 1.3)
    ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_match_prediction(one: pd.DataFrame, matches: pd.DataFrame, home: str, away: str, date):
    """One-page picture of a match prediction. Returns the matplotlib figure."""
    one = one.dropna(subset=["model"]).reset_index(drop=True)
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
    _plot_probabilities(fig.add_subplot(gs[0, 0]), one, home, away)
    _plot_scoreline_grid(fig.add_subplot(gs[0, 1]), one, home, away)
    _plot_form(fig.add_subplot(gs[1, 0]), matches, home, date, "home", HOME_COLOR)
    _plot_form(fig.add_subplot(gs[1, 1]), matches, away, date, "away", AWAY_COLOR)
    title = f"{home} vs {away}, {pd.Timestamp(date):%d %B %Y}"
    if "actual_score" in one.columns and isinstance(one["actual_score"].iloc[0], str):
        title += f"  (actual: {one['actual_score'].iloc[0]})"
    fig.suptitle(title, fontsize=14, fontweight="bold")
    return fig


def plot_next_round(preds: pd.DataFrame, model: str = "Poisson"):
    """Stacked home/draw/away bars for every predicted match of a round."""
    df = preds[preds["model"] == model].sort_values("date", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, max(2.5, 0.5 * len(df) + 1)), constrained_layout=True)
    y = np.arange(len(df))
    left = np.zeros(len(df))
    for col, color, name in [("p_H", HOME_COLOR, "home win"), ("p_D", DRAW_COLOR, "draw"),
                             ("p_A", AWAY_COLOR, "away win")]:
        ax.barh(y, df[col], left=left, color=color, label=name)
        for yi, (l, v) in enumerate(zip(left, df[col])):
            if v > 0.08:
                ax.text(l + v / 2, yi, f"{v:.0%}", ha="center", va="center", color="white", fontsize=9)
        left += df[col].to_numpy()
    ax.set_yticks(y)
    ax.set_yticklabels([f"{h} vs {a}" for h, a in zip(df["home_team"], df["away_team"])])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.set_title(f"Next round: chance of each result ({model} model)", loc="left", fontweight="bold")
    ax.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.18))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return fig
