"""From a matchup view to priced bets, and the walk-forward backtest that grades them.

Two stages, both fitted only on games that finished before the one being priced:

1. a ridge regression turns the matchup features into the bot's own margin and
   total;
2. that number is shrunk toward the sportsbook line by however much the bot's
   disagreements with the line have actually been worth (``beta``). ``beta`` is
   measured only on out-of-sample predictions and taken one standard error below
   its estimate, so it is zero unless the evidence for an edge is real. A ``beta``
   of zero means "the line already knows everything the bot knows", and then no
   bet is ever recommended.

A bet is recommended only when its expected value at the offered price clears
``min_ev``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, replace
from statistics import NormalDist
from typing import Any

import numpy as np

from nba_bets.features import (
    MARGIN_FEATURES,
    TOTAL_FEATURES,
    League,
    Matchup,
    available_from_box,
    game_day,
)

DEFAULT_PRICE = -110.0
MIN_EV = 0.02
MIN_TRAIN = 500
MAX_GAP = 8.0  # points; larger disagreements with the line are treated as this size
MIN_EVIDENCE = 300  # out-of-sample games needed before any edge over the line is credited
EDGE_BUCKETS = (0.0, 0.02, 0.04, 0.07, 1.0)

# --------------------------------------------------------------------------- odds math


def decimal_odds(american: float | None) -> float:
    a = DEFAULT_PRICE if american is None else american
    return 1 + (a / 100 if a > 0 else 100 / -a)


def implied(american: float | None) -> float:
    return 1 / decimal_odds(american)


def devig(a: float | None, b: float | None) -> tuple[float, float]:
    pa, pb = implied(a), implied(b)
    return pa / (pa + pb), pb / (pa + pb)


def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# --------------------------------------------------------------------------- fitting


@dataclass
class Fit:
    w_margin: np.ndarray
    w_total: np.ndarray
    beta_margin: float
    beta_total: float
    sigma_margin: float
    sigma_total: float
    n_train: int
    win_scale: float = 0.11  # logistic slope turning an expected margin into a win probability

    def win_probability(self, margin: float) -> float:
        return 1 / (1 + math.exp(-self.win_scale * margin))

    def model_margin(self, features: dict[str, float]) -> float:
        return float(self.w_margin @ np.array([features[k] for k in MARGIN_FEATURES]))

    def model_total(self, features: dict[str, float]) -> float:
        x = np.array([features[k] for k in TOTAL_FEATURES] + [1.0])
        return float(self.w_total @ x)


def _ridge(x: np.ndarray, y: np.ndarray, lam: float, free: int) -> np.ndarray:
    """Ridge with per-column scaling; the last ``free`` columns are not penalised."""
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    xs = x / scale
    penalty = np.full(x.shape[1], lam)
    if free:
        penalty[-free:] = 0.0
    w = np.linalg.solve(xs.T @ xs + np.diag(penalty), xs.T @ y)
    return w / scale


def credited_slope(pairs: list[tuple[float, float]]) -> tuple[float, float, float]:
    """How much of a disagreement with the line to believe.

    ``pairs`` are out-of-sample (model - line, result - line). The fit is on what a
    bet actually pays for, the side of the line the result landed on, so a few
    overtime blowouts cannot masquerade as an edge: P(result beats line) is modelled
    as ``0.5 + slope * disagreement / 2``. Returns the raw slope, its standard
    error, and the slope actually credited: one standard error below the estimate,
    floored at zero.
    """
    kept = [
        (float(np.clip(d, -MAX_GAP, MAX_GAP)), math.copysign(1.0, r)) for d, r in pairs if r != 0
    ]
    if len(kept) < MIN_EVIDENCE:
        return 0.0, 0.0, 0.0
    d, won = np.array(kept).T
    denom = float(d @ d)
    if denom <= 0:
        return 0.0, 0.0, 0.0
    slope = float(d @ won / denom)
    se = float(np.sqrt(np.mean((won - slope * d) ** 2) / denom))
    return slope, se, max(slope - se, 0.0)


def fit(rows: list[dict[str, Any]], lam: float = 25.0) -> Fit:
    xm = np.array([[r["features"][k] for k in MARGIN_FEATURES] for r in rows])
    xt = np.array([[r["features"][k] for k in TOTAL_FEATURES] + [1.0] for r in rows])
    margin = np.array([r["margin"] for r in rows])
    total = np.array([r["total"] for r in rows])
    # The home indicator is the last margin feature and acts as the intercept.
    w_margin = _ridge(xm, margin, lam, free=1)
    w_total = _ridge(xt, total, lam, free=1)
    pm, pt = xm @ w_margin, xt @ w_total

    # Spread of results around the line itself: the scale for cover probabilities.
    sigma_m, sigma_t = float(np.std(margin - pm)), float(np.std(total - pt))
    spreads = [(r["margin"], -r["spread"]) for r in rows if r["spread"] is not None]
    totals = [(r["total"], r["total_line"]) for r in rows if r["total_line"] is not None]
    if len(spreads) >= 200:
        sigma_m = float(np.std([a - b for a, b in spreads]))
    if len(totals) >= 200:
        sigma_t = float(np.std([a - b for a, b in totals]))
    win_scale = 0.11
    if len(spreads) >= 200:
        # Learn "expected margin -> win probability" from the best expected margin there is.
        won = np.array([a > 0 for a, _ in spreads], dtype=float)
        line = np.array([b for _, b in spreads])
        grid = np.linspace(0.05, 0.25, 81)
        loss = [
            -np.mean(won * np.log(p) + (1 - won) * np.log(1 - p))
            for p in (_sigmoid(g * line) for g in grid)
        ]
        win_scale = float(grid[int(np.argmin(loss))])
    return Fit(w_margin, w_total, 0.0, 0.0, sigma_m, sigma_t, len(rows), win_scale)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.clip(1 / (1 + np.exp(-x)), 1e-6, 1 - 1e-6)


# --------------------------------------------------------------------------- pricing


@dataclass
class Candidate:
    market: str  # "spread" | "total" | "moneyline"
    side: str  # "home" | "away" | "over" | "under"
    line: float | None
    price: float
    prob: float
    ev: float

    @property
    def bucket(self) -> int:
        return (
            max(i for i, low in enumerate(EDGE_BUCKETS[:-1]) if self.ev >= low)
            if self.ev >= 0
            else -1
        )


@dataclass
class Pricing:
    model_margin: float
    model_total: float
    fair_margin: float
    fair_total: float
    p_home_win: float  # after shrinking toward the line: what bets are priced with
    candidates: list[Candidate]
    model_p_home: float = 0.5  # the model's own view, before looking at the line
    market_p_home: float | None = None

    def best(self, min_ev: float = MIN_EV) -> list[Candidate]:
        """At most one side bet and one total bet per game, best expected value first."""
        picks = []
        for markets in (("spread", "moneyline"), ("total",)):
            pool = [c for c in self.candidates if c.market in markets and c.ev >= min_ev]
            if pool:
                picks.append(max(pool, key=lambda c: c.ev))
        return sorted(picks, key=lambda c: -c.ev)


def _candidate(
    market: str, side: str, line: float | None, price: float | None, prob: float
) -> Candidate:
    dec = decimal_odds(price)
    return Candidate(
        market,
        side,
        line,
        DEFAULT_PRICE if price is None else price,
        prob,
        prob * (dec - 1) - (1 - prob),
    )


def price(fitted: Fit, features: dict[str, float], odds: dict[str, Any] | None) -> Pricing:
    odds = odds or {}
    model_margin, model_total = fitted.model_margin(features), fitted.model_total(features)
    spread, total_line = odds.get("spread"), odds.get("total")
    fair_margin, fair_total = model_margin, model_total
    model_p = p_home_win = fitted.win_probability(model_margin)
    market_p = None
    normal = NormalDist()
    out: list[Candidate] = []
    if spread is not None:
        gap = float(np.clip(model_margin + spread, -MAX_GAP, MAX_GAP))
        p_cover = 0.5 + 0.5 * fitted.beta_margin * gap
        z = normal.inv_cdf(p_cover)
        fair_margin = -spread + z * fitted.sigma_margin
        out.append(_candidate("spread", "home", spread, odds.get("spread_odds_home"), p_cover))
        out.append(_candidate("spread", "away", -spread, odds.get("spread_odds_away"), 1 - p_cover))
        p_home_win = fitted.win_probability(fair_margin)
        if odds.get("ml_home") is not None and odds.get("ml_away") is not None:
            # Start from the book's own no-vig win probability and move it only by the
            # edge we credit ourselves with on the spread.
            market_p = min(max(devig(odds["ml_home"], odds["ml_away"])[0], 1e-4), 1 - 1e-4)
            p_home_win = norm_cdf(normal.inv_cdf(market_p) + z)
            out.append(_candidate("moneyline", "home", None, odds["ml_home"], p_home_win))
            out.append(_candidate("moneyline", "away", None, odds["ml_away"], 1 - p_home_win))
    if total_line is not None:
        gap = float(np.clip(model_total - total_line, -MAX_GAP, MAX_GAP))
        p_over = 0.5 + 0.5 * fitted.beta_total * gap
        fair_total = total_line + normal.inv_cdf(p_over) * fitted.sigma_total
        out.append(_candidate("total", "over", total_line, odds.get("over_odds"), p_over))
        out.append(_candidate("total", "under", total_line, odds.get("under_odds"), 1 - p_over))
    return Pricing(
        model_margin, model_total, fair_margin, fair_total, p_home_win, out, model_p, market_p
    )


def settle(c: Candidate, margin: float, total: float) -> float:
    """Profit in units for a one-unit stake."""
    if c.market == "spread":
        diff = (margin if c.side == "home" else -margin) + (c.line or 0.0)
    elif c.market == "total":
        diff = (total - (c.line or 0.0)) if c.side == "over" else ((c.line or 0.0) - total)
    else:
        diff = margin if c.side == "home" else -margin
    if diff == 0:
        return 0.0
    return decimal_odds(c.price) - 1 if diff > 0 else -1.0


# --------------------------------------------------------------------------- backtest


def build_rows(
    games: list[dict[str, Any]], *, availability: str = "actual", line: str = "close"
) -> tuple[list[dict[str, Any]], League]:
    """Replay history in order, recording each game's pre-game view before applying it.

    ``availability="actual"`` tells the view who played, standing in for the injury
    report. ``"none"`` withholds it: every recent rotation player is assumed to play.
    ``line="open"`` grades against the opening number at a flat -110 instead of the close.
    """
    league = League()
    rows: list[dict[str, Any]] = []
    for game in games:
        home, away = game["home"], game["away"]
        known = availability == "actual"
        view: Matchup = league.matchup(
            home["id"], away["id"], game["date"],
            home_available=available_from_box(home) if known else None,
            away_available=available_from_box(away) if known else None,
            neutral=game.get("neutral", False),
        )  # fmt: skip
        odds = game.get("odds") or {}
        if line == "open":
            odds = {"spread": odds.get("open_spread"), "total": odds.get("open_total")}
        rows.append(
            {
                "id": game["id"], "date": game["date"], "season": game["season"],
                "home": home["abbr"], "away": away["abbr"], "features": view.features,
                "margin": home["score"] - away["score"], "total": home["score"] + away["score"],
                "spread": odds.get("spread"), "total_line": odds.get("total"), "odds": odds,
                "warm": min(league.teams[home["id"]].games_played, league.teams[away["id"]].games_played),
            }
        )  # fmt: skip
        league.apply(game)
    return rows, league


def walk_forward(rows: list[dict[str, Any]], *, min_ev: float = MIN_EV) -> dict[str, Any]:
    """Refit at the start of every month on everything before it, then bet that month."""
    by_month: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        day = game_day(row["date"])
        by_month[(day.year, day.month)].append(index)

    bets: list[dict[str, Any]] = []
    graded: list[dict[str, Any]] = []
    fits: list[Fit] = []
    evidence_m: list[tuple[float, float]] = []
    evidence_t: list[tuple[float, float]] = []
    for month in sorted(by_month):
        indices = by_month[month]
        train = [r for r in rows[: indices[0]] if r["warm"] >= 5]
        if len(train) < MIN_TRAIN:
            continue
        fitted = replace(
            fit(train),
            beta_margin=credited_slope(evidence_m)[2],
            beta_total=credited_slope(evidence_t)[2],
        )
        fits.append(fitted)
        for index in indices:
            row = rows[index]
            if row["spread"] is None and row["total_line"] is None:
                continue
            pricing = price(fitted, row["features"], row["odds"])
            graded.append(
                {
                    "row": row, "model_margin": pricing.model_margin, "model_total": pricing.model_total,
                    "p_home_win": pricing.p_home_win,
                }
            )  # fmt: skip
            if row["spread"] is not None:
                evidence_m.append(
                    (pricing.model_margin + row["spread"], row["margin"] + row["spread"])
                )
            if row["total_line"] is not None:
                evidence_t.append(
                    (pricing.model_total - row["total_line"], row["total"] - row["total_line"])
                )
            for pick in pricing.best(min_ev):
                bets.append(
                    {
                        "date": row["date"], "season": row["season"], "game": f"{row['away']} @ {row['home']}",
                        "market": pick.market, "side": pick.side, "line": pick.line, "price": pick.price,
                        "prob": pick.prob, "ev": pick.ev, "bucket": pick.bucket,
                        "profit": settle(pick, row["margin"], row["total"]),
                    }
                )  # fmt: skip
    return {
        "bets": bets, "graded": graded, "fits": fits,
        "edge_margin": credited_slope(evidence_m), "edge_total": credited_slope(evidence_t),
    }  # fmt: skip


def fit_live(rows: list[dict[str, Any]]) -> tuple[Fit, dict[str, Any]]:
    """The fit used for today's card: weights from all history, edge from the backtest."""
    result = walk_forward(rows)
    fitted = replace(
        fit([r for r in rows if r["warm"] >= 5]),
        beta_margin=result["edge_margin"][2],
        beta_total=result["edge_total"][2],
    )
    return fitted, result


def record(bets: list[dict[str, Any]]) -> dict[str, float]:
    wins = sum(1 for b in bets if b["profit"] > 0)
    losses = sum(1 for b in bets if b["profit"] < 0)
    profit = sum(b["profit"] for b in bets)
    n = len(bets)
    decided = wins + losses
    return {
        "bets": n, "wins": wins, "losses": losses, "pushes": n - decided,
        "hit_rate": wins / decided if decided else 0.0, "units": profit,
        "roi": profit / n if n else 0.0,
        # Standard error of ROI, so a reader can see how much of it could be luck.
        "roi_se": float(np.std([b["profit"] for b in bets]) / math.sqrt(n)) if n > 1 else 0.0,
    }  # fmt: skip


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    bets, graded = result["bets"], result["graded"]
    out: dict[str, Any] = {
        "overall": record(bets),
        "by_market": {},
        "by_season": {},
        "by_bucket": {},
    }
    for key, name in (("market", "by_market"), ("season", "by_season")):
        for value in sorted({b[key] for b in bets}, key=str):
            out[name][str(value)] = record([b for b in bets if b[key] == value])
    for market in ("spread", "total", "moneyline"):
        for bucket in range(len(EDGE_BUCKETS) - 1):
            chosen = [b for b in bets if b["market"] == market and b["bucket"] == bucket]
            if chosen:
                out["by_bucket"][f"{market}:{bucket}"] = record(chosen)

    with_spread = [g for g in graded if g["row"]["spread"] is not None]
    with_total = [g for g in graded if g["row"]["total_line"] is not None]
    if with_spread:
        margin = np.array([g["row"]["margin"] for g in with_spread])
        model = np.array([g["model_margin"] for g in with_spread])
        market = -np.array([g["row"]["spread"] for g in with_spread])
        out["margin"] = {
            "games": len(with_spread),
            "model_mae": float(np.abs(margin - model).mean()),
            "market_mae": float(np.abs(margin - market).mean()),
            "model_winner_accuracy": float(((model > 0) == (margin > 0)).mean()),
            "market_winner_accuracy": float(((market > 0) == (margin > 0))[market != 0].mean()),
            "model_market_correlation": float(np.corrcoef(model, market)[0, 1]),
            "ats_by_disagreement": _ats(model - market, margin - market),
        }
    if with_total:
        total = np.array([g["row"]["total"] for g in with_total])
        model = np.array([g["model_total"] for g in with_total])
        market = np.array([g["row"]["total_line"] for g in with_total])
        out["total"] = {
            "games": len(with_total),
            "model_mae": float(np.abs(total - model).mean()),
            "market_mae": float(np.abs(total - market).mean()),
            "ou_by_disagreement": _ats(model - market, total - market),
        }
    fits: list[Fit] = result["fits"]
    if fits:
        names = ("slope", "standard_error", "credited")
        out["calibration"] = {
            "edge_vs_line_margin": dict(zip(names, result["edge_margin"], strict=True)),
            "edge_vs_line_total": dict(zip(names, result["edge_total"], strict=True)),
            "sigma_margin": fits[-1].sigma_margin,
            "sigma_total": fits[-1].sigma_total,
            "weights_margin": dict(
                zip(MARGIN_FEATURES, map(float, fits[-1].w_margin), strict=True)
            ),
            "weights_total": dict(
                zip((*TOTAL_FEATURES, "intercept"), map(float, fits[-1].w_total), strict=True)
            ),
        }
    return out


def _ats(disagreement: np.ndarray, residual: np.ndarray) -> dict[str, dict[str, float]]:
    """How often siding with the model against the line would have won, by gap size."""
    table = {}
    for low, high in ((0, 1.5), (1.5, 3), (3, 5), (5, 99)):
        mask = (np.abs(disagreement) >= low) & (np.abs(disagreement) < high) & (residual != 0)
        if mask.sum():
            won = np.sign(disagreement[mask]) == np.sign(residual[mask])
            table[f"{low}-{high}"] = {"games": int(mask.sum()), "hit_rate": float(won.mean())}
    return table
