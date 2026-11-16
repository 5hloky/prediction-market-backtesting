"""Player props: points, rebounds, assists, threes, steals, blocks and combinations.

For every player in a matchup view the bot has an expected box-score line (see
``features._player_line``). This module turns those expectations into "more or
less than the line" probabilities, and grades them the same way game bets are
graded: walk-forward, against real sportsbook prop lines, crediting the bot only
with as much disagreement with the book as has held up out of sample.

Backtest lines are each prop's *opening* number, because the stored "current"
number of a finished game was updated in play. Opening props are softer than
the numbers available near tip-off, so treat the record as an upper bound.
Only props posted with both a "more" and a "less" price are graded.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from nba_bets import model
from nba_bets.features import League, PlayerView, available_from_box, game_day

MARKETS: dict[str, tuple[str, ...]] = {
    "pts": ("pts",),
    "reb": ("reb",),
    "ast": ("ast",),
    "fg3m": ("fg3m",),
    "stl": ("stl",),
    "blk": ("blk",),
    "pra": ("pts", "reb", "ast"),
    "pr": ("pts", "reb"),
    "pa": ("pts", "ast"),
    "ra": ("reb", "ast"),
}
LABELS = {
    "pts": "Points", "reb": "Rebounds", "ast": "Assists", "fg3m": "3-pointers", "stl": "Steals",
    "blk": "Blocks", "pra": "Pts+Reb+Ast", "pr": "Pts+Reb", "pa": "Pts+Ast", "ra": "Reb+Ast",
}  # fmt: skip
ASSUMED_HOLD = 0.07  # two-way margin assumed when a book shows only the "more" price
MIN_PROJECTED_MINUTES = 8.0
RECENT_GAMES = 2
RECENT_EVIDENCE = 1500  # graded props per market that the credited edge is measured on
GAP_BUCKETS = ((0.0, 0.05), (0.05, 0.10), (0.10, 1.0))


def projected(p: PlayerView) -> dict[str, float]:
    return {market: sum(p.proj[part] for part in parts) for market, parts in MARKETS.items()}


def actual(row: dict[str, Any]) -> dict[str, float]:
    base = {
        "pts": row["pts"], "reb": row["orb"] + row["drb"], "ast": row["ast"],
        "fg3m": row["tpm"], "stl": row["stl"], "blk": row["blk"],
    }  # fmt: skip
    return {market: sum(base[part] for part in parts) for market, parts in MARKETS.items()}


def market_prob(over: float, under: float | None) -> float:
    """The book's no-vig probability of "more"."""
    if under is not None:
        return model.devig(over, under)[0]
    return min(max(model.implied(over) - ASSUMED_HOLD / 2, 0.02), 0.98)


def under_price(over: float, under: float | None) -> float:
    if under is not None:
        return under
    p = min(max(1 + ASSUMED_HOLD - model.implied(over), 0.05), 0.95)
    return -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p


# --------------------------------------------------------------------------- distribution


@dataclass
class Calibration:
    """Per market: a bias correction and how variance grows with the mean."""

    scale: float = 1.0
    linear: float = 1.2
    quadratic: float = 0.08

    def prob_over(self, mean: float, line: float) -> float:
        """P(stat > line) under a negative binomial with this market's variance."""
        m = max(mean * self.scale, 1e-3)
        variance = max(self.linear * m + self.quadratic * m * m, m * 1.02)
        r = m * m / (variance - m)
        p = r / (r + m)
        pmf = p**r
        cdf = pmf
        for k in range(math.floor(line)):
            pmf *= (k + r) / (k + 1) * (1 - p)
            cdf += pmf
        return min(max(1 - cdf, 0.0), 1.0)


def calibrate(records: list[dict[str, Any]]) -> dict[str, Calibration]:
    out: dict[str, Calibration] = {}
    for market in MARKETS:
        mu = np.array([r["proj"][market] for r in records])
        y = np.array([r["actual"][market] for r in records])
        if len(mu) < 500 or mu.sum() <= 0:
            out[market] = Calibration()
            continue
        scale = float(y.sum() / mu.sum())
        m = mu * scale
        # resid^2 ~ linear * m + quadratic * m^2, weighted so big means do not dominate.
        x = np.stack([m, m * m], axis=1) / m[:, None]
        coef, *_ = np.linalg.lstsq(x, (y - m) ** 2 / m, rcond=None)
        out[market] = Calibration(scale, float(max(coef[0], 1.02)), float(max(coef[1], 0.0)))
    return out


# --------------------------------------------------------------------------- history


def build_records(
    games: list[dict[str, Any]],
    lines: dict[str, dict[str, Any]],
    *,
    availability: str = "none",
) -> tuple[list[dict[str, Any]], League]:
    """One record per player-game: pre-game projection, what happened, and the line.

    The default withholds who played (``availability="none"``): opening prop lines
    are posted before late injury news, so giving the projection the final
    inactive list would be hindsight.
    """
    league = League()
    records: list[dict[str, Any]] = []
    for game in games:

        def who(side: dict[str, Any]) -> list[dict[str, Any]] | None:
            if availability == "actual":
                return available_from_box(side)
            if availability == "recent":
                return recent_rotation(league, side["id"])
            return None

        view = league.matchup(
            game["home"]["id"],
            game["away"]["id"],
            game["date"],
            home_available=who(game["home"]),
            away_available=who(game["away"]),
            neutral=game.get("neutral", False),
            confirmed=availability == "actual",
        )
        game_lines = lines.get(game["id"]) or {}
        for side, team_view in ((game["home"], view.home), (game["away"], view.away)):
            box = {row["id"]: row for row in side["players"] if row["min"] > 0}
            warm = league.teams[side["id"]].games_played >= 5
            for p in team_view.players:
                if p.id not in box or p.minutes < MIN_PROJECTED_MINUTES or not warm:
                    continue
                records.append(
                    {
                        "game": game["id"],
                        "date": game["date"],
                        "season": game["season"],
                        "player": p.name,
                        "proj": projected(p),
                        "actual": actual(box[p.id]),
                        "lines": game_lines.get(p.id) or {},
                    }
                )
        league.apply(game)
    return records, league


def recent_rotation(
    league: League, team_id: str, window: int = RECENT_GAMES
) -> list[dict[str, Any]] | None:
    """Players who appeared in the team's last few games: a no-hindsight guess at who is healthy."""
    team = league.teams.get(team_id)
    if team is None or team.games_played < window:
        return None
    players = [p for p in league.rotation(team_id) if team.games_played - p.last_team_game < window]
    if len(players) < 7:
        return None
    return [{"id": p.id, "name": p.name, "pos": p.pos} for p in players]


def credited(pairs: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Share of the bot's disagreement with the book's probability that proved real.

    ``pairs`` are out-of-sample (bot probability - book probability, result - book
    probability). Only the most recent results count, because the edge against
    prop lines has been shrinking year over year. Returns (slope, standard error,
    slope credited after one standard error of caution, floored at zero and capped
    at one).
    """
    pairs = pairs[-RECENT_EVIDENCE:]
    if len(pairs) < model.MIN_EVIDENCE:
        return 0.0, 0.0, 0.0
    d, r = np.array(pairs).T
    denom = float(d @ d)
    if denom <= 0:
        return 0.0, 0.0, 0.0
    slope = float(d @ r / denom)
    se = float(np.sqrt(np.mean((r - slope * d) ** 2) / denom))
    return slope, se, float(np.clip(slope - se, 0.0, 1.0))


def price(
    mean: float, line: list[float | None], calibration: Calibration, belief: float
) -> dict[str, Any]:
    """Model view and the best side of one prop. ``line`` is [line, more price, less price]."""
    number, over, under = line
    q = market_prob(over, under)
    p_model = calibration.prob_over(mean, number)
    p_used = q + belief * (p_model - q)
    more = p_model >= q
    side_price = over if more else under_price(over, under)
    prob = p_used if more else 1 - p_used
    return {
        "line": number,
        "side": "more" if more else "less",
        "price": side_price,
        "two_way": under is not None,
        "market_prob": q,
        "model_prob": p_model,
        "prob": prob,
        "ev": prob * (model.decimal_odds(side_price) - 1) - (1 - prob),
        "gap": abs(p_model - q),
    }


def walk_forward(records: list[dict[str, Any]], *, min_ev: float = model.MIN_EV) -> dict[str, Any]:
    by_month: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        day = game_day(record["date"])
        by_month[(day.year, day.month)].append(index)
    evidence: dict[str, list[tuple[float, float]]] = {m: [] for m in MARKETS}
    graded: list[dict[str, Any]] = []
    for month in sorted(by_month):
        indices = by_month[month]
        if indices[0] < 3000 or not any(records[i]["lines"] for i in indices):
            continue
        calibration = calibrate(records[: indices[0]])
        belief = {m: credited(evidence[m])[2] for m in MARKETS}
        for index in indices:
            record = records[index]
            for market, line in record["lines"].items():
                if market not in MARKETS:
                    continue
                result = record["actual"][market]
                if result == line[0]:
                    continue
                view = price(record["proj"][market], line, calibration[market], belief[market])
                won_over = float(result > line[0])
                if not view["two_way"]:
                    # Only a "more" price was posted. The book's true probability cannot be
                    # recovered from one side, so these lines are never evidence and never bets.
                    continue
                evidence[market].append(
                    (view["model_prob"] - view["market_prob"], won_over - view["market_prob"])
                )
                won = won_over == (view["side"] == "more")
                graded.append(
                    {
                        **view,
                        "market": market,
                        "season": record["season"],
                        "won": won,
                        "bet": view["ev"] >= min_ev,
                        "profit": model.decimal_odds(view["price"]) - 1 if won else -1.0,
                        "abs_err_model": abs(
                            result - record["proj"][market] * calibration[market].scale
                        ),
                        "abs_err_line": abs(result - line[0]),
                        "over_won": won_over,
                    }
                )
    return {"graded": graded, "edge": {m: credited(evidence[m]) for m in MARKETS}}


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for market in MARKETS:
        rows = [g for g in result["graded"] if g["market"] == market]
        if not rows:
            continue
        slope, se, cred = result["edge"][market]
        buckets = {}
        for low, high in GAP_BUCKETS:
            chosen = [g for g in rows if low <= g["gap"] < high]
            if chosen:
                buckets[f"{low:.2f}-{high:.2f}"] = {
                    "props": len(chosen),
                    "hit_rate": float(np.mean([g["won"] for g in chosen])),
                    "book_expected": float(
                        np.mean(
                            [
                                g["market_prob"] if g["side"] == "more" else 1 - g["market_prob"]
                                for g in chosen
                            ]
                        )
                    ),
                }
        out[market] = {
            "props": len(rows),
            "model_mae": float(np.mean([g["abs_err_model"] for g in rows])),
            "line_mae": float(np.mean([g["abs_err_line"] for g in rows])),
            "hit_rate": float(np.mean([g["won"] for g in rows])),
            "by_gap": buckets,
            "edge": {"slope": slope, "standard_error": se, "credited": cred},
            "bets": model.record([g for g in rows if g["bet"]]),
        }
    bets = [g for g in result["graded"] if g["bet"]]
    out["all_bets"] = model.record(bets)
    out["bets_by_season"] = {
        str(season): model.record([g for g in bets if g["season"] == season])
        for season in sorted({g["season"] for g in bets})
    }
    return out


def fit_live(
    records: list[dict[str, Any]],
) -> tuple[dict[str, Calibration], dict[str, float], dict[str, Any]]:
    """Calibration from all history, plus the belief the backtest has earned per market."""
    result = walk_forward(records)
    return calibrate(records), {m: result["edge"][m][2] for m in MARKETS}, summarize(result)
