from __future__ import annotations

import copy

import numpy as np
import pytest

from nba_bets import espn, model, props, sim
from nba_bets.features import League, available_from_box, points_per_possession


def _player(pid: str, pos: str, minutes: float, pts: float, starter: bool = True) -> dict:
    """A box-score line whose volume scales with minutes and whose efficiency rises with scoring."""
    share = minutes / 34
    ftm, tpm = round(pts * 0.15), round(pts * 0.1)
    fgm = (pts - ftm - 3 * tpm) / 2 + tpm
    return {
        "id": pid, "name": f"P{pid}", "pos": pos, "starter": starter, "min": minutes, "pts": pts,
        "fgm": fgm, "fga": fgm / (0.40 + pts / 200), "tpm": tpm, "tpa": tpm * 2.8, "ftm": ftm,
        "fta": ftm * 1.25, "orb": share, "drb": 4 * share, "ast": 4 * share, "tov": 1.5 * share,
        "stl": share, "blk": 0.5 * share, "pf": 2 * share,
    }  # fmt: skip


def _side(team: str, score: float, strength: float) -> dict:
    players = [
        _player(f"{team}{i}", "GGFFCGFC"[i], 34 - 3 * i, (22 - 2 * i) * strength, i < 5)
        for i in range(8)
    ]
    return {
        "id": team, "abbr": team, "name": f"Team {team}", "score": score, "periods": 4,
        "fgm": 41, "fga": 88, "tpm": 13, "tpa": 36, "ftm": 17, "fta": 22, "orb": 10, "drb": 33,
        "ast": 25, "tov": 13, "players": players,
    }  # fmt: skip


def _game(index: int, home: str, away: str, home_score: float, away_score: float) -> dict:
    return {
        "id": str(index), "date": f"2025-11-{1 + index // 4:02d}T00:00Z", "season": 2026, "stype": 2,
        "neutral": False, "home": _side(home, home_score, 1.1), "away": _side(away, away_score, 0.9),
        "odds": {"spread": -4.5, "total": 221.5, "ml_home": -180.0, "ml_away": 150.0},
    }  # fmt: skip


def _season(n: int = 60) -> list[dict]:
    rng = np.random.default_rng(0)
    teams = ["A", "B", "C", "D"]
    games = []
    for i in range(n):
        home, away = teams[i % 4], teams[(i + 1 + i // 4) % 4]
        if home == away:
            away = teams[(i + 2) % 4]
        games.append(_game(i, home, away, 112 + rng.integers(-9, 10), 108 + rng.integers(-9, 10)))
    return games


def test_american_odds_math() -> None:
    assert model.decimal_odds(-110) == pytest.approx(1.9091, abs=1e-4)
    assert model.decimal_odds(150) == pytest.approx(2.5)
    home, away = model.devig(-110, -110)
    assert home == pytest.approx(0.5) and away == pytest.approx(0.5)


def test_settle_spread_total_and_moneyline() -> None:
    home_minus_4 = model.Candidate("spread", "home", -4.5, -110, 0.5, 0.0)
    assert model.settle(home_minus_4, margin=5, total=200) == pytest.approx(100 / 110)
    assert model.settle(home_minus_4, margin=4, total=200) == -1.0
    push = model.Candidate("spread", "away", 4.0, -110, 0.5, 0.0)
    assert model.settle(push, margin=4, total=200) == 0.0
    under = model.Candidate("total", "under", 221.5, -105, 0.5, 0.0)
    assert model.settle(under, margin=0, total=220) == pytest.approx(100 / 105)
    dog = model.Candidate("moneyline", "away", None, 150, 0.4, 0.0)
    assert model.settle(dog, margin=-3, total=200) == pytest.approx(1.5)


def test_pregame_view_cannot_see_the_game_it_describes() -> None:
    games = _season()
    league = League()
    for game in games[:-1]:
        league.apply(game)
    last = games[-1]
    before = league.matchup(last["home"]["id"], last["away"]["id"], last["date"]).features
    blowout = copy.deepcopy(last)
    blowout["home"]["score"], blowout["away"]["score"] = 160.0, 80.0
    rows, _ = model.build_rows(games[:-1] + [blowout], availability="none")
    assert rows[-1]["features"] == pytest.approx(before)


def test_missing_star_lowers_lineup_value() -> None:
    games = _season()
    league = League()
    for game in games:
        league.apply(game)
    full = available_from_box(games[-1]["home"])
    home, away = games[-1]["home"]["id"], games[-1]["away"]["id"]
    healthy = league.matchup(home, away, "2025-12-01T00:00Z", home_available=full)
    short = league.matchup(home, away, "2025-12-01T00:00Z", home_available=full[1:])
    assert short.home.value_delta < healthy.home.value_delta
    assert [name for name, _ in short.home.out] == [full[0]["name"]]
    assert sum(p.minutes for p in short.home.players) == pytest.approx(240.0)
    assert max(p.minutes for p in short.home.players) <= 40.0 + 1e-6


def test_simulation_mean_matches_closed_form() -> None:
    games = _season()
    league = League()
    for game in games:
        league.apply(game)
    view = league.matchup(games[-1]["home"]["id"], games[-1]["away"]["id"], "2025-12-01T00:00Z")
    result = sim.simulate(view, n=20000, seed=1)
    # Overtime adds a little to the simulated total, so compare regulation-scale means loosely.
    assert result.home.mean() == pytest.approx(view.possessions * view.home.ppp, rel=0.02)
    assert result.away.mean() == pytest.approx(view.possessions * view.away.ppp, rel=0.02)
    assert result.home_players.sum(axis=1).mean() <= result.home.mean()
    assert points_per_possession(view.home.rates) > 0.9


def test_simulation_centres_on_requested_score() -> None:
    games = _season()
    league = League()
    for game in games:
        league.apply(game)
    view = league.matchup(games[-1]["home"]["id"], games[-1]["away"]["id"], "2025-12-01T00:00Z")
    result = sim.simulate(view, n=20000, seed=2, target_home=120.0, target_away=104.0)
    assert result.home.mean() == pytest.approx(120.0, rel=0.02)
    assert result.away.mean() == pytest.approx(104.0, rel=0.02)


def test_no_edge_is_credited_without_evidence() -> None:
    assert model.credited_slope([(3.0, 5.0)] * 10) == (0.0, 0.0, 0.0)
    rng = np.random.default_rng(3)
    noise = [
        (float(d), float(r))
        for d, r in zip(rng.normal(0, 3, 2000), rng.normal(0, 13, 2000), strict=True)
    ]
    assert model.credited_slope(noise)[2] == pytest.approx(0.0, abs=0.004)
    signal = [(d, d + r) for d, r in noise]
    assert model.credited_slope(signal)[2] > 0.01


def test_zero_edge_means_no_bets() -> None:
    fitted = model.Fit(
        np.zeros(len(model.MARGIN_FEATURES)), np.zeros(len(model.TOTAL_FEATURES) + 1),
        0.0, 0.0, 13.5, 18.0, 1000,
    )  # fmt: skip
    features = dict.fromkeys((*model.MARGIN_FEATURES, *model.TOTAL_FEATURES), 0.0)
    odds = {"spread": -9.5, "total": 230.5, "ml_home": -400.0, "ml_away": 320.0}
    pricing = model.price(fitted, features, odds)
    assert pricing.best() == []
    assert all(c.ev < 0 for c in pricing.candidates)


def test_core_odds_ignores_live_books_and_reads_home_line() -> None:
    payload = {
        "items": [
            {"provider": {"name": "ESPN Bet - Live Odds"}, "spread": 15.5, "overUnder": 250.5},
            {
                "provider": {"name": "ESPN BET"}, "spread": -4.5, "overUnder": 228.5,
                "homeTeamOdds": {"moneyLine": -180, "spreadOdds": -110},
                "awayTeamOdds": {"moneyLine": 150, "spreadOdds": -110},
            },
        ]
    }  # fmt: skip
    odds = espn.parse_core_odds(payload)
    assert odds is not None
    assert (odds["book"], odds["spread"], odds["total"], odds["ml_home"]) == (
        "ESPN BET",
        -4.5,
        228.5,
        -180.0,
    )
    assert espn.parse_core_odds({"items": [payload["items"][0]]}) is None


def test_prop_parser_reads_pairs_and_ladders() -> None:
    def item(kind: str, athlete: str, line: str, price: str) -> dict:
        return {
            "type": {"name": kind},
            "athlete": {"$ref": f"http://x/athletes/{athlete}?lang=en"},
            "odds": {
                "total": {"value": "99.5", "open": line},
                "american": {"value": "+900", "open": price},
            },
        }

    items = [
        item("Total Points", "1", "21.5", "-112"),
        item("Total Points", "1", "21.5", "-108"),
        item("Steals Milestones", "1", "1+", "-160"),
        item("Steals Milestones", "1", "2+", "+250"),
        item("Total Rebounds", "2", "4.5", "-300"),
        item("Total Rebounds", "2", "6.5", "+105"),
        item("Total Rebounds", "2", "8.5", "+320"),
    ]
    parsed = espn.parse_props(items, opening=True)
    assert parsed["1"]["pts"] == [21.5, -112.0, -108.0]
    assert parsed["1"]["stl"] == [0.5, -160.0, None]
    assert parsed["2"]["reb"] == [6.5, 105.0, None]


def test_prop_probability_is_a_sane_distribution() -> None:
    calibration = props.Calibration(scale=1.0, linear=1.5, quadratic=0.02)
    probs = [calibration.prob_over(6.0, line) for line in (2.5, 5.5, 8.5, 12.5)]
    assert probs == sorted(probs, reverse=True)
    assert 0.4 < probs[1] < 0.6
    assert calibration.prob_over(6.0, 60.5) == pytest.approx(0.0, abs=1e-6)
    assert calibration.prob_over(9.0, 5.5) > calibration.prob_over(6.0, 5.5)


def test_prop_without_credited_edge_is_never_positive_value() -> None:
    calibration = props.Calibration()
    view = props.price(9.0, [5.5, -115.0, -105.0], calibration, belief=0.0)
    assert view["side"] == "more" and view["two_way"]
    assert view["prob"] == pytest.approx(view["market_prob"])
    assert view["ev"] < 0
    trusted = props.price(9.0, [5.5, -115.0, -105.0], calibration, belief=1.0)
    assert trusted["ev"] > 0
    one_sided = props.price(9.0, [5.5, -115.0, None], calibration, belief=1.0)
    assert not one_sided["two_way"]


def test_prop_records_match_box_scores_and_stay_pregame() -> None:
    games = _season()
    records, _ = props.build_records(games, {}, availability="actual")
    assert records
    last = records[-1]
    row = next(p for p in games[-1]["away"]["players"] if f"P{p['id']}" == last["player"])
    assert last["actual"]["pra"] == pytest.approx(row["pts"] + row["orb"] + row["drb"] + row["ast"])
    assert last["proj"]["pra"] == pytest.approx(
        last["proj"]["pts"] + last["proj"]["reb"] + last["proj"]["ast"]
    )
    changed = [dict(g) for g in games]
    changed[-1] = _game(len(games) - 1, games[-1]["home"]["id"], games[-1]["away"]["id"], 160, 80)
    for side in ("home", "away"):
        for p in changed[-1][side]["players"]:
            p["pts"] *= 3
    again, _ = props.build_records(changed, {}, availability="actual")
    assert again[-1]["proj"] == pytest.approx(last["proj"])
