"""The daily card: scan a slate, read the injury report, analyse each game, pick.

A pick is labelled ``BET`` only when the walk-forward backtest credits the bot with
an edge over the line and the expected value at the offered price clears the
threshold. Otherwise the game still gets the bot's best ``LEAN`` (the market where
it disagrees with the line most) together with how often leans of that size have
actually won, so the reader is never left guessing how much weight to put on it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np

from nba_bets import espn, model, sim, store
from nba_bets.features import League, Matchup, TeamView, game_day

OUT_STATUSES = {"out", "suspension", "injured reserve"}
PICKS_DIR = store.DATA_DIR / "picks"
GAP_BUCKETS = ((0, 1.5), (1.5, 3), (3, 5), (5, 99))


@dataclass
class State:
    league: League
    fitted: model.Fit
    summary: dict[str, Any]
    games: int
    through: str


@dataclass
class GameCard:
    event_id: str
    tip: str
    matchup: Matchup
    odds: dict[str, Any] | None
    pricing: model.Pricing
    sim: sim.SimResult
    injuries: dict[str, list[dict[str, str]]]  # team id -> entries
    pick: dict[str, Any] = field(default_factory=dict)


def us_today() -> date:
    return (datetime.now(UTC) - timedelta(hours=6)).date()


def load_state(*, refresh: bool = True) -> State:
    if refresh:
        store.sync(verbose=False)
    games = store.load_games()
    if not games:
        raise SystemExit("no game history cached; run `python -m nba_bets sync` first")
    rows, league = model.build_rows(games)
    fitted, result = model.fit_live(rows)
    return State(league, fitted, model.summarize(result), len(games), games[-1]["date"])


# --------------------------------------------------------------------------- inputs


def injury_report() -> dict[str, list[dict[str, str]]]:
    """Team id -> [{id, name, status, detail, note}] from the league-wide report."""
    report: dict[str, list[dict[str, str]]] = {}
    for team in espn.injuries().get("injuries") or []:
        entries = []
        for item in team.get("injuries") or []:
            athlete = item.get("athlete") or {}
            athlete_id = str(athlete.get("id") or "")
            if not athlete_id:
                for link in athlete.get("links") or []:
                    found = re.search(r"/id/(\d+)", link.get("href") or "")
                    if found:
                        athlete_id = found.group(1)
                        break
            details = item.get("details") or {}
            entries.append(
                {
                    "id": athlete_id,
                    "name": athlete.get("displayName") or "?",
                    "status": item.get("status") or "",
                    "detail": " ".join(
                        x for x in (details.get("type"), details.get("detail")) if x
                    ),
                    "note": item.get("shortComment") or "",
                }
            )
        report[str(team.get("id"))] = entries
    return report


def _available(
    league: League, team_id: str, injuries: list[dict[str, str]]
) -> list[dict[str, Any]] | None:
    """Recent rotation, minus anyone ruled out, limited to the current roster."""
    out_ids = {i["id"] for i in injuries if i["status"].lower() in OUT_STATUSES}
    out_names = {i["name"] for i in injuries if i["status"].lower() in OUT_STATUSES}
    try:
        roster = {str(a.get("id")) for a in espn.roster(team_id).get("athletes") or []}
    except espn.EspnError:
        roster = set()
    players = [
        p
        for p in league.rotation(team_id)
        if p.id not in out_ids and p.name not in out_names and (not roster or p.id in roster)
    ]
    if len(players) < 7:
        return None
    return [{"id": p.id, "name": p.name, "pos": p.pos} for p in players]


def slate(day: date) -> list[dict[str, Any]]:
    events = []
    for event in espn.scoreboard(day.strftime("%Y%m%d")).get("events") or []:
        competition = (event.get("competitions") or [{}])[0]
        sides = {c.get("homeAway"): c for c in competition.get("competitors") or []}
        if set(sides) != {"home", "away"}:
            continue
        odds = espn.parse_scoreboard_odds(competition)
        state = ((event.get("status") or {}).get("type") or {}).get("state")
        if odds is None and state == "pre":
            try:
                odds = espn.parse_core_odds(espn.core_odds(str(event["id"])))
            except espn.EspnError:
                odds = None
        events.append(
            {
                "id": str(event["id"]),
                "date": event.get("date"),
                "state": state,
                "neutral": bool(competition.get("neutralSite")),
                "home": sides["home"].get("team") or {},
                "away": sides["away"].get("team") or {},
                "odds": odds,
            }
        )
    return events


# --------------------------------------------------------------------------- analysis


def _gap_record(summary: dict[str, Any], market: str, gap: float) -> dict[str, float] | None:
    key = "ats_by_disagreement" if market == "spread" else "ou_by_disagreement"
    table = (summary.get("margin" if market == "spread" else "total") or {}).get(key) or {}
    for low, high in GAP_BUCKETS:
        if low <= abs(gap) < high:
            return table.get(f"{low}-{high}")
    return None


def _choose(state: State, card: GameCard) -> dict[str, Any]:
    home, away = card.matchup.home, card.matchup.away
    pricing, odds = card.pricing, card.odds or {}

    def label(c: model.Candidate) -> str:
        if c.market == "total":
            return f"{c.side.capitalize()} {c.line:g}"
        team = home.abbr if c.side == "home" else away.abbr
        return f"{team} moneyline" if c.market == "moneyline" else f"{team} {c.line:+g}"

    bets = pricing.best()
    if bets:
        top = bets[0]
        return {
            "grade": "BET", "market": top.market, "side": top.side, "line": top.line,
            "price": top.price, "label": label(top), "prob": top.prob, "ev": top.ev,
            "why": f"model win probability {top.prob:.1%} at {top.price:+.0f}, expected value {top.ev:+.1%} per unit",
        }  # fmt: skip
    options = []
    if odds.get("spread") is not None:
        gap = pricing.model_margin + odds["spread"]
        options.append(
            ("spread", "home" if gap > 0 else "away", gap, gap / state.fitted.sigma_margin)
        )
    if odds.get("total") is not None:
        gap = pricing.model_total - odds["total"]
        options.append(
            ("total", "over" if gap > 0 else "under", gap, gap / state.fitted.sigma_total)
        )
    if not options:
        side = "home" if pricing.model_margin > 0 else "away"
        team = home.abbr if side == "home" else away.abbr
        return {
            "grade": "NO LINE", "market": "moneyline", "side": side, "line": None, "price": None,
            "label": f"{team} to win", "prob": max(pricing.p_home_win, 1 - pricing.p_home_win), "ev": None,
            "why": "no sportsbook line posted yet, so this is the projected winner only",
        }  # fmt: skip
    market, side, gap, _ = max(options, key=lambda o: abs(o[3]))
    candidate = next(c for c in pricing.candidates if c.market == market and c.side == side)
    history = _gap_record(state.summary, market, gap)
    past = (
        f"leans this size have won {history['hit_rate']:.1%} of {history['games']} back-tested games"
        if history
        else "no back-test sample for a gap this size"
    )
    return {
        "grade": "LEAN", "market": market, "side": side, "line": candidate.line,
        "price": candidate.price, "label": label(candidate), "prob": candidate.prob, "ev": candidate.ev,
        "gap": gap, "history": history,
        "why": f"model differs from the line by {abs(gap):.1f} points; {past} (52.4% is break-even at -110)",
    }  # fmt: skip


def analyse(
    state: State, event: dict[str, Any], injuries: dict[str, list[dict[str, str]]]
) -> GameCard:
    home_id, away_id = str(event["home"].get("id")), str(event["away"].get("id"))
    league = state.league
    for team_id, info in ((home_id, event["home"]), (away_id, event["away"])):
        team = league._team(team_id)
        team.abbr = team.abbr or info.get("abbreviation") or team_id
        team.name = team.name or info.get("displayName") or team.abbr
    view = league.matchup(
        home_id, away_id, event["date"],
        home_available=_available(league, home_id, injuries.get(home_id, [])),
        away_available=_available(league, away_id, injuries.get(away_id, [])),
        neutral=event.get("neutral", False),
    )  # fmt: skip
    pricing = model.price(state.fitted, view.features, event.get("odds"))
    result = sim.simulate(
        view,
        target_home=(pricing.model_total + pricing.model_margin) / 2,
        target_away=(pricing.model_total - pricing.model_margin) / 2,
    )
    card = GameCard(
        event["id"], event["date"], view, event.get("odds"), pricing, result,
        {home_id: injuries.get(home_id, []), away_id: injuries.get(away_id, [])},
    )  # fmt: skip
    card.pick = _choose(state, card)
    return card


def cards_for(state: State, day: date) -> list[GameCard]:
    injuries = injury_report()
    return [analyse(state, e, injuries) for e in slate(day) if e["state"] == "pre"]


def find_game(state: State, query: str, day: date, lookahead: int = 4) -> GameCard | None:
    """Match a request such as ``LAL@PHX``, ``lakers suns`` or ``celtics``."""
    tokens = [t for t in re.split(r"[^a-z0-9]+", query.lower()) if t and t not in ("vs", "at", "v")]
    injuries: dict[str, list[dict[str, str]]] | None = None
    for offset in range(lookahead):
        for event in slate(day + timedelta(days=offset)):
            if event["state"] != "pre":
                continue
            names = [
                " ".join(
                    str(team.get(k) or "").lower()
                    for k in ("abbreviation", "displayName", "shortDisplayName", "location")
                )
                for team in (event["home"], event["away"])
            ]
            hits = [any(_matches(t, n) for n in names) for t in tokens]
            both = sum(any(_matches(t, n) for t in tokens) for n in names)
            if tokens and all(hits) and (both == 2 or len(tokens) == 1):
                injuries = injuries or injury_report()
                return analyse(state, event, injuries)
    return None


def _matches(token: str, names: str) -> bool:
    words = names.split()
    return token in words or (len(token) >= 4 and any(w.startswith(token) for w in words))


# --------------------------------------------------------------------------- rendering


def _record(team_log: list[dict[str, Any]]) -> str:
    wins = sum(1 for g in team_log if g["pts"] > g["opp_pts"])
    ats = [g["pts"] - g["opp_pts"] + g["spread"] for g in team_log if g["spread"] is not None]
    ou = [g["pts"] + g["opp_pts"] - g["total"] for g in team_log if g["total"] is not None]
    last = team_log[-10:]
    return (
        f"{wins}-{len(team_log) - wins}, "
        f"{sum(1 for x in ats if x > 0)}-{sum(1 for x in ats if x < 0)} against the spread, "
        f"overs {sum(1 for x in ou if x > 0)}-{sum(1 for x in ou if x < 0)}, "
        f"last {len(last)}: {sum(1 for g in last if g['pts'] > g['opp_pts'])}-"
        f"{sum(1 for g in last if g['pts'] <= g['opp_pts'])}"
    )


def _team_block(
    state: State, view: TeamView, opp: TeamView, points: np.ndarray, injuries: list[dict[str, str]]
) -> list[str]:
    team = state.league.teams[view.id]
    r = view.rates
    lines = [
        f"**{view.name}** ({_record(team.log) if team.log else 'no games yet this season'})",
        (
            f"- Ratings per 100 possessions vs league: offence {team.off_rating:+.1f}, defence {team.def_rating:+.1f}; "
            f"pace {team.pace:.1f}; rest {view.rest_days if view.rest_days < 5 else '4+'} day(s), "
            f"{view.games_last_7} game(s) in the last week"
        ),
        (
            f"- Expected possession vs this defence: turnover {r['p_tov']:.1%}, shots {r['s3']:.0%} from three "
            f"({r['p3']:.1%}) / twos at {r['p2']:.1%}, offensive rebound on {r['orb']:.1%} of misses "
            f"→ {view.ppp:.3f} points per possession"
        ),
    ]
    out = [i for i in injuries if i["status"].lower() in OUT_STATUSES]
    doubt = [i for i in injuries if i["status"].lower() not in OUT_STATUSES]
    usual = dict(view.out)
    if out:
        named = ", ".join(
            f"{i['name']} ({i['detail'] or i['status']}"
            + (f", usually {usual[i['name']]:.0f} min" if i["name"] in usual else "")
            + ")"
            for i in out
        )
        lines.append(f"- Out: {named}")
    if doubt:
        lines.append(
            "- Questionable (treated as playing): "
            + ", ".join(f"{i['name']} ({i['detail'] or i['status']})" for i in doubt)
        )
    lines.append(
        f"- Lineup value vs the usual rotation: {view.value_delta:+.1f} box-score points"
        + ("" if out or doubt else " (nobody on the injury report)")
    )
    order = np.argsort([-p.usage for p in view.players])[:4]
    for i in order:
        p = view.players[i]
        low, high = np.percentile(points[:, i], [10, 90])
        note = f"{opp.abbr} allow {p.matchup - 1:+.0%} to {p.group}s"
        if abs(p.history) >= 0.03:
            note += f", own history vs {opp.abbr} {p.history:+.0%}"
        lines.append(
            f"- {p.name} ({p.pos}, {p.minutes:.0f} min, {p.usage:.0%} of plays): "
            f"{points[:, i].mean():.1f} pts projected (10-90%: {low:.0f}-{high:.0f}); {note}"
        )
    return lines


def render(state: State, card: GameCard) -> str:
    m, pricing, odds = card.matchup, card.pricing, card.odds or {}
    home, away = m.home, m.away
    tip = datetime.fromisoformat(card.tip).strftime("%a %b %d, %H:%M UTC")
    home_pts = (pricing.model_total + pricing.model_margin) / 2
    away_pts = (pricing.model_total - pricing.model_margin) / 2
    lines = [f"## {away.name} @ {home.name} — {tip}", ""]
    # Independent possessions scatter more than real games do (teams ease off with a lead),
    # so the simulated spread is pulled in to the scatter real results have shown.
    margins = _rescale(card.sim.margin, state.fitted.sigma_margin)
    totals = _rescale(card.sim.total, state.fitted.sigma_total)
    pick = card.pick
    price = f" at {pick['price']:+.0f}" if pick.get("price") is not None else ""
    lines += [f"**{pick['grade']}: {pick['label']}{price}** — {pick['why']}.", ""]
    if odds.get("spread") is not None or odds.get("total") is not None:
        spread = (
            f"{home.abbr} {odds['spread']:+g}" if odds.get("spread") is not None else "no spread"
        )
        total = f"total {odds['total']:g}" if odds.get("total") is not None else "no total"
        ml = (
            f", moneyline {home.abbr} {odds['ml_home']:+.0f} / {away.abbr} {odds['ml_away']:+.0f}"
            if odds.get("ml_home") is not None and odds.get("ml_away") is not None
            else ""
        )
        lines.append(f"- Line ({odds.get('book')}): {spread}, {total}{ml}")
    lines += [
        (
            f"- Projection: {away.abbr} {away_pts:.1f}, {home.abbr} {home_pts:.1f} "
            f"({home.abbr} {-pricing.model_margin:+.1f}, total {pricing.model_total:.1f})"
        ),
        f"- {home.abbr} win probability: {pricing.model_p_home:.0%} by the model"
        + (
            f", {pricing.market_p_home:.0%} implied by the moneyline"
            if pricing.market_p_home is not None
            else ""
        ),
        (
            f"- Possession simulation ({len(card.sim.home):,} games, {m.possessions:.1f} possessions a side): "
            f"middle 80% of {home.abbr} margins {np.percentile(margins, 10):+.0f} to {np.percentile(margins, 90):+.0f}, "
            f"totals {np.percentile(totals, 10):.0f}-{np.percentile(totals, 90):.0f}"
        ),
    ]
    if m.h2h:
        meetings = "; ".join(
            f"{g['date'][:10]} {g['away_abbr']} {g['away_score']:.0f} @ {g['home_abbr']} {g['home_score']:.0f}"
            for g in m.h2h[-4:]
        )
        home_wins = sum(
            1 for g in m.h2h if (g["home_score"] > g["away_score"]) == (g["home"] == home.id)
        )
        lines.append(
            f"- Head to head, last {len(m.h2h)}: {home.abbr} {home_wins}-{len(m.h2h) - home_wins} ({meetings})"
        )
    else:
        lines.append("- Head to head: no meetings in the cached seasons")
    lines.append("")
    lines += _team_block(state, away, home, card.sim.away_players, card.injuries.get(away.id, []))
    lines.append("")
    lines += _team_block(state, home, away, card.sim.home_players, card.injuries.get(home.id, []))
    return "\n".join(lines)


def _rescale(draws: np.ndarray, sigma: float) -> np.ndarray:
    spread = float(np.std(draws))
    return draws.mean() + (draws - draws.mean()) * min(sigma / spread, 1.0) if spread > 0 else draws


def render_slate(state: State, cards: list[GameCard], day: date) -> str:
    edge_m = state.summary["calibration"]["edge_vs_line_margin"]["credited"]
    edge_t = state.summary["calibration"]["edge_vs_line_total"]["credited"]
    head = [
        f"# NBA card for {day:%A %B %d, %Y}",
        "",
        f"History: {state.games:,} games through {state.through[:10]}. "
        + (
            "The back-test credits this model with **no edge over the sportsbook line**, so every pick "
            "below is a LEAN (the model's side), not a recommended wager."
            if edge_m == 0 and edge_t == 0
            else "The back-test credits the model with an edge in at least one market; BET picks clear the expected-value threshold."
        ),
        "",
    ]
    if not cards:
        return "\n".join(head + ["No games waiting to tip off on this date."])
    ranked = sorted(
        cards,
        key=lambda c: (
            c.pick["grade"] != "BET",
            -abs(c.pick.get("ev") or 0),
            -abs(c.pick.get("gap") or 0),
        ),
    )
    head.append("| Game | Pick | Grade | Model vs line |")
    head.append("|---|---|---|---|")
    for c in ranked:
        gap = f"{abs(c.pick['gap']):.1f} pts" if c.pick.get("gap") is not None else "—"
        head.append(
            f"| {c.matchup.away.abbr} @ {c.matchup.home.abbr} | {c.pick['label']} | {c.pick['grade']} | {gap} |"
        )
    return "\n".join(head) + "\n\n" + "\n\n".join(render(state, c) for c in ranked)


# --------------------------------------------------------------------------- ledger


def save_picks(cards: list[GameCard], day: date) -> None:
    """Append today's picks to the forward ledger, keeping the first pick made per game."""
    PICKS_DIR.mkdir(parents=True, exist_ok=True)
    path = PICKS_DIR / f"{day.isoformat()}.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    for card in cards:
        if card.event_id in saved or card.pick.get("grade") == "NO LINE":
            continue
        keep = ("grade", "market", "side", "line", "price", "label", "prob", "ev")
        saved[card.event_id] = {
            "game": f"{card.matchup.away.abbr} @ {card.matchup.home.abbr}",
            "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
            **{k: card.pick.get(k) for k in keep},
        }
    path.write_text(json.dumps(saved, indent=1))


def grade() -> dict[str, dict[str, float]]:
    """Settle every saved pick whose game has finished."""
    results = {g["id"]: g for g in store.load_games()}
    settled: dict[str, list[dict[str, Any]]] = {"BET": [], "LEAN": []}
    for path in sorted(PICKS_DIR.glob("*.json")) if PICKS_DIR.exists() else []:
        for event_id, pick in json.loads(path.read_text()).items():
            game = results.get(event_id)
            if game is None or pick["grade"] not in settled:
                continue
            candidate = model.Candidate(
                pick["market"], pick["side"], pick["line"],
                pick["price"] if pick["price"] is not None else model.DEFAULT_PRICE, pick["prob"] or 0.5, 0.0,
            )  # fmt: skip
            margin = game["home"]["score"] - game["away"]["score"]
            total = game["home"]["score"] + game["away"]["score"]
            settled[pick["grade"]].append({"profit": model.settle(candidate, margin, total)})
    return {grade_name: model.record(bets) for grade_name, bets in settled.items()}


__all__ = [
    "GameCard",
    "State",
    "cards_for",
    "find_game",
    "game_day",
    "grade",
    "load_state",
    "render",
    "render_slate",
    "save_picks",
    "us_today",
]
