"""Thin client for ESPN's public (unauthenticated, undocumented) NBA endpoints.

Everything the bot knows comes from here: schedules, final box scores, sportsbook
lines and the league injury report. The endpoints are unofficial and can change
shape without notice, so every parser below is defensive and returns ``None``
rather than guessing when a field is missing.
"""

from __future__ import annotations

import gzip
import json
import time
import urllib.error
import urllib.request
from typing import Any

SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"
USER_AGENT = "Mozilla/5.0 (compatible; nba-bets/1.0)"

# Books we trust for a pre-game number, best first. In-play feeds are never used.
BOOK_PRIORITY = ("DraftKings", "ESPN BET", "MGM", "Caesars Sportsbook (New Jersey)", "consensus")


class EspnError(RuntimeError):
    pass


def get_json(url: str, *, retries: int = 4, timeout: float = 30.0) -> dict[str, Any]:
    last: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
                if response.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 404):
                raise EspnError(f"{exc.code} for {url}") from exc
            last = exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last = exc
        time.sleep(1.5 * (attempt + 1))
    raise EspnError(f"giving up on {url}: {last}")


def scoreboard(yyyymmdd: str) -> dict[str, Any]:
    return get_json(f"{SITE}/scoreboard?dates={yyyymmdd}&limit=100")


def summary(event_id: str) -> dict[str, Any]:
    return get_json(f"{SITE}/summary?event={event_id}")


def core_odds(event_id: str) -> dict[str, Any]:
    return get_json(f"{CORE}/events/{event_id}/competitions/{event_id}/odds?limit=100")


def injuries() -> dict[str, Any]:
    return get_json(f"{SITE}/injuries")


def roster(team_id: str) -> dict[str, Any]:
    return get_json(f"{SITE}/teams/{team_id}/roster")


# --------------------------------------------------------------------------- parsing


def _num(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower().replace("−", "-")
    if text in ("", "--", "-", "off"):
        return None
    if text in ("even", "ev"):
        return 100.0
    text = text.lstrip("ou")
    try:
        return float(text)
    except ValueError:
        return None


def _pair(text: str) -> tuple[float, float]:
    made, _, attempted = text.partition("-")
    return (_num(made) or 0.0, _num(attempted) or 0.0)


def parse_core_odds(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Pick one pre-game book and normalise its closing (and opening) numbers.

    ``spread`` is the home team's line: negative means the home team is favoured.
    """
    books: dict[str, dict[str, Any]] = {}
    for item in payload.get("items") or []:
        name = (item.get("provider") or {}).get("name") or ""
        if "live" in name.lower():
            continue
        if item.get("spread") is None and item.get("overUnder") is None:
            continue
        books[name] = item
    if not books:
        return None
    name = next((b for b in BOOK_PRIORITY if b in books), min(books))
    item = books[name]
    home, away = item.get("homeTeamOdds") or {}, item.get("awayTeamOdds") or {}

    def opened(side: dict[str, Any], key: str) -> float | None:
        return _num((((side.get("open") or {}).get(key)) or {}).get("american"))

    return {
        "book": name,
        "spread": _num(item.get("spread")),
        "total": _num(item.get("overUnder")),
        "ml_home": _num(home.get("moneyLine")),
        "ml_away": _num(away.get("moneyLine")),
        "spread_odds_home": _num(home.get("spreadOdds")),
        "spread_odds_away": _num(away.get("spreadOdds")),
        "over_odds": _num(item.get("overOdds")),
        "under_odds": _num(item.get("underOdds")),
        "open_spread": opened(home, "pointSpread"),
        "open_total": _num((((item.get("open") or {}).get("total")) or {}).get("american")),
    }


def parse_scoreboard_odds(competition: dict[str, Any]) -> dict[str, Any] | None:
    """Current line for an upcoming game, from the scoreboard payload."""
    entries = competition.get("odds") or []
    if not entries:
        return None
    item = entries[0]

    def at(*path: str) -> float | None:
        node: Any = item
        for key in path:
            node = (node or {}).get(key)
        return _num(node)

    spread = at("pointSpread", "home", "close", "line")
    if spread is None and item.get("spread") is not None:
        favourite_home = (item.get("homeTeamOdds") or {}).get("favorite")
        spread = -abs(float(item["spread"])) if favourite_home else abs(float(item["spread"]))
    return {
        "book": (item.get("provider") or {}).get("name") or "unknown",
        "spread": spread,
        "total": _num(item.get("overUnder")),
        "ml_home": at("moneyline", "home", "close", "odds"),
        "ml_away": at("moneyline", "away", "close", "odds"),
        "spread_odds_home": at("pointSpread", "home", "close", "odds"),
        "spread_odds_away": at("pointSpread", "away", "close", "odds"),
        "over_odds": at("total", "over", "close", "odds"),
        "under_odds": at("total", "under", "close", "odds"),
        "open_spread": at("pointSpread", "home", "open", "line"),
        "open_total": at("total", "over", "open", "line"),
    }


def _team_box(team: dict[str, Any]) -> dict[str, float]:
    stats = {s.get("name"): s.get("displayValue") for s in team.get("statistics") or []}
    fgm, fga = _pair(stats.get("fieldGoalsMade-fieldGoalsAttempted") or "0-0")
    tpm, tpa = _pair(stats.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted") or "0-0")
    ftm, fta = _pair(stats.get("freeThrowsMade-freeThrowsAttempted") or "0-0")
    tov = _num(stats.get("totalTurnovers"))
    if tov is None:
        tov = _num(stats.get("turnovers")) or 0.0
    return {
        "fgm": fgm,
        "fga": fga,
        "tpm": tpm,
        "tpa": tpa,
        "ftm": ftm,
        "fta": fta,
        "orb": _num(stats.get("offensiveRebounds")) or 0.0,
        "drb": _num(stats.get("defensiveRebounds")) or 0.0,
        "ast": _num(stats.get("assists")) or 0.0,
        "tov": tov,
    }


def _players(block: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group in block.get("statistics") or []:
        keys = group.get("keys") or []
        for entry in group.get("athletes") or []:
            athlete = entry.get("athlete") or {}
            row = dict(zip(keys, entry.get("stats") or [], strict=False))
            minutes = _num(row.get("minutes")) or 0.0
            fgm, fga = _pair(row.get("fieldGoalsMade-fieldGoalsAttempted") or "0-0")
            tpm, tpa = _pair(
                row.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted") or "0-0"
            )
            ftm, fta = _pair(row.get("freeThrowsMade-freeThrowsAttempted") or "0-0")
            out.append(
                {
                    "id": str(athlete.get("id")),
                    "name": athlete.get("displayName") or "?",
                    "pos": ((athlete.get("position") or {}).get("abbreviation")) or "",
                    "starter": bool(entry.get("starter")),
                    "min": minutes,
                    "pts": _num(row.get("points")) or 0.0,
                    "fgm": fgm,
                    "fga": fga,
                    "tpm": tpm,
                    "tpa": tpa,
                    "ftm": ftm,
                    "fta": fta,
                    "orb": _num(row.get("offensiveRebounds")) or 0.0,
                    "drb": _num(row.get("defensiveRebounds")) or 0.0,
                    "ast": _num(row.get("assists")) or 0.0,
                    "tov": _num(row.get("turnovers")) or 0.0,
                    "stl": _num(row.get("steals")) or 0.0,
                    "blk": _num(row.get("blocks")) or 0.0,
                    "pf": _num(row.get("fouls")) or 0.0,
                }
            )
    return out


def parse_game(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Flatten a finished game's summary payload into the compact record we store."""
    header = payload.get("header") or {}
    competitions = header.get("competitions") or []
    if not competitions:
        return None
    competition = competitions[0]
    if ((competition.get("status") or {}).get("type") or {}).get("name") != "STATUS_FINAL":
        return None
    box = payload.get("boxscore") or {}
    team_boxes = {str((t.get("team") or {}).get("id")): t for t in box.get("teams") or []}
    player_blocks = {str((t.get("team") or {}).get("id")): t for t in box.get("players") or []}
    sides: dict[str, dict[str, Any]] = {}
    for competitor in competition.get("competitors") or []:
        team = competitor.get("team") or {}
        team_id = str(team.get("id"))
        if team_id not in team_boxes or team_id not in player_blocks:
            return None
        score = _num(competitor.get("score"))
        if score is None:
            return None
        sides[competitor.get("homeAway")] = {
            "id": team_id,
            "abbr": team.get("abbreviation") or team_id,
            "name": team.get("displayName") or team.get("abbreviation") or team_id,
            "score": score,
            "periods": len(competitor.get("linescores") or []),
            **_team_box(team_boxes[team_id]),
            "players": _players(player_blocks[team_id]),
        }
    if set(sides) != {"home", "away"} or sides["home"]["fga"] <= 0 or sides["away"]["fga"] <= 0:
        return None
    season = header.get("season") or {}
    return {
        "id": str(header.get("id")),
        "date": competition.get("date"),
        "season": season.get("year"),
        "stype": season.get("type"),
        "neutral": bool(competition.get("neutralSite")),
        "home": sides["home"],
        "away": sides["away"],
        "odds": None,
    }
