"""Local cache of finished games (box score + closing line), built from ESPN.

The cache lives in ``nba_bets/data/`` (git-ignored) and is filled incrementally:
``sync`` only fetches days and games it has not already stored.
"""

from __future__ import annotations

import gzip
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from nba_bets import espn

DATA_DIR = Path(__file__).resolve().parent / "data"
GAMES_PATH = DATA_DIR / "games.jsonl.gz"
DAYS_PATH = DATA_DIR / "days_done.json"

# ESPN season types: 2 regular season, 3 playoffs, 5 play-in. Preseason (1) and the
# All-Star break exhibitions are never used.
SEASON_TYPES = (2, 3, 5)
FIRST_SEASON = 2024  # 2023-24, the earliest season worth the download


def season_window(season: int) -> tuple[date, date]:
    return date(season - 1, 10, 15), date(season, 6, 30)


def load_games(path: Path = GAMES_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        games = [json.loads(line) for line in handle if line.strip()]
    games.sort(key=lambda g: (g["date"], g["id"]))
    return games


def _save_games(games: dict[str, dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = GAMES_PATH.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as handle:
        for game in sorted(games.values(), key=lambda g: (g["date"], g["id"])):
            handle.write(json.dumps(game, separators=(",", ":")) + "\n")
    tmp.replace(GAMES_PATH)


def _final_ids(day: date) -> list[str] | None:
    """Ids of finished, countable games on ``day``; ``None`` if the lookup failed."""
    try:
        payload = espn.scoreboard(day.strftime("%Y%m%d"))
    except espn.EspnError:
        return None
    ids = []
    for event in payload.get("events") or []:
        if (event.get("season") or {}).get("type") not in SEASON_TYPES:
            continue
        if ((event.get("status") or {}).get("type") or {}).get("name") != "STATUS_FINAL":
            continue
        names = " ".join(
            (c.get("team") or {}).get("displayName") or ""
            for c in (event.get("competitions") or [{}])[0].get("competitors") or []
        )
        if "All-Star" in names or "Team " in names:
            continue
        ids.append(str(event["id"]))
    return ids


def _fetch_game(event_id: str) -> dict[str, Any] | None:
    try:
        game = espn.parse_game(espn.summary(event_id))
    except espn.EspnError:
        return None
    if game is None:
        return None
    try:
        game["odds"] = espn.parse_core_odds(espn.core_odds(event_id))
    except espn.EspnError:
        game["odds"] = None
    return game


def sync(first_season: int = FIRST_SEASON, *, workers: int = 8, verbose: bool = True) -> int:
    """Bring the cache up to date. Returns the number of games added."""
    games = {g["id"]: g for g in load_games()}
    done: set[str] = set(json.loads(DAYS_PATH.read_text())) if DAYS_PATH.exists() else set()
    today = datetime.now(UTC).date()
    # A day is only "done" once it is safely in the past, so late finals are not missed.
    settled = today - timedelta(days=2)

    days: list[date] = []
    for season in range(first_season, today.year + 2):
        start, end = season_window(season)
        day = start
        while day <= min(end, today):
            if day.isoformat() not in done:
                days.append(day)
            day += timedelta(days=1)

    def log(message: str) -> None:
        if verbose:
            print(message, file=sys.stderr, flush=True)

    log(f"scanning {len(days)} days for finished games")
    wanted: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for day, ids in zip(days, pool.map(_final_ids, days), strict=True):
            if ids is None:
                continue
            wanted.extend(i for i in ids if i not in games)
            if day <= settled:
                done.add(day.isoformat())

    log(f"fetching {len(wanted)} new games")
    added = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, game in enumerate(pool.map(_fetch_game, wanted), start=1):
            if game is not None:
                games[game["id"]] = game
                added += 1
            if index % 250 == 0:
                log(f"  {index}/{len(wanted)}")
                _save_games(games)
    _save_games(games)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DAYS_PATH.write_text(json.dumps(sorted(done)))
    log(f"cache holds {len(games)} games (+{added})")
    return added


# --------------------------------------------------------------------------- player props

PROPS_PATH = DATA_DIR / "props.jsonl.gz"
FIRST_PROPS_SEASON = 2025  # ESPN has no prop history before 2024-25


def load_props(path: Path = PROPS_PATH) -> dict[str, dict[str, Any]]:
    """Game id -> opening prop lines, ``{athlete_id: {stat: [line, over, under]}}``."""
    if not path.exists():
        return {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {row["id"]: row["props"] for row in map(json.loads, handle)}


def _fetch_props(event_id: str) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for book in espn.PROP_BOOKS:
        parsed = espn.parse_props(espn.prop_bets(event_id, book), opening=True)
        if sum(map(len, parsed.values())) > sum(map(len, best.values())):
            best = parsed
    return best


def sync_props(*, workers: int = 8, verbose: bool = True) -> int:
    """Fetch opening prop lines for every cached game that does not have them yet."""
    props = load_props()
    wanted = [
        g["id"] for g in load_games() if g["season"] >= FIRST_PROPS_SEASON and g["id"] not in props
    ]
    if verbose:
        print(f"fetching props for {len(wanted)} games", file=sys.stderr, flush=True)

    def save() -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = PROPS_PATH.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as handle:
            for game_id, value in props.items():
                handle.write(
                    json.dumps({"id": game_id, "props": value}, separators=(",", ":")) + "\n"
                )
        tmp.replace(PROPS_PATH)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (game_id, value) in enumerate(
            zip(wanted, pool.map(_fetch_props, wanted), strict=True), 1
        ):
            props[game_id] = value
            if index % 200 == 0:
                save()
                if verbose:
                    print(f"  {index}/{len(wanted)}", file=sys.stderr, flush=True)
    save()
    return len(wanted)
