"""Point-in-time league state and the pre-game view of a matchup.

``League`` is a rolling picture of every team and player, advanced one finished
game at a time. ``League.matchup`` reads that picture *before* a game is applied,
so nothing it returns can see the game's own result. The one piece of same-day
information it accepts is who is available, which is what an injury report gives
you before tip-off.

What the view contains:

* team structure: opponent-adjusted efficiency ratings, pace and the shot /
  turnover / foul / rebound profile on both ends;
* injuries: tonight's projected rotation compared with the rotation that produced
  the team's recent numbers, so a star sitting out moves the shot mix, the
  efficiency and the lineup's box-score value;
* player-vs-player: how each defence has handled each position group, applied to
  the players who will actually take the shots, plus each player's own history
  against tonight's opponent;
* head-to-head and situational context: recent meetings, rest, form.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np

KEYS = (
    "min", "pts", "fg2a", "fg2m", "fg3a", "fg3m", "fta", "ftm",
    "tov", "orb", "drb", "ast", "stl", "blk", "pf",
)  # fmt: skip
IX = {k: i for i, k in enumerate(KEYS)}
GROUPS = ("G", "F", "C")

# Typical per-minute production, used only until real data has accumulated.
_PRIOR_PER_MIN = np.array(
    [1.0, 0.47, 0.21, 0.115, 0.155, 0.056, 0.09, 0.07, 0.057, 0.043, 0.135, 0.11, 0.033, 0.02, 0.08]
)
# Typical team totals per game, same purpose.
_PRIOR_TEAM = {
    "poss": 99.5, "fg2a": 52.0, "fg2m": 28.3, "fg3a": 37.5, "fg3m": 13.5,
    "fta": 22.0, "ftm": 17.2, "tov": 14.0, "orb": 10.8, "opp_drb": 33.0,
}  # fmt: skip
TEAM_KEYS = tuple(_PRIOR_TEAM)

TEAM_HALF_LIFE = 20.0  # games
PLAYER_HALF_LIFE = 15.0  # games played
MINUTES_HALF_LIFE = 6.0  # games played
RATING_K = 0.08
HOME_EDGE_PER_100 = 2.4
SEASON_CARRY = 0.7
FT_PER_TRIP = 1 / 0.44
ROTATION_WINDOW = 8  # team games a player may miss and still count as in the rotation


def group_of(position: str) -> str:
    position = (position or "").upper()
    if position in ("PG", "SG", "G"):
        return "G"
    if position == "C":
        return "C"
    return "F"


def _decay(half_life: float) -> float:
    return 0.5 ** (1.0 / half_life)


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _expit(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def combine(off: float, dfn: float, league: float) -> float:
    """Offence rate vs defence rate allowed, relative to league (log-odds additive)."""
    return _expit(_logit(off) + _logit(dfn) - _logit(league))


def player_vector(p: dict[str, Any]) -> np.ndarray:
    v = np.zeros(len(KEYS))
    v[IX["min"]] = p["min"]
    v[IX["pts"]] = p["pts"]
    v[IX["fg2a"]] = p["fga"] - p["tpa"]
    v[IX["fg2m"]] = p["fgm"] - p["tpm"]
    v[IX["fg3a"]] = p["tpa"]
    v[IX["fg3m"]] = p["tpm"]
    for k in ("fta", "ftm", "tov", "orb", "drb", "ast", "stl", "blk", "pf"):
        v[IX[k]] = p[k]
    return v


def box_value(v: np.ndarray) -> float:
    """Hollinger game score: one number for a box-score line."""
    fgm = v[IX["fg2m"]] + v[IX["fg3m"]]
    fga = v[IX["fg2a"]] + v[IX["fg3a"]]
    pts = 2 * v[IX["fg2m"]] + 3 * v[IX["fg3m"]] + v[IX["ftm"]]
    return float(
        pts + 0.4 * fgm - 0.7 * fga - 0.4 * (v[IX["fta"]] - v[IX["ftm"]])
        + 0.7 * v[IX["orb"]] + 0.3 * v[IX["drb"]] + v[IX["stl"]] + 0.7 * v[IX["ast"]]
        + 0.7 * v[IX["blk"]] - 0.4 * v[IX["pf"]] - v[IX["tov"]]
    )  # fmt: skip


def team_line(side: dict[str, Any], opp: dict[str, Any]) -> dict[str, float]:
    return {
        "poss": side["fga"] - side["orb"] + side["tov"] + 0.44 * side["fta"],
        "fg2a": side["fga"] - side["tpa"],
        "fg2m": side["fgm"] - side["tpm"],
        "fg3a": side["tpa"],
        "fg3m": side["tpm"],
        "fta": side["fta"],
        "ftm": side["ftm"],
        "tov": side["tov"],
        "orb": side["orb"],
        "opp_drb": opp["drb"],
    }


def rates_from(sums: dict[str, float]) -> dict[str, float]:
    fga = sums["fg2a"] + sums["fg3a"]
    plays = fga + 0.44 * sums["fta"] + sums["tov"]
    return {
        "p_tov": sums["tov"] / plays,
        "p_ft": 0.44 * sums["fta"] / plays,
        "s3": sums["fg3a"] / fga,
        "p2": sums["fg2m"] / sums["fg2a"],
        "p3": sums["fg3m"] / sums["fg3a"],
        "ftp": sums["ftm"] / sums["fta"],
        "orb": sums["orb"] / (sums["orb"] + sums["opp_drb"]),
    }


def points_per_possession(r: dict[str, float]) -> float:
    """Closed-form expectation of the possession model in :mod:`nba_bets.sim`."""
    shot = 1 - r["p_tov"] - r["p_ft"]
    per_play = r["p_ft"] * r["ftp"] * FT_PER_TRIP + shot * (
        r["s3"] * 3 * r["p3"] + (1 - r["s3"]) * 2 * r["p2"]
    )
    again = shot * (r["s3"] * (1 - r["p3"]) + (1 - r["s3"]) * (1 - r["p2"])) * r["orb"]
    return per_play / (1 - again)


@dataclass
class Player:
    id: str
    name: str
    pos: str
    team: str = ""
    sums: np.ndarray = field(default_factory=lambda: np.zeros(len(KEYS)))
    games: float = 0.0
    mpg: float = 0.0  # minutes when he plays
    share: float = 0.0  # minutes per team game, absences counted as zero
    last_team_game: int = -1
    starts: float = 0.0
    vs: dict[str, list[float]] = field(default_factory=dict)  # opponent -> [pts, min]

    @property
    def group(self) -> str:
        return group_of(self.pos)


@dataclass
class Team:
    id: str
    abbr: str = ""
    name: str = ""
    off: dict[str, float] = field(default_factory=lambda: dict.fromkeys(TEAM_KEYS, 0.0))
    dfn: dict[str, float] = field(default_factory=lambda: dict.fromkeys(TEAM_KEYS, 0.0))
    weight: float = 0.0
    pace: float = _PRIOR_TEAM["poss"]
    off_rating: float = 0.0  # points per 100 above league, opponent-adjusted
    def_rating: float = 0.0  # points per 100 saved versus league, opponent-adjusted
    form: float = 0.0
    games_played: int = 0
    last_date: str = ""
    recent_dates: list[str] = field(default_factory=list)
    roster: set[str] = field(default_factory=set)
    dvp: dict[str, list[float]] = field(default_factory=lambda: {g: [0.0, 0.0] for g in GROUPS})
    log: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PlayerView:
    id: str
    name: str
    pos: str
    group: str
    minutes: float
    usage: float  # share of the team's possession-ending plays
    p_tov: float
    p_ft: float
    s3: float
    p2: float
    p3: float
    ftp: float
    matchup: float  # opponent's defence against this position, 1.0 = neutral
    history: float  # own scoring rate vs this opponent relative to own norm, shrunk
    ppm: float
    proj_pts: float = 0.0


@dataclass
class TeamView:
    id: str
    abbr: str
    name: str
    players: list[PlayerView]
    rates: dict[str, float]
    ppp: float
    out: list[tuple[str, float]]  # (name, usual minutes) for rotation players not available
    value_delta: float  # box value of tonight's rotation minus the usual rotation
    matchup_edge: float
    history_edge: float
    rest_days: int
    games_last_7: int


@dataclass
class Matchup:
    home: TeamView
    away: TeamView
    possessions: float
    features: dict[str, float]
    h2h: list[dict[str, Any]]

    @property
    def sim_margin(self) -> float:
        return self.possessions * (self.home.ppp - self.away.ppp)

    @property
    def sim_total(self) -> float:
        return self.possessions * (self.home.ppp + self.away.ppp)


MARGIN_FEATURES = (
    "sim_margin", "rating_margin", "value_diff", "matchup_diff",
    "history_diff", "h2h", "rest_diff", "form_diff", "home",
)  # fmt: skip
TOTAL_FEATURES = ("sim_total", "rating_total", "value_sum", "rest_sum", "pace_dev")


class League:
    def __init__(self) -> None:
        self.teams: dict[str, Team] = {}
        self.players: dict[str, Player] = {}
        self.totals = {k: v * 400 for k, v in _PRIOR_TEAM.items()}
        self.weight = 400.0  # decayed number of team-games behind ``totals``
        self.pace = _PRIOR_TEAM["poss"]
        self.pos = {g: _PRIOR_PER_MIN * 20000 for g in GROUPS}
        self.points_per_100 = 114.0
        self.season: int | None = None
        self.h2h: dict[tuple[str, str], list[dict[str, Any]]] = {}

    # ------------------------------------------------------------------ reads

    @property
    def rates(self) -> dict[str, float]:
        return rates_from(self.totals)

    def _team(self, team_id: str) -> Team:
        if team_id not in self.teams:
            self.teams[team_id] = Team(team_id)
        return self.teams[team_id]

    def _team_rates(self, sums: dict[str, float]) -> dict[str, float]:
        """A team's rates, shrunk toward league average by a few games' worth of prior."""
        league = {k: v / self.weight for k, v in self.totals.items()}
        rates = rates_from({k: sums[k] + 5.0 * league[k] for k in TEAM_KEYS})
        # Three-point accuracy is mostly luck game to game, so it gets a much heavier prior.
        rates["p3"] = (sums["fg3m"] + 18.0 * league["fg3m"]) / (
            sums["fg3a"] + 18.0 * league["fg3a"]
        )
        return rates

    def player_rates(self, player: Player | None, group: str) -> np.ndarray:
        """Per-minute production, shrunk toward the league norm for the position."""
        prior = self.pos[group] / self.pos[group][IX["min"]]
        if player is None:
            return prior.copy()
        s = player.sums
        minutes = s[IX["min"]]
        out = (s + 120.0 * prior) / (minutes + 120.0)
        for made, att, k in (("fg2m", "fg2a", 60.0), ("fg3m", "fg3a", 90.0), ("ftm", "fta", 30.0)):
            pct = (s[IX[made]] + k * prior[IX[made]] / prior[IX[att]]) / (s[IX[att]] + k)
            out[IX[made]] = out[IX[att]] * pct
        out[IX["pts"]] = 2 * out[IX["fg2m"]] + 3 * out[IX["fg3m"]] + out[IX["ftm"]]
        out[IX["min"]] = 1.0
        return out

    def rotation(self, team_id: str) -> list[Player]:
        """Players who have appeared for the team recently enough to be expected tonight."""
        team = self._team(team_id)
        return [
            self.players[i]
            for i in team.roster
            if team.games_played - self.players[i].last_team_game <= ROTATION_WINDOW
            and self.players[i].mpg > 0
        ]

    def dvp(self, team_id: str) -> dict[str, float]:
        """Scoring allowed to each position group relative to the team's own average."""
        team = self._team(team_id)
        raw, weights = {}, {}
        for g in GROUPS:
            league = self.pos[g][IX["pts"]] / self.pos[g][IX["min"]]
            pts, minutes = team.dvp[g]
            raw[g] = ((pts + 700.0 * league) / (minutes + 700.0)) / league
            weights[g] = self.pos[g][IX["pts"]]
        mean = sum(raw[g] * weights[g] for g in GROUPS) / sum(weights.values())
        return {g: raw[g] / mean for g in GROUPS}

    # ------------------------------------------------------------- projection

    def _minutes(self, players: list[Player | None], fallback: list[float]) -> np.ndarray:
        """Expected minutes tonight, summing to 240.

        When regulars are missing, the others each pick up only a few extra minutes;
        the rest goes down the bench, which is how coaches actually cover absences.
        """
        base = np.array(
            [
                p.mpg if p is not None and p.mpg > 0 else f
                for p, f in zip(players, fallback, strict=True)
            ]
        )
        base = np.maximum(base, 0.5)
        if base.sum() >= 240.0:
            return base * 240.0 / base.sum()
        ceiling = np.minimum(base + 4.0, 40.0)
        minutes = base.copy()
        for cap in (ceiling, np.maximum(ceiling, 30.0), np.full(len(base), 44.0)):
            for _ in range(8):
                room = np.maximum(cap - minutes, 0.0)
                short = 240.0 - minutes.sum()
                if short <= 1e-9 or room.sum() <= 1e-9:
                    break
                minutes += np.minimum(room, short * room / room.sum())
        return minutes * 240.0 / minutes.sum()

    def _team_view(
        self,
        team_id: str,
        opp_id: str,
        date: str,
        available: list[dict[str, Any]] | None,
    ) -> TeamView:
        team, opp = self._team(team_id), self._team(opp_id)
        rotation = self.rotation(team_id)
        if available is None:
            entries = [{"id": p.id, "name": p.name, "pos": p.pos} for p in rotation]
        else:
            entries = available
        if not entries:  # brand-new team with no history: a generic eight-man rotation
            entries = [
                {"id": f"{team_id}-{i}", "name": f"Player {i + 1}", "pos": "GGFFCGFC"[i]}
                for i in range(8)
            ]
        ids = {e["id"] for e in entries}
        players = [self.players.get(e["id"]) for e in entries]
        groups = [
            group_of(e.get("pos") or (p.pos if p else ""))
            for e, p in zip(entries, players, strict=True)
        ]
        minutes = self._minutes(players, [12.0] * len(entries))
        rates = np.array([self.player_rates(p, g) for p, g in zip(players, groups, strict=True)])
        tonight = rates.T @ minutes

        usual_players = [p for p in self.players.values() if p.team == team_id and p.share > 0.25]
        usual_share = np.array([p.share for p in usual_players])
        if usual_players and usual_share.sum() > 120:
            usual_minutes = usual_share * 240.0 / usual_share.sum()
            usual = (
                np.array([self.player_rates(p, p.group) for p in usual_players]).T @ usual_minutes
            )
        else:
            usual = tonight

        def ratio(fn: Any, power: float = 1.0) -> float:
            return float(np.clip((fn(tonight) / fn(usual)) ** power, 0.85, 1.18))

        def fga(v: np.ndarray) -> float:
            return v[IX["fg2a"]] + v[IX["fg3a"]]

        def plays(v: np.ndarray) -> float:
            return fga(v) + 0.44 * v[IX["fta"]] + v[IX["tov"]]

        # Team-level baseline: this offence against this defence, relative to league.
        league = self.rates
        off = self._team_rates(team.off)
        dfn = self._team_rates(opp.dfn)
        r = {k: combine(off[k], dfn[k], league[k]) for k in league}
        # Injury adjustment: how tonight's rotation differs from the usual one.
        r["p_tov"] *= ratio(lambda v: v[IX["tov"]] / plays(v))
        r["p_ft"] *= ratio(lambda v: v[IX["fta"]] / plays(v))
        r["s3"] *= ratio(lambda v: v[IX["fg3a"]] / fga(v))
        r["p2"] *= ratio(lambda v: v[IX["fg2m"]] / v[IX["fg2a"]])
        r["p3"] *= ratio(lambda v: v[IX["fg3m"]] / v[IX["fg3a"]])
        r["ftp"] *= ratio(lambda v: v[IX["ftm"]] / v[IX["fta"]])
        r["orb"] *= ratio(lambda v: v[IX["orb"]], 0.7)

        # Player-vs-position: weight the defence's weak spots by who takes the shots.
        dvp = self.dvp(opp_id)
        play_rate = (
            rates[:, IX["fg2a"]]
            + rates[:, IX["fg3a"]]
            + 0.44 * rates[:, IX["fta"]]
            + rates[:, IX["tov"]]
        )
        usage = minutes * play_rate
        usage = usage / usage.sum()
        matchup = np.array([dvp[g] ** 0.5 for g in groups])
        ppm = rates[:, IX["pts"]]
        history = np.zeros(len(entries))
        for i, p in enumerate(players):
            if p is not None and opp_id in p.vs:
                pts, mins = p.vs[opp_id]
                history[i] = ((pts + 150.0 * ppm[i]) / (mins + 150.0)) / ppm[i] - 1.0
        matchup_edge = float(usage @ (matchup - 1.0))
        history_edge = float(usage @ history)
        r["p2"] *= 1 + matchup_edge
        r["p3"] *= 1 + matchup_edge

        # Per-player profiles, rescaled so the usage-weighted mix equals the team rates.
        p_play = np.maximum(play_rate, 1e-6)
        shots = np.maximum(rates[:, IX["fg2a"]] + rates[:, IX["fg3a"]], 1e-6)
        shot_w = usage * shots / p_play

        def fit(raw: np.ndarray, weights: np.ndarray, target: float, cap: float) -> np.ndarray:
            mean = float(weights @ raw / weights.sum())
            return np.clip(raw * target / max(mean, 1e-6), 0.0, cap)

        p_tov = fit(rates[:, IX["tov"]] / p_play, usage, r["p_tov"], 0.45)
        p_ft = fit(0.44 * rates[:, IX["fta"]] / p_play, usage, r["p_ft"], 0.45)
        s3 = fit(rates[:, IX["fg3a"]] / shots, shot_w, r["s3"], 0.98)
        w2, w3 = shot_w * (1 - s3), shot_w * s3
        p2 = fit(
            rates[:, IX["fg2m"]] / np.maximum(rates[:, IX["fg2a"]], 1e-6) * matchup,
            w2,
            r["p2"],
            0.85,
        )
        p3 = fit(
            rates[:, IX["fg3m"]] / np.maximum(rates[:, IX["fg3a"]], 1e-6) * matchup,
            w3,
            r["p3"],
            0.6,
        )
        ft_w = usage * p_ft
        ftp = fit(rates[:, IX["ftm"]] / np.maximum(rates[:, IX["fta"]], 1e-6), ft_w, r["ftp"], 0.97)

        views = [
            PlayerView(
                id=e["id"], name=e.get("name") or (p.name if p else "?"),
                pos=e.get("pos") or (p.pos if p else ""), group=groups[i],
                minutes=float(minutes[i]), usage=float(usage[i]), p_tov=float(p_tov[i]),
                p_ft=float(p_ft[i]), s3=float(s3[i]), p2=float(p2[i]), p3=float(p3[i]),
                ftp=float(ftp[i]), matchup=float(matchup[i] ** 2), history=float(history[i]),
                ppm=float(ppm[i]),
            )
            for i, (e, p) in enumerate(zip(entries, players, strict=True))
        ]  # fmt: skip

        value = np.array([box_value(row) for row in rates])
        usual_value = box_value(usual)
        out = sorted(
            ((p.name, p.share) for p in rotation if p.id not in ids and p.share >= 8.0),
            key=lambda item: -item[1],
        )
        rest = 5
        if team.last_date:
            rest = min(_days_between(team.last_date, date), 5)
        return TeamView(
            id=team_id, abbr=team.abbr or team_id, name=team.name or team.abbr or team_id,
            players=views, rates=r, ppp=points_per_possession(r), out=out,
            value_delta=float(value @ minutes) - usual_value, matchup_edge=matchup_edge,
            history_edge=history_edge, rest_days=rest,
            games_last_7=sum(1 for d in team.recent_dates if _days_between(d, date) <= 7),
        )  # fmt: skip

    def matchup(
        self,
        home_id: str,
        away_id: str,
        date: str,
        *,
        home_available: list[dict[str, Any]] | None = None,
        away_available: list[dict[str, Any]] | None = None,
        neutral: bool = False,
        regulation_minutes: float = 48.0,
    ) -> Matchup:
        """Pre-game view. ``*_available`` are ``{"id","name","pos"}`` dicts, or ``None`` for
        "everyone in the recent rotation plays"."""
        home = self._team_view(home_id, away_id, date, home_available)
        away = self._team_view(away_id, home_id, date, away_available)
        th, ta = self._team(home_id), self._team(away_id)
        possessions = th.pace * ta.pace / self.pace * regulation_minutes / 48.0
        court = 0.0 if neutral else 1.0
        # Home teams score a little more efficiently and concede a little less.
        bump = court * HOME_EDGE_PER_100 / 200.0
        home.ppp += bump
        away.ppp -= bump
        rating_margin = (
            (th.off_rating - ta.def_rating) - (ta.off_rating - th.def_rating)
        ) * possessions / 100.0 + court * HOME_EDGE_PER_100 * possessions / 100.0
        rating_total = (
            (
                2 * self.points_per_100
                + th.off_rating
                - ta.def_rating
                + ta.off_rating
                - th.def_rating
            )
            * possessions
            / 100.0
        )
        meetings = self.h2h.get(_pair(home_id, away_id), [])[-6:]
        signed = [m["residual"] if m["home"] == home_id else -m["residual"] for m in meetings]
        features = {
            "sim_margin": possessions * (home.ppp - away.ppp),
            "rating_margin": rating_margin,
            "value_diff": home.value_delta - away.value_delta,
            "matchup_diff": 100 * (home.matchup_edge - away.matchup_edge),
            "history_diff": 100 * (home.history_edge - away.history_edge),
            "h2h": sum(signed) / (len(signed) + 4.0),
            "rest_diff": float(min(home.rest_days, 3) - min(away.rest_days, 3)),
            "form_diff": th.form - ta.form,
            "home": court,
            "sim_total": possessions * (home.ppp + away.ppp),
            "rating_total": rating_total,
            "value_sum": home.value_delta + away.value_delta,
            "rest_sum": float(min(home.rest_days, 3) + min(away.rest_days, 3)),
            "pace_dev": possessions - self.pace,
        }
        for view in (home, away):
            for p in view.players:
                p.proj_pts = _player_points(p, possessions, view.rates)
        return Matchup(home, away, possessions, features, meetings)

    # ---------------------------------------------------------------- updates

    def apply(self, game: dict[str, Any]) -> None:
        """Advance the state with a finished game."""
        if self.season is not None and game["season"] != self.season:
            self._new_season()
        self.season = game["season"]
        home, away = game["home"], game["away"]
        th, ta = self._team(home["id"]), self._team(away["id"])
        for team, side in ((th, home), (ta, away)):
            team.abbr, team.name = side["abbr"], side["name"]
        overtime = max(home.get("periods", 4), 4) - 4
        minutes = 48.0 + 5.0 * overtime
        line_h, line_a = team_line(home, away), team_line(away, home)
        poss = (line_h["poss"] + line_a["poss"]) / 2.0
        line_h["poss"] = line_a["poss"] = poss
        court = 0.0 if game.get("neutral") else 1.0

        expected_h = (
            self.points_per_100 + th.off_rating - ta.def_rating + court * HOME_EDGE_PER_100 / 2
        )
        expected_a = (
            self.points_per_100 + ta.off_rating - th.def_rating - court * HOME_EDGE_PER_100 / 2
        )
        actual_h, actual_a = 100 * home["score"] / poss, 100 * away["score"] / poss
        err_h, err_a = actual_h - expected_h, actual_a - expected_a
        residual = (err_h - err_a) * poss / 100.0
        th.off_rating += RATING_K * err_h
        ta.def_rating -= RATING_K * err_h
        ta.off_rating += RATING_K * err_a
        th.def_rating -= RATING_K * err_a
        self.pace += 0.004 * (poss * 48.0 / minutes - self.pace)
        self.points_per_100 += 0.004 * ((actual_h + actual_a) / 2 - self.points_per_100)
        th.form = 0.87 * th.form + 0.13 * residual
        ta.form = 0.87 * ta.form - 0.13 * residual

        self.h2h.setdefault(_pair(home["id"], away["id"]), []).append(
            {
                "date": game["date"], "home": home["id"], "home_abbr": home["abbr"],
                "away_abbr": away["abbr"], "home_score": home["score"],
                "away_score": away["score"], "residual": residual,
            }
        )  # fmt: skip

        decay_team, slow = _decay(TEAM_HALF_LIFE), _decay(1200.0)
        for team, own, other in ((th, line_h, line_a), (ta, line_a, line_h)):
            for key in TEAM_KEYS:
                team.off[key] = team.off[key] * decay_team + own[key]
                team.dfn[key] = team.dfn[key] * decay_team + other[key]
                self.totals[key] = self.totals[key] * slow + own[key]
            team.weight = team.weight * decay_team + 1.0
            self.weight = self.weight * slow + 1.0
            alpha = 1 / min(team.games_played + 4.0, 14.0)
            team.pace += alpha * (poss * 48.0 / minutes - team.pace)

        for team, side, opp_team in ((th, home, ta), (ta, away, th)):
            self._apply_players(team, side, opp_team, decay_team)
            team.games_played += 1
            team.last_date = game["date"]
            team.recent_dates = (team.recent_dates + [game["date"]])[-6:]
            team.log.append(
                {
                    "date": game["date"], "opp": opp_team.id, "home": team is th,
                    "pts": side["score"], "opp_pts": (away if team is th else home)["score"],
                    "spread": _side_spread(game, team is th), "total": (game.get("odds") or {}).get("total"),
                }
            )  # fmt: skip

    def _apply_players(
        self, team: Team, side: dict[str, Any], opp: Team, decay_team: float
    ) -> None:
        decay_p, decay_m, slow = (
            _decay(PLAYER_HALF_LIFE),
            _decay(MINUTES_HALF_LIFE),
            _decay(40000.0),
        )
        played: set[str] = set()
        for entry in opp.dvp.values():
            entry[0] *= decay_team
            entry[1] *= decay_team
        for row in side["players"]:
            if row["min"] <= 0:
                continue
            player = self.players.get(row["id"])
            if player is None:
                player = self.players[row["id"]] = Player(row["id"], row["name"], row["pos"])
            if player.team != team.id:
                if player.team in self.teams:
                    self.teams[player.team].roster.discard(player.id)
                player.team, player.share = team.id, 0.0
                team.roster.add(player.id)
            player.name, player.pos = row["name"], row["pos"] or player.pos
            vec = player_vector(row)
            player.sums = player.sums * decay_p + vec
            player.games = player.games * decay_p + 1.0
            player.mpg = (
                row["min"] if player.mpg == 0 else decay_m * player.mpg + (1 - decay_m) * row["min"]
            )
            player.starts = 0.8 * player.starts + 0.2 * float(row["starter"])
            player.last_team_game = team.games_played + 1
            seen = player.vs.setdefault(opp.id, [0.0, 0.0])
            seen[0] += row["pts"]
            seen[1] += row["min"]
            group = player.group
            self.pos[group] = self.pos[group] * slow + vec
            opp.dvp[group][0] += row["pts"]
            opp.dvp[group][1] += row["min"]
            played.add(player.id)
        box = {row["id"]: row["min"] for row in side["players"]}
        for player_id in team.roster:
            player = self.players[player_id]
            player.share = decay_team * player.share + (1 - decay_team) * box.get(player_id, 0.0)
            if player.id in played and player.share < 0.5 * player.mpg and team.games_played < 6:
                player.share = player.mpg  # early season: no usual rotation to compare against yet

    def _new_season(self) -> None:
        for team in self.teams.values():
            team.off_rating *= SEASON_CARRY
            team.def_rating *= SEASON_CARRY
            team.form = 0.0
            team.weight *= 0.3
            for key in TEAM_KEYS:
                team.off[key] *= 0.3
                team.dfn[key] *= 0.3
            for entry in team.dvp.values():
                entry[0] *= 0.3
                entry[1] *= 0.3
            team.games_played = 0
            team.last_date = ""
            team.recent_dates = []
            team.log = []
        for player in self.players.values():
            player.last_team_game = 0
            player.share = 0.0


def _player_points(p: PlayerView, possessions: float, team_rates: dict[str, float]) -> float:
    shot = max(1 - p.p_tov - p.p_ft, 0.0)
    per_play = p.p_ft * p.ftp * FT_PER_TRIP + shot * (p.s3 * 3 * p.p3 + (1 - p.s3) * 2 * p.p2)
    r = team_rates
    team_shot = 1 - r["p_tov"] - r["p_ft"]
    again = team_shot * (r["s3"] * (1 - r["p3"]) + (1 - r["s3"]) * (1 - r["p2"])) * r["orb"]
    return possessions / (1 - again) * p.usage * per_play


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _side_spread(game: dict[str, Any], is_home: bool) -> float | None:
    spread = (game.get("odds") or {}).get("spread")
    if spread is None:
        return None
    return spread if is_home else -spread


def _days_between(earlier: str, later: str) -> int:
    return (game_day(later) - game_day(earlier)).days


def game_day(iso: str) -> date:
    """US calendar day a game belongs to (tip-off times are stored in UTC)."""
    stamp = datetime.fromisoformat(iso) if "T" in iso else None
    if stamp is None:
        return date.fromisoformat(iso[:10])
    return (stamp - timedelta(hours=6)).date()


def available_from_box(side: dict[str, Any]) -> list[dict[str, Any]]:
    """Who actually played: the backtest's stand-in for the pre-game injury report."""
    return [
        {"id": p["id"], "name": p["name"], "pos": p["pos"]} for p in side["players"] if p["min"] > 0
    ]
