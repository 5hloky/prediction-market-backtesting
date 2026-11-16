"""Possession-by-possession Monte Carlo of a game.

Each possession is a chain of plays. A play is used by one player, drawn by usage,
and ends as a turnover, a trip to the line, or a two- or three-point attempt taken
at that player's own rates. A missed shot can be rebounded by the offence, which
starts another play in the same possession.

The mean of this simulation equals ``features.points_per_possession`` times the
possession count. Independent possessions spread final scores wider than real
games do, so cover probabilities are taken from the calibrated model in
:mod:`nba_bets.model`, not from the raw spread of these draws. The simulation is
used for the expected score, the shape of the game and who scores the points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nba_bets.features import Matchup, TeamView, points_per_possession

MAX_PLAYS = 6  # plays in one possession before we stop following offensive rebounds
POSSESSION_SD = 3.5


@dataclass
class SimResult:
    home: np.ndarray  # points per simulated game
    away: np.ndarray
    home_players: np.ndarray  # [simulation, player] points
    away_players: np.ndarray

    @property
    def margin(self) -> np.ndarray:
        return self.home - self.away

    @property
    def total(self) -> np.ndarray:
        return self.home + self.away


def _simulate_side(
    view: TeamView, possessions: np.ndarray, bump: float, rng: np.random.Generator
) -> np.ndarray:
    """Return a [simulation, player] array of points."""
    players = view.players
    n_sims, n_players = len(possessions), len(players)
    usage = np.array([p.usage for p in players])
    usage = usage / usage.sum()
    p_tov = np.array([p.p_tov for p in players])
    p_ft = np.array([p.p_ft for p in players])
    s3 = np.array([p.s3 for p in players])
    p2 = np.array([p.p2 for p in players])
    p3 = np.array([p.p3 for p in players])
    ftp = np.array([p.ftp for p in players])
    # Home-court edge: a small, uniform change in shot making.
    scale = 1.0 + bump
    p2, p3 = np.clip(p2 * scale, 0, 0.95), np.clip(p3 * scale, 0, 0.95)

    width = int(possessions.max())
    live = np.arange(width)[None, :] < possessions[:, None]
    sim_index = np.broadcast_to(np.arange(n_sims)[:, None], live.shape)[live]
    points = np.zeros((n_sims, n_players))
    for _ in range(MAX_PLAYS):
        count = len(sim_index)
        if count == 0:
            break
        who = rng.choice(n_players, size=count, p=usage)
        roll = rng.random(count)
        turnover = roll < p_tov[who]
        foul = ~turnover & (roll < p_tov[who] + p_ft[who])
        shot = ~turnover & ~foul
        three = shot & (rng.random(count) < s3[who])
        made = shot & (rng.random(count) < np.where(three, p3[who], p2[who]))
        scored = np.where(made, np.where(three, 3.0, 2.0), 0.0)
        # A trip is two shots, or three about a quarter of the time (and-ones, three-shot fouls).
        attempts = 2 + (rng.random(count) < 0.2727)
        scored = scored + foul * rng.binomial(attempts, ftp[who])
        np.add.at(points, (sim_index, who), scored)
        again = shot & ~made & (rng.random(count) < view.rates["orb"])
        sim_index = sim_index[again]
    return points


def simulate(
    matchup: Matchup,
    n: int = 10000,
    seed: int | None = 7,
    *,
    target_home: float | None = None,
    target_away: float | None = None,
) -> SimResult:
    """Simulate ``n`` games.

    ``target_*`` are expected scores to centre the simulation on (the calibrated
    model's projection). Without them the matchup's own expectation is used. Either
    way the adjustment is a uniform change in shot making, so who takes the shots
    and how the possessions end keep the structure the matchup describes.
    """
    rng = np.random.default_rng(seed)
    possessions = np.maximum(
        np.rint(rng.normal(matchup.possessions, POSSESSION_SD, size=n)).astype(int), 70
    )
    bumps = []
    for view, target in ((matchup.home, target_home), (matchup.away, target_away)):
        wanted = view.ppp if target is None else target / matchup.possessions
        bumps.append(_shooting_bump(view, wanted))
    home = _simulate_side(matchup.home, possessions, bumps[0], rng)
    away = _simulate_side(matchup.away, possessions, bumps[1], rng)
    home_pts, away_pts = home.sum(axis=1), away.sum(axis=1)
    # Overtime: ties are broken by five-minute periods played at the same rates.
    tied = home_pts == away_pts
    while tied.any():
        extra = np.maximum(np.rint(possessions[tied] * 5 / 48).astype(int), 8)
        extra_home = rng.poisson(extra * home_pts.mean() / possessions.mean())
        extra_away = rng.poisson(extra * away_pts.mean() / possessions.mean())
        home_pts[tied] += extra_home
        away_pts[tied] += extra_away
        tied = home_pts == away_pts
    return SimResult(home_pts, away_pts, home, away)


def _shooting_bump(view: TeamView, wanted_ppp: float) -> float:
    """Proportional change in two- and three-point accuracy that yields ``wanted_ppp``."""
    low, high = -0.4, 0.4
    for _ in range(40):
        mid = (low + high) / 2
        rates = dict(view.rates)
        rates["p2"] = min(rates["p2"] * (1 + mid), 0.95)
        rates["p3"] = min(rates["p3"] * (1 + mid), 0.95)
        if points_per_possession(rates) < wanted_ppp:
            low = mid
        else:
            high = mid
    return (low + high) / 2
