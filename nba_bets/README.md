# nba_bets

A bot that scans the day's NBA games, reads the injury report, analyses each
matchup, simulates the game possession by possession, and gives one pick per
game plus a sheet of player props: points, rebounds, assists, threes, steals,
blocks and the combinations, each as more or less than the sportsbook's line.
Every pick is graded against a walk-forward backtest, and the bot tells you what
that backtest says.

**Read this first.** Over 3,434 back-tested games the bot's projections are close
to the sportsbook's (average miss of 11.3 points on the margin against the line's
10.9), but siding with the bot against the closing line has won about 50% of the
time. Break-even at -110 is 52.4%. So today the bot credits itself with no edge
on game lines and labels every game pick a `LEAN`, not a `BET`.

Player props are different, with a caveat. Against *opening* prop lines the bot's
bets returned +6.9% over 4,379 back-tested bets, mostly on rebounds, assists,
steals and blocks. Most of that came from 2024-25 (+14.7%). The last two seasons
were +2.7% and +2.1%, which is within noise of zero, and opening lines are softer
than what you can bet near tip-off. Points and the points combinations show
nothing. So a prop marked `BET` means "the best-supported pick the bot has", not
"this will win".

Treat the whole thing as a research and handicapping tool. It is not a way to
make money, and nothing here is betting advice. Full numbers are in
[BACKTEST.md](BACKTEST.md).

## Use

Needs Python 3.11+ and `numpy`. Run from the repository root.

```bash
python -m nba_bets sync                 # first run: ~25 minutes, three-plus seasons
python -m nba_bets today                # today's card (refreshes history first)
python -m nba_bets today --date 2026-11-17
python -m nba_bets game "LAL @ PHX"     # one game; also "lakers suns" or "celtics"
python -m nba_bets backtest             # walk-forward record against closing lines
python -m nba_bets backtest --props     # the same for player props, against opening lines
python -m nba_bets grade                # settle the picks the bot has made so far
```

To run it daily, schedule `python -m nba_bets today`, for example with cron:

```
30 15 * * *  cd /path/to/repo && python -m nba_bets today > nba_card.md
```

Each run of `today` or `game` saves its picks with the line at that moment under
`nba_bets/picks/`. `grade` settles them once the games finish, which builds a true
forward record that no backtest can fake. That record matters most for props,
because it is graded at the prices the bot actually saw.

## Reading the props sheet

Each game has one row per player and one column per stat. A cell reads
`7.4 · More 5.5 (57%)`: the projection, the side of the line the bot prefers, and
its chance of winning after shrinking the raw model toward the book by what the
backtest has earned. A ★ marks a `BET`: a prop with both prices posted whose
expected value clears 2%. Before books post lines the cell is the projection
alone. The slate opens with the fifteen strongest props.

## What it looks at

| You asked for | What the bot does |
|---|---|
| Injury reports | Pulls the league injury report. Players ruled out are removed from the rotation, their minutes go mostly down the bench, and the team's shot mix, efficiency and lineup value are recomputed. Questionable players are listed and treated as playing. |
| History between the two teams | Last six meetings, scored against what the teams' ratings predicted at the time. |
| History against everyone else | Opponent-adjusted offence and defence ratings, pace, record, record against the spread, overs and unders, recent form, rest. |
| Player against player | How each defence has handled guards, forwards and centres, applied to the players who will take the shots, plus each player's own scoring history against tonight's opponent. |
| How possessions turn out | A Monte Carlo of the game. Each possession is a chain of plays used by a specific player and ending in a turnover, free throws, or a two or three taken at that player's rates, with offensive rebounds extending it. |
| Player stats | Scoring and threes come from the possession model. Rebounds, assists, steals and blocks are the player's per-minute rate times projected minutes, game pace and what the opponent concedes to that position. Each stat gets a negative-binomial spread fitted to past misses. |
| Back-tested bets | A ridge model is refit each month on earlier games only, and its disagreements with the line are credited only as far as they have held up out of sample. |

## What the backtest found

- **Useful:** opponent-adjusted team ratings, the injury adjustment and the
  possession model. Knowing who is out improves the margin projection from an
  average miss of 11.35 to 11.25.
- **Not useful, measured:** with the model trained on the first two seasons and
  tested on the 1,353 games since, removing head-to-head history, positional
  matchups or a player's history against an opponent each changed the average
  miss by less than 0.01 points. They are shown on the card for context, but
  they barely move the pick.
- **No edge over the line** on spreads or totals, at any size of disagreement.
  When the bot disagreed with the closing spread by five points or more it was
  right 44% of the time, which says the market knew something the bot did not.

## Limits you should know about

- **Lineups in the backtest use hindsight.** History has no archive of injury
  reports, so the backtest uses who actually played. That flatters the bot
  slightly. `backtest --availability none` withholds it entirely; the conclusion
  is the same.
- **Closing lines, one book.** Bets are graded at the closing number from one
  sportsbook per game (DraftKings where available, otherwise ESPN BET). Live, you
  see an earlier line. ESPN's stored opening totals are too noisy to grade
  against, so `backtest --line open` is for curiosity only.
- **No true defender matchups.** Box scores do not record who guarded whom, so
  "player against player" is position against team defence.
- **Props are graded at opening prices, and only two-way ones.** A finished
  game's stored "current" prop price was updated in play, so only the opener is
  trustworthy. Props posted with a more price alone are shown as leans and never
  bet. The prop backtest does not tell the bot who sat out.
- **The prop edge is shrinking.** The bot credits itself only from the most
  recent 1,500 graded props per market, so a market that stops working stops
  producing bets.
- **Unofficial data source.** Everything comes from ESPN's public, undocumented
  endpoints. They can change or disappear, and you should check their terms
  before relying on them.

If you bet, set a limit you can afford to lose. In the US, 1-800-GAMBLER offers
free, confidential help.

## Layout

| File | Role |
|---|---|
| `espn.py` | Fetching and parsing schedules, box scores, lines, injuries |
| `store.py` | Incremental local cache of finished games (`data/`, git-ignored) |
| `features.py` | Point-in-time team and player state, and the pre-game matchup view |
| `sim.py` | Possession-by-possession simulation |
| `model.py` | Calibration, pricing, and the walk-forward backtest |
| `props.py` | Player-stat distributions, prop pricing and the prop backtest |
| `picks/` | The forward record: every pick the bot has made, with the line it saw |
| `daily.py` | Slate scan, injury report, pick selection, card rendering, pick ledger |
| `tests/` | `pytest nba_bets/tests --rootdir nba_bets` |
