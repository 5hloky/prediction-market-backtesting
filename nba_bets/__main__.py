"""Command line for the NBA best-bets bot.

python -m nba_bets sync                 # download / refresh game history
python -m nba_bets today                # card for today's slate
python -m nba_bets today --date 2026-11-17
python -m nba_bets game "LAL @ PHX"     # one game, by team names or abbreviations
python -m nba_bets backtest             # walk-forward record against closing lines
python -m nba_bets grade                # settle the picks the bot has made so far
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from typing import Any

from nba_bets import daily, model, store


def backtest_report(summary: dict[str, Any], availability: str, line: str) -> str:
    def row(name: str, r: dict[str, float]) -> str:
        return (
            f"| {name} | {r['bets']:.0f} | {r['wins']:.0f}-{r['losses']:.0f}-{r['pushes']:.0f} | "
            f"{r['hit_rate']:.1%} | {r['units']:+.1f} | {r['roi']:+.1%} ± {r['roi_se']:.1%} |"
        )

    margin, total, cal = (
        summary.get("margin", {}),
        summary.get("total", {}),
        summary.get("calibration", {}),
    )
    out = [
        f"# Walk-forward backtest (lineups: {availability}, line: {line})",
        "",
        "Every month is priced by a model fitted only on earlier games.",
        "",
        "## How good is the projection?",
        "",
        "| | Bot | Sportsbook line |",
        "|---|---|---|",
        f"| Average miss on the margin | {margin.get('model_mae', 0):.2f} | {margin.get('market_mae', 0):.2f} |",
        f"| Picks the winner | {margin.get('model_winner_accuracy', 0):.1%} | {margin.get('market_winner_accuracy', 0):.1%} |",
        f"| Average miss on the total | {total.get('model_mae', 0):.2f} | {total.get('market_mae', 0):.2f} |",
        "",
        (
            f"Games graded: {margin.get('games', 0)}. Correlation between the bot's margin and the line: "
            f"{margin.get('model_market_correlation', 0):.2f}."
        ),
        "",
        "## Does siding with the bot against the line win?",
        "",
        "Break-even at -110 is 52.4%.",
        "",
        "| Gap between bot and line | Spread: games | Spread: hit rate | Total: games | Total: hit rate |",
        "|---|---|---|---|---|",
    ]
    ats, ou = margin.get("ats_by_disagreement", {}), total.get("ou_by_disagreement", {})
    for low, high in daily.GAP_BUCKETS:
        key = f"{low}-{high}"
        a, o = ats.get(key, {}), ou.get(key, {})
        name = f"{low}-{high} pts" if high < 99 else f"{low}+ pts"
        out.append(
            f"| {name} | {a.get('games', 0)} | {a.get('hit_rate', 0):.1%} | {o.get('games', 0)} | {o.get('hit_rate', 0):.1%} |"
        )
    em, et = cal.get("edge_vs_line_margin", {}), cal.get("edge_vs_line_total", {})
    out += [
        "",
        (
            "Edge credited after one standard error of caution (zero means no bets are recommended): "
            f"spread {em.get('credited', 0):.4f} (raw {em.get('slope', 0):+.4f} ± {em.get('standard_error', 0):.4f}), "
            f"total {et.get('credited', 0):.4f} (raw {et.get('slope', 0):+.4f} ± {et.get('standard_error', 0):.4f})."
        ),
        "",
        "## Bets the bot would have placed",
        "",
        "| | Bets | W-L-P | Hit rate | Units | ROI |",
        "|---|---|---|---|---|---|",
        row("All", summary["overall"]),
    ]
    out += [row(name.capitalize(), r) for name, r in summary["by_market"].items()]
    out += [
        row(f"Season {int(name) - 1}-{name[2:]}", r) for name, r in summary["by_season"].items()
    ]
    weights = cal.get("weights_margin", {})
    if weights:
        out += ["", "## What the margin model leans on", ""]
        out += [f"- `{k}`: {v:+.3f}" for k, v in weights.items()]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="nba_bets", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sync = sub.add_parser("sync", help="download or refresh game history from ESPN")
    sync.add_argument("--first-season", type=int, default=store.FIRST_SEASON)
    for name, text in (("today", "card for a slate"), ("game", "analysis and pick for one game")):
        p = sub.add_parser(name, help=text)
        if name == "game":
            p.add_argument("query", help='teams, e.g. "LAL @ PHX" or "celtics"')
        p.add_argument("--date", type=date.fromisoformat, default=None, help="US date, YYYY-MM-DD")
        p.add_argument("--no-refresh", action="store_true", help="skip the history refresh")
    bt = sub.add_parser("backtest", help="walk-forward record against the line")
    bt.add_argument("--availability", choices=("actual", "none"), default="actual")
    bt.add_argument("--line", choices=("close", "open"), default="close")
    bt.add_argument("--json", action="store_true")
    sub.add_parser("grade", help="settle the picks made so far")
    args = parser.parse_args(argv)

    if args.command == "sync":
        store.sync(args.first_season)
    elif args.command == "backtest":
        rows, _ = model.build_rows(
            store.load_games(), availability=args.availability, line=args.line
        )
        summary = model.summarize(model.walk_forward(rows))
        print(
            json.dumps(summary, indent=1)
            if args.json
            else backtest_report(summary, args.availability, args.line)
        )
    elif args.command == "grade":
        store.sync(verbose=False)
        for name, r in daily.grade().items():
            print(
                f"{name}: {r['bets']:.0f} settled, {r['wins']:.0f}-{r['losses']:.0f}-{r['pushes']:.0f}, "
                f"hit rate {r['hit_rate']:.1%}, {r['units']:+.2f} units"
            )
    else:
        state = daily.load_state(refresh=not args.no_refresh)
        day = args.date or daily.us_today()
        if args.command == "today":
            cards = daily.cards_for(state, day)
            daily.save_picks(cards, day)
            print(daily.render_slate(state, cards, day))
        else:
            card = daily.find_game(state, args.query, day)
            if card is None:
                raise SystemExit(f"no upcoming game matching {args.query!r} in the next few days")
            daily.save_picks([card], daily.game_day(card.tip))
            print(daily.render(state, card))


if __name__ == "__main__":
    main()
