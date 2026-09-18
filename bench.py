#!/usr/bin/env python3
"""
The evaluation harness.

One episode is an anecdote. This runs the agent against each opponent over a range of seeds,
in both seats, and reports the distribution — mean, worst, best and how often it wins.

**Both seats, always.** The two farms are symmetric but the map is not: the shed-access
tiles at (4,4) and (5,4) sit on opposite sides of the NW/NE boundary, and a policy that
quietly depends on which side its farmer wakes up on would look fine in seat 0 and lose in
seat 1. Running one seat only is how that gets missed.

  python3 bench.py --agent main.py --seeds 12
  python3 bench.py --agent main.py --opponents starter random pass --seeds 20 --json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sim import Environment, resolve_agent  # noqa: E402


def play(agent, opponent, seed: int, agent_seat: int) -> dict:
    """One episode, agent in the given seat."""
    players = [opponent, agent] if agent_seat == 1 else [agent, opponent]
    environment = Environment()
    environment.run(players, seed=seed)
    rewards = environment.rewards()
    my_index = 1 if agent_seat == 1 else 0
    return {
        "seed": seed,
        "seat": agent_seat,
        "mine": rewards[my_index],
        "theirs": rewards[1 - my_index],
        "margin": rewards[my_index] - rewards[1 - my_index],
    }


def summarise(label: str, episodes: list[dict]) -> dict:
    mine = [e["mine"] for e in episodes]
    margins = [e["margin"] for e in episodes]
    wins = sum(1 for m in margins if m > 0)
    ties = sum(1 for m in margins if m == 0)
    return {
        "opponent": label,
        "episodes": len(episodes),
        "mean_mine": round(statistics.fmean(mine), 1),
        "worst_mine": min(mine),
        "best_mine": max(mine),
        "mean_margin": round(statistics.fmean(margins), 1),
        "worst_margin": min(margins),
        "wins": wins,
        "ties": ties,
        "losses": len(margins) - wins - ties,
        "win_rate": round(wins / len(margins), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a Kaggriculture agent.")
    parser.add_argument("--agent", default="main.py")
    parser.add_argument("--opponents", nargs="+", default=["starter", "random", "pass"])
    parser.add_argument("--seeds", type=int, default=10, help="episodes per opponent, split across both seats")
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", default=None, help="write the JSON result here")
    args = parser.parse_args()

    agent = resolve_agent(args.agent)
    results = {}

    for name in args.opponents:
        opponent = resolve_agent(name)
        episodes = []
        for index in range(args.seeds):
            # Alternate seats so each seed is played from both sides.
            seat = index % 2
            episodes.append(play(agent, opponent, args.start_seed + index, seat))
        results[name] = summarise(name, episodes)

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    print(f"agent {args.agent} · {args.seeds} episodes per opponent, alternating seats\n")
    print(f"{'opponent':>10} {'mean':>9} {'worst':>8} {'best':>8} {'margin':>9} {'W-T-L':>10} {'win%':>7}")
    print("-" * 68)
    for row in results.values():
        record = f"{row['wins']}-{row['ties']}-{row['losses']}"
        print(
            f"{row['opponent']:>10} {row['mean_mine']:>9.1f} {row['worst_mine']:>8.0f} "
            f"{row['best_mine']:>8.0f} {row['mean_margin']:>+9.1f} {record:>10} "
            f"{row['win_rate'] * 100:>6.0f}%"
        )
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
