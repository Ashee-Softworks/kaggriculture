#!/usr/bin/env python3
"""
Read one downloaded episode replay and report what actually happened.

`kaggle competitions replay <episode_id>` writes a multi-megabyte JSON. This pulls out the
only things that decide a score: each player's bank over time, what they spent it on, the
market price of what they sold, and how the episode ended.

  python3 analyse_replay.py /tmp/replay/episode-110329561-replay.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def analyse(path: Path) -> dict:
    """Everything the replay decides, as data rather than as prose."""
    replay = json.loads(path.read_text(encoding="utf-8"))
    steps = replay.get("steps") or []
    config = replay.get("configuration") or {}

    result: dict = {
        "file": str(path),
        "steps": len(steps),
        "configuration": {
            key: config.get(key)
            for key in ("episodeSteps", "boardSize", "startingMoney", "turnsPerDay", "actTimeout", "seed")
            if key in config
        },
        "players": [],
        "trajectory": [],
    }
    if not steps:
        return result

    last = steps[-1]
    for index, agent in enumerate(last):
        observation = agent.get("observation") or {}
        farms = observation.get("farms") or []
        private = observation.get("private") or {}
        tiles = (farms[index].get("tiles") if index < len(farms) else None) or []
        kinds: Counter = Counter()
        for row in tiles:
            for tile in row:
                if tile is None:
                    kinds["empty"] += 1
                elif tile == "LOCKED":
                    kinds["locked"] += 1
                elif isinstance(tile, dict):
                    kinds[tile.get("kind", "?")] += 1
        result["players"].append(
            {
                "player": index,
                "status": agent.get("status"),
                "reward": agent.get("reward"),
                "money": farms[index].get("money") if index < len(farms) else None,
                "quadrants": farms[index].get("unlocked_quadrants") if index < len(farms) else None,
                "tiles": dict(kinds),
                "shed": private.get("shed") or {},
                "seeds": private.get("seeds") or {},
            }
        )

    for step_index in range(0, len(steps), 60):
        observation = (steps[step_index][0].get("observation") or {})
        farms = observation.get("farms") or []
        market = observation.get("market") or {}
        result["trajectory"].append(
            {
                "step": step_index,
                "day": observation.get("day"),
                "money": [farm.get("money") for farm in farms],
                "wheatPrice": (market.get("prices") or {}).get("WHEAT"),
                "wheatInventory": (market.get("inventory") or {}).get("WHEAT"),
            }
        )

    observation = last[0].get("observation") or {}
    result["finalMarket"] = {
        "prices": (observation.get("market") or {}).get("prices"),
        "inventory": (observation.get("market") or {}).get("inventory"),
        "town": (observation.get("town") or {}).get("unlocked_shops"),
    }
    return result


def render(data: dict) -> str:
    """The readable form of one episode."""
    lines = [
        f"# {Path(data['file']).name}",
        "",
        f"steps {data['steps']}",
        "",
    ]
    for key, value in data.get("configuration", {}).items():
        lines.append(f"- `{key}` = {value}")
    lines += ["", "## Final state", "", "| player | reward | money | quadrants | tiles |", "|---|---|---|---|---|"]
    for player in data.get("players", []):
        lines.append(
            f"| {player['player']} | {player['reward']} | {player['money']} | "
            f"{player.get('quadrants')} | {player.get('tiles')} |"
        )
    lines += ["", "## Bank over time", "", "| step | day | P0 | P1 | wheat price | wheat inv |", "|---|---|---|---|---|---|"]
    for row in data.get("trajectory", []):
        money = row.get("money") or []
        lines.append(
            f"| {row['step']} | {row['day']} | {money[0] if len(money) > 0 else '—'} | "
            f"{money[1] if len(money) > 1 else '—'} | {row.get('wheatPrice')} | {row.get('wheatInventory')} |"
        )
    final = data.get("finalMarket") or {}
    lines += ["", "## Final market", "", f"- prices {final.get('prices')}", f"- town {final.get('town')}", ""]
    return "\n".join(lines)


def main() -> int:
    """
    Read one or more replays and write the result where it can be committed.

    The replays themselves are six megabytes each and are not committed. What is committed is
    the extracted result: both players' final money, what they built, and the bank over time.
    That is the part a later reader needs, and it is small enough to live beside the code.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Read a Kaggriculture episode replay.")
    parser.add_argument("replays", nargs="+", help="paths to episode replay JSON files")
    parser.add_argument("--out", default=None, help="directory to write results into")
    parser.add_argument("--json", action="store_true", help="print JSON instead of a report")
    args = parser.parse_args()

    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    combined = []
    for raw in args.replays:
        path = Path(raw)
        data = analyse(path)
        combined.append(data)
        episode = path.stem.replace("episode-", "").replace("-replay", "")
        if out:
            (out / f"{episode}.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            (out / f"{episode}.md").write_text(render(data), encoding="utf-8")
        if not args.json:
            print(render(data))
            print()

    if out and combined:
        index = [
            "# Episode results",
            "",
            "Extracted from the replays Kaggle serves for submission 56322264. The replays",
            "themselves are not committed — six megabytes each — only what they decide.",
            "",
            "| episode | kind | P0 reward | P1 reward | P0 quadrants | P1 quadrants |",
            "|---|---|---|---|---|---|",
        ]
        kinds = {"110329561": "validation", "110330784": "public"}
        for data in combined:
            episode = Path(data["file"]).stem.replace("episode-", "").replace("-replay", "")
            players = data.get("players") or []
            p0 = players[0] if len(players) > 0 else {}
            p1 = players[1] if len(players) > 1 else {}
            index.append(
                f"| [{episode}]({episode}.md) | {kinds.get(episode, 'unknown')} | {p0.get('reward')} | "
                f"{p1.get('reward')} | {p0.get('quadrants')} | {p1.get('quadrants')} |"
            )
        index.append("")
        (out / "README.md").write_text("\n".join(index), encoding="utf-8")

    if args.json:
        print(json.dumps(combined, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
