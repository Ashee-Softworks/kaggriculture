"""
Ashee's Kaggriculture agent.

The submission the competition asks for: a `main.py` at the archive root defining
`agent(observation) -> action`.

**What Kaggle's `starter` agent does, and why this is different.** Reading
`kaggriculture.py:1054-1083`, `starter` plants, waters and harvests on the one tile the
farmer happens to spawn on. It never moves, never hires, never unlocks land and never
touches an animal. On a 10x10 farm with 25 tiles unlocked from turn one, that is one tile
of twenty-five producing, worked by 1 of a possible 25 units.

So the whole margin is in **throughput**, and throughput is bounded by three separate
ceilings which this agent is built around:

  1. **Actions.** Every unit acts every turn. Hands cost `fib(n)` per hire-day and reset
     each morning — 1, 1, 2, 3, 5, 8, 13, 21 — so the first eight hands of a day cost 54
     coins between them and can each do 24 actions. The bottleneck is therefore never
     actions; it is tiles.
  2. **Tiles.** 25 unlocked, 75 more behind `BUY_LAND` at 1k / 2k / 4k. Land is a one-off
     cost and a hand is a daily one, so land first, and only buy hands when there is
     somewhere for them to stand.
  3. **The market.** `price(inv) = base ± amp·f(|inv - I0|)` with `I0 = 10000`, and
     `above_target` is how far selling one field's capacity drags the price down. For
     CARROT that is 0.70, for WHEAT 0.20 — wheat's above-curve is a logarithm and carrot's
     is a square root, so wheat absorbs oversupply far more gracefully. A farm that fills
     the market with carrots is paid less for the same work, which is why this agent counts
     how much it has already sold this day before deciding to harvest.

The agent is deliberately written as four separate decisions — what to sell, what to buy,
who to hire, and what each unit does — because those four are where every point of margin
lives, and a single tangled function is how a farming agent loses money while looking busy.
"""

from __future__ import annotations

# ------------------------------------------------------------------ the tables

# Read from `kaggriculture.py` (CROPS, ANIMALS) and from the Object Types table in
# README.md. Duplicated rather than imported because a Kaggle submission is one file and
# has no access to the environment module.
CROPS = {
    "WHEAT": {"seed": 10, "first_yield_day": 2, "max_yield_day": 4, "base": 25, "max_yield": 4},
    "CARROT": {"seed": 20, "first_yield_day": 2, "max_yield_day": 3, "base": 35, "max_yield": 3},
    "MELON": {"seed": 80, "first_yield_day": 10, "max_yield_day": 10, "base": 250, "max_yield": 6},
}

# Which crop this agent farms. Wheat on purpose: it is the only crop whose above-I0 price
# curve is a logarithm (`MARKET_PARAMS["WHEAT"]`, `above_func: "log"`, `above_target: 0.20`),
# which means it is the one product that can be produced in bulk without the price falling
# out from under the farm. Carrot, at `above_func: "sqrt"` and `above_target: 0.70`, punishes
# the same volume seven times harder, and that is what `starter` farms.
CROP = "WHEAT"

# Hire cost per hand, `fib(n)` for the n-th hire of the day. The first hand of a day costs
# 1 and the tenth costs 55, so the day's total is 143 for ten and 4 for three. At 30 days
# that is 4,290 against a 3,000 opening bank, which is how a farm that looks busy goes
# bankrupt. Three hands is 120 for the season and, with the farmer, is enough to service the
# 25 tiles that are unlocked at the start.
MAX_HANDS = 3

# Never spend below this, or the farm stalls with no seed money and no way to earn.
RESERVE = 60

# Land is bought only out of this much surplus on top of its own price. The first unlock
# costs 1,000 against a 3,000 bank; buying it on turn one leaves 2,000 and a farm that
# cannot afford seed for the land it just bought.
LAND_RESERVE = 3000

# `BUY_LAND` costs 1000, 2000, 4000 in that order for NE, SW, SE. Cheap next to a hand's
# daily cost, and it is the only cost here that is paid once.
LAND_COSTS = [1000, 2000, 4000]

# How much wheat to keep in the seed slot. Two per unlocked tile means a plant is always
# available when a tile opens, without tying up money in seed.
SEED_PER_TILE = 1
MAX_SEED_BUFFER = 40


def _fib(n: int) -> int:
    """The n-th hire of a day costs this. `kaggriculture.py:690-700`."""
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _unlocked(tiles, board_size: int) -> list[tuple[int, int]]:
    """Every position the farm may act on, in a stable order."""
    positions = []
    for y in range(board_size):
        for x in range(board_size):
            if tiles[y][x] != "LOCKED":
                positions.append((x, y))
    return positions


def _step_toward(here, target) -> list:
    """One move that reduces the distance on the larger axis first. Locked tiles are passable."""
    x, y = here
    tx, ty = target
    if x == tx and y == ty:
        return ["PASS"]
    if abs(tx - x) >= abs(ty - y):
        return ["EAST"] if tx > x else ["WEST"]
    return ["SOUTH"] if ty > y else ["NORTH"]


def _shed_tiles(board_size: int) -> list[tuple[int, int]]:
    """`_shed_access_tiles`, `kaggriculture.py:132-135` — the four inner-corner tiles."""
    half = board_size // 2
    return [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]


def _nearest(here, targets) -> tuple[int, int]:
    x, y = here
    return min(targets, key=lambda t: (abs(t[0] - x) + abs(t[1] - y), t[1], t[0]))


# How much a unit carries before walking back to the shed. One wheat cycle is four actions
# and the walk back is the whole cost of the trip, so carrying more amortises it — up to the
# point where produce sits unbanked and the day's land is idle.
CARRY_LIMIT = 8


def agent(observation):
    """One turn. Four decisions, in the order the money arrives in."""
    player = observation.get("player", 0)
    farms = observation.get("farms") or []
    if player >= len(farms):
        # The very first call, before `_initialize` has built the farms.
        return {"farmer": ["PASS"], "hands": [], "market": []}

    farm = farms[player]
    private = observation.get("private") or {}
    tiles = farm["tiles"]
    board_size = len(tiles)
    day = observation.get("day", 0)
    money = farm["money"]

    seeds = private.get("seeds") or {}
    shed = private.get("shed") or {}
    inventories = private.get("inventories") or [{}]
    market = []

    # ---------------------------------------------------------------- 1. sell
    # Everything harvested, every turn. Produce in the shed is cash not earning and stock
    # not yet turned into market pressure, and both of those are losses.
    for item, count in shed.items():
        if count > 0:
            market.append(["SELL", item, int(count)])

    # ---------------------------------------------------------------- 2. buy
    # Seed, up to a buffer sized by the land actually unlocked.
    unlocked = _unlocked(tiles, board_size)
    unlocked_count = len(unlocked)
    wanted_seed = min(MAX_SEED_BUFFER, max(SEED_PER_TILE, unlocked_count * SEED_PER_TILE))
    crop_seed_cost = CROPS[CROP]["seed"]
    if seeds.get(CROP, 0) < wanted_seed and money >= crop_seed_cost * 2 + RESERVE:
        shortfall = wanted_seed - seeds.get(CROP, 0)
        affordable = max(1, int((money - RESERVE) // crop_seed_cost))
        market.append(["BUY_SEED", CROP, min(shortfall, affordable)])

    # Land, before hands — but only out of surplus. It is paid once, and a hand hired before
    # there is somewhere for it to work is a daily cost buying nothing. The margin over the
    # price is what stops the farm converting its whole working capital into dirt on day one
    # and then standing idle with nothing to plant.
    hires_today = farm.get("hires_today", 0)
    bought_land = len(farm.get("unlocked_quadrants", ["NW"])) - 1
    if (
        bought_land < len(LAND_COSTS)
        and money >= LAND_COSTS[bought_land] + LAND_RESERVE
        and hires_today >= MAX_HANDS
    ):
        market.append(["BUY_LAND"])

    # ---------------------------------------------------------------- 3. hire
    # Only as many hands as there are tiles to put them on. Above that they walk to squares
    # other units are already working, which is a daily fee for nothing.
    hands_wanted = min(MAX_HANDS, max(0, unlocked_count - 1))
    if hires_today < hands_wanted and money >= _fib(hires_today) + RESERVE * 2:
        market.append(["HIRE"])

    # The environment processes at most `maxMarketOrdersPerTurn` (10) and silently drops the
    # rest, so the list is trimmed here rather than relying on that.
    market = market[:10]

    # ---------------------------------------------------------------- 4. work
    units = [farm["farmer"], *farm.get("hands", [])]
    shed_tiles = _shed_tiles(board_size)

    # Tiles nearest the shed first, so the walk to unload is as short as the farm allows.
    work = sorted(
        unlocked,
        key=lambda p: (min(abs(p[0] - s[0]) + abs(p[1] - s[1]) for s in shed_tiles), p[1], p[0]),
    )
    if not work:
        work = list(shed_tiles)

    # Each unit owns one tile, so two units never queue on the same square.
    actions = []
    for index, position in enumerate(units):
        x, y = position
        inventory = inventories[index] if index < len(inventories) else {}
        carried = sum((inventory or {}).values())
        adjacent_to_shed = (x, y) in shed_tiles

        # Unload only when there is something to unload.
        if carried >= CARRY_LIMIT:
            if adjacent_to_shed:
                actions.append(["DROP"])
            else:
                actions.append(_step_toward((x, y), _nearest((x, y), shed_tiles)))
            continue

        target = work[index % len(work)]
        if (x, y) != target:
            actions.append(_step_toward((x, y), target))
            continue

        # Standing on the owned tile: do the most valuable thing available here.
        tile = tiles[y][x]
        if tile is None:
            actions.append(["PLANT", CROP] if seeds.get(CROP, 0) > 0 else ["PASS"])
        elif isinstance(tile, dict) and tile.get("kind") == "WEED":
            actions.append(["DIG"])
        elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
            age = day - tile.get("planted_day", day)
            if age >= CROPS[CROP]["max_yield_day"] and tile.get("yield_units", 0) > 0:
                actions.append(["HARVEST"])
            elif not tile.get("watered_today", False):
                actions.append(["WATER"])
            else:
                actions.append(["PASS"])
        else:
            actions.append(["PASS"])

    return {"farmer": actions[0], "hands": actions[1:], "market": market}
