# Kaggriculture

An agent for the Kaggle **Kaggriculture** competition (closes 2026-09-30, $50,000, 9,388
teams), plus the local harness that measures it.

## Why there is a local harness at all

The official way to run this environment is `pip install kaggle-environments`. That package
requires jax, transformers, open_spiel, pettingzoo and pygame. **pygame has no wheel for
Python 3.14 and does not compile on this machine without SDL2 development headers**, so the
package cannot be installed here.

The environment itself is a single file whose only dependency outside the standard library is
one function. So `sim.py` drives **Kaggle's own unmodified interpreter** directly, and the
interpreter that runs locally is the interpreter that runs on Kaggle:

```
env/kaggriculture.py    sha256 bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e
                        == upstream, byte for byte
```

`sim.py` does not reimplement the game. It reproduces the framework *loop* around the
interpreter — the ten lines of `kaggle-environments/core.py` that decide when a step happens
and what `observation.step` means — and cites each by line number. See `NOTICE`.

## Run it

No dependencies. Python 3.14, standard library only.

```sh
python3 sim.py --agent main.py --opponent starter --seed 1   # one episode
python3 bench.py --agent main.py --seeds 16                  # a distribution
python3 bench.py --agent starter --seeds 16                  # the baseline, for comparison
```

| File | What it is |
|---|---|
| `main.py` | **the submission.** `agent(observation) -> action`, single file, as the competition requires |
| `sim.py` | the runner: Kaggle's interpreter, inside a faithful step loop |
| `bench.py` | the evaluation: both seats, many seeds, distribution not anecdote |
| `env/` | Kaggle's environment and its two documents, unmodified |
| `vendor/` | one function from `kaggle-environments`, so the environment imports unmodified |
| `bench/out/` | measured results, committed |

## What the agent does

Reading `kaggriculture.py:1054-1083`, Kaggle's `starter` plants, waters and harvests on the
one tile its farmer spawns on. **It never moves, never hires, never unlocks land, and never
touches an animal** — one tile of the twenty-five unlocked on turn one.

So the margin is throughput, and throughput has three separate ceilings:

1. **Actions are not the constraint.** Every unit acts every turn, and hands cost `fib(n)` per
   hire-day — 1, 1, 2, 3, 5, 8, 13, 21. But the first **ten** hands of a day cost 143, which
   over 30 days is **4,290 against a 3,000 opening bank**. The first version of this agent
   hired ten a day and finished on **810 versus a do-nothing opponent on 3,000**. Three hands
   cost 4/day and are enough for the starting land.
2. **Land is the constraint, and it is paid once** — 1,000 / 2,000 / 4,000 for NE, SW, SE.
   But buying it on turn one converts the whole working capital into dirt and leaves nothing
   for seed. `LAND_RESERVE` is what stops that.
3. **The market is the real constraint.** `price(inv) = base ± amp·f(|inv - I0|)` with
   `I0 = 10000`, and `above_target` is how far one field's capacity drags the price down.
   For **WHEAT** that curve is a **logarithm** at 0.20; for **CARROT** — what `starter`
   farms — it is a **square root at 0.70**, seven times harsher. That is why this agent
   farms wheat.

## Measured

16 episodes per opponent, alternating seats, seeds 1–16. Reproduce with `bench.py`.

| Agent | vs | mean money | worst | best | mean margin | W-T-L |
|---|---|---|---|---|---|---|
| **ashee (`main.py`)** | `starter` | **3917.2** | 3006 | 4127 | **+376.4** | 15-0-1 |
| **ashee (`main.py`)** | `random` | **3923.8** | 3638 | 4180 | +3917.5 | 16-0-0 |
| **ashee (`main.py`)** | `pass` | **3923.4** | 3006 | 4188 | +923.4 | 16-0-0 |
| kaggle `starter` | `starter` | 3539.4 | 3384 | 3908 | +0.0 | 0-16-0 |
| kaggle `starter` | `random` | 3582.9 | 3419 | 3885 | +3572.2 | 16-0-0 |
| kaggle `starter` | `pass` | 3591.0 | 3419 | 4382 | +591.0 | 16-0-0 |

**Mean over all opponents: 3,921 against the baseline's 3,571 — +350, and 94% head-to-head
against `starter`.** `starter` versus itself is 0-16-0 because it is deterministic: every
game is an exact tie.

One episode was lost (`starter`, worst case 3006). It is left in the table rather than
averaged away.

## What actually happened on Kaggle

It was played. **Submission [56322264](https://www.kaggle.com/competitions/kaggriculture)**, a
`VALIDATION` episode and then a `PUBLIC` one. The public episode is the one that counts:

| | Player 0 — this agent | Player 1 — the opponent |
|---|---|---|
| Final reward | **4,087** | **91,360** |
| Quadrants unlocked | 2 | 3 |
| Structures built | none | **18 pastures + 1 coop** |
| Tiles working | 4 plants | 15 plants + 19 animals |
| Ended with | 41 empty tiles | 6 empty tiles |

**This agent was beaten by 22×, and the local benchmark did not see it coming.** Against
`pass`, `random` and `starter` it won at 3,921 against 3,571 — because all three of those
work the single tile their farmer spawns on. Winning meant being slightly less tiny. The
opponent found by the public episode actually uses its farm: three quadrants, nineteen
animals on pasture, and animals produce **indefinitely** (`first_yield_day` then a fixed
interval for the rest of the season), which is where a 91,360 comes from and a 4,087 cannot.

That is the failure this repository's own documentation keeps warning about — matching the
shape of rigour without the thing itself. `bench.py` measured *something* correctly and
pointed at the wrong thing. A benchmark against opponents that nobody else plays is a
benchmark of the opponent, not of the agent.

**The reported `publicScore` is not the reward.** It read `600.0` an hour ago and `484.7`
now, while the agent's money in the recorded episodes was 3,869 and 4,087. The metric is
documented nowhere in `AGENTS.md` or `README.md` — those files define the *reward* as "the
most money in the bank at the end of the game" and say nothing about how a season becomes a
leaderboard number. **The gap is recorded here as an open question rather than guessed at.**

What is established: the agent submits, runs, and completes a 720-turn season without error,
and finishes with more money than it started. What is not: anything about where it places.

## Two things this harness still cannot tell you

**1. A local mean is not a leaderboard position.** `starter` scores ~3,500 against a passive
opponent while the live leaderboard runs 2,945–3,172. Two agents share one market and one
town demand, so prices fall as both sell. The local numbers are only useful for comparing two
agents *against the same opponent* — which is why an opponent worth beating had to be found
before any of them meant anything.

**2. The validation episode was self-play.** In episode 110329561 both players ended on
exactly **3,869** with identical farms and an identical bank trajectory — the opponent was
this agent. A validation episode that plays the submission against itself cannot reveal that
the submission is 4 tiles tall. It took the *public* episode to show that.

## Licence

**BSD Zero Clause License** (`0BSD`) for everything this project wrote — `main.py`, `sim.py`,
`bench.py`, and this document. Copyright (C) 2026 Ashee Softworks. The licence is five lines, and
the grant is one sentence of it in `LICENSE`:

> Permission to use, copy, modify, and/or distribute this software for any purpose with or
> without fee is hereby granted.

No attribution is required, no changes have to be published, and anyone who takes a copy may
close it, relicense it and submit it as their own.

**It does not cover everything in this repository, and the exception is not this project's to
give.** Everything under `env/` is Kaggle's — the environment, the JSON schema and the four
documents — as is one function under `vendor/`. They remain under **Apache License 2.0, Copyright
2022 Kaggle, Inc.** `env/kaggriculture.py` is a byte-for-byte copy, and its sha256 is in the section
above. The list of those files is in `NOTICE` and is **not repeated here**, because a list kept in
two places drifts — and this was one: the first version of this paragraph named a directory that
does not exist, because it was copied from a version of `NOTICE` that had the same error in it.

**Changed on 2026-09-24.** Until then this repository carried `PROPRIETARY AND CONFIDENTIAL — ALL
RIGHTS RESERVED / NO LICENSE IS GRANTED`. That wording is still in every commit before the change —
a later commit does not unpublish it, and the previous commit on this subject is titled *"legal:
apply the organisation's proprietary licence to this repository"*. An open grant cannot be recalled
either, so every copy taken under 0BSD stays free, for anyone, for good. Both directions are
permanent.

