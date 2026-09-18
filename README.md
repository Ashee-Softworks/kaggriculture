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

## Two things this harness cannot tell you

**1. The leaderboard number is not this number.** `starter` scores ~3,500 locally against a
passive opponent, but the live leaderboard tops out at **3,182.7**. The game is two players
competing for **one shared market** and one shared town demand, so two strong agents crash
each other's prices toward the `PRICE_FLOOR` of 1. Kaggle also runs submissions over many
episodes and pairs them against a pool. **A single local episode against a built-in opponent
is therefore not comparable to a leaderboard score**, and nothing here should be read as a
predicted rank.

What the local numbers do establish is narrower and still useful: the agent beats Kaggle's
own baseline on money, across seats and seeds, by a margin much larger than the seed-to-seed
spread.

**2. It has never been played on Kaggle.** Submitting requires accepting the competition
rules on the website — a legal acceptance by the account holder, and the only thing standing
between this code and a score. The API returns `403` until then, and reports it plainly:

```
$ kaggle competitions download kaggriculture -p data
403 Client Error: Forbidden
```

Nothing in this directory has been submitted, and no score has been claimed.
