"""
A local runner for Kaggriculture that uses Kaggle's own interpreter.

**Why this exists.** The official way to run the environment is
`pip install kaggle-environments`, which pulls in jax, transformers, open_spiel, pettingzoo
and pygame. pygame has no wheel for Python 3.14 and does not compile on this host without
SDL2 development headers, so the package cannot be installed here. The environment itself,
however, is a single file whose only outside dependency is one function — so the interpreter
can be driven directly, and that is what this module does.

**What it does not do.** It does not reimplement the game. `env/kaggriculture.py` is an
unmodified copy of Kaggle's file and `interpreter()` is called directly. What is reproduced
here is the *framework loop* around it, and every rule below is read from
`kaggle-environments/core.py`, cited by line as it was at the time of writing:

  * `step()` builds `action_state[index] = {**state[index], "action": None}` then sets
    `action`                                                      — core.py:275-291
  * `self.state = self.__run_interpreter(action_state, logs)`     — core.py:293
  * after the interpreter returns, `new_state[0].observation.step` is set to
    `0 if self.done else len(self.steps)`                         — core.py:626
  * then, if that step is `>= episodeSteps - 1`, every ACTIVE/INACTIVE agent becomes
    DONE                                                          — core.py:296-299
  * the state is appended to `steps`                              — core.py:301
  * `done` is `all(s.status != "ACTIVE" for s in self.state)`     — core.py:511-513
  * `run()` loops `while not self.done`                           — core.py:326-328

One consequence of that ordering looks like a bug and is not one: on the final call the
interpreter marks the agents DONE, so `self.done` is true when line 626 runs and
`observation.step` is written back as `0`. The recorded final state therefore reads
`step: 0`. Rewards live on the agent states and are unaffected.

  python3 sim.py --agent main.py --opponent starter --seed 7
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV_DIR = HERE / "env"
VENDOR = HERE / "vendor"


class Struct(dict):
    """
    A mapping that also answers to attribute access, which both the interpreter and
    `resolve_episode_seed` require (`cfg.episodeSteps`, `obs0.farms`, `setattr(config, ...)`).
    """

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name: str, value) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as error:
            raise AttributeError(name) from error


def load_environment():
    """
    Import Kaggle's environment file without editing it.

    `vendor/` is put on the path first so that the file's own
    `from kaggle_environments.utils import resolve_episode_seed` resolves to the vendored
    copy of that one function.
    """
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))

    spec = importlib.util.spec_from_file_location("kaggriculture", ENV_DIR / "kaggriculture.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {ENV_DIR / 'kaggriculture.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["kaggriculture"] = module
    spec.loader.exec_module(module)
    return module


def default_configuration() -> Struct:
    """
    Every default the environment's own specification declares, and nothing invented.

    The specification mixes two shapes: some keys carry a full JSON-schema object with a
    `default` inside it (`boardSize`, `startingMoney`), and others are a bare scalar
    (`episodeSteps: 720`, `actTimeout: 1`). Both are read, because reading only the first
    shape silently loses `episodeSteps` and the episode then has no length.
    """
    specification = json.loads((ENV_DIR / "kaggriculture.json").read_text(encoding="utf-8"))
    config = Struct()
    for key, value in specification.get("configuration", {}).items():
        if isinstance(value, dict):
            if "default" in value:
                config[key] = value["default"]
        else:
            config[key] = value
    config.setdefault("runTimeout", 3600)
    config.setdefault("actTimeout", 1)
    return config


class Environment:
    """
    The part of `kaggle_environments.Environment` that Kaggriculture actually touches.

    Three attributes are read by the interpreter: `configuration`, `info` and `done`. They
    are all here. Everything else is the step loop.
    """

    def __init__(self, configuration: Struct | None = None):
        self.configuration = configuration if configuration is not None else default_configuration()
        self.info: dict = {}
        self.module = load_environment()
        self.interpreter = self.module.interpreter
        self.steps: list = []
        self.state: list = []

    @property
    def done(self) -> bool:
        """core.py:511-513 — done when no agent is still ACTIVE."""
        return all(s.status != "ACTIVE" for s in self.state)

    def reset(self, num_agents: int = 2) -> None:
        """Build the initial state, which is the state the first `step()` is handed."""
        self.state = [
            Struct(
                action=None,
                status="ACTIVE",
                reward=0,
                info={},
                observation=Struct(step=0, remainingOverageTime=60, player=index),
            )
            for index in range(num_agents)
        ]
        self.steps = [self.state]
        self.info = {}

    def step(self, actions: list) -> list:
        """core.py:256-305."""
        if self.done:
            raise RuntimeError("environment is done")
        if len(actions) != len(self.state):
            raise ValueError(f"{len(self.state)} actions required, got {len(actions)}")

        action_state = []
        for index, action in enumerate(actions):
            entry = {**self.state[index], "action": None}
            # The framework validates the action against a schema and marks INVALID on
            # failure. The interpreter already treats a non-dict action as an empty one
            # (line 914: `s.action if isinstance(s.action, dict) else {}`), so an invalid
            # action is passed through as the interpreter's own no-op rather than being
            # rejected a second time here.
            entry["action"] = action if isinstance(action, dict) else {}
            action_state.append(entry)

        self.state = self._run_interpreter(action_state)

        # core.py:296-299
        if self.state[0].observation.step >= self.configuration.episodeSteps - 1:
            for agent_state in self.state:
                if agent_state.status in ("ACTIVE", "INACTIVE"):
                    agent_state.status = "DONE"

        self.steps.append(self.state)
        return self.state

    def _run_interpreter(self, action_state: list) -> list:
        """core.py:621-626."""
        args = [[Struct(entry) for entry in action_state], self]
        new_state = self.interpreter(*args[: self.interpreter.__code__.co_argcount])
        new_state = [entry if isinstance(entry, Struct) else Struct(entry) for entry in new_state]
        # core.py:626 — read carefully before changing: `self.done` here refers to the state
        # the interpreter has just returned, not the state it was handed.
        self.state = new_state
        new_state[0].observation.step = 0 if self.done else len(self.steps)
        return new_state

    def run(self, agents: list, seed: int | None = None) -> list:
        """core.py:307-334, with the wall-clock guard replaced by the step counter."""
        if seed is not None:
            self.configuration["seed"] = seed
        self.reset(len(agents))

        while not self.done and len(self.steps) < int(self.configuration.episodeSteps):
            observations = [copy.deepcopy(self.state[i].observation) for i in range(len(agents))]
            actions = [agent(observations[i]) for i, agent in enumerate(agents)]
            started = time.perf_counter()
            self.step(actions)
            # The framework charges each agent's elapsed time against a per-agent budget of
            # `actTimeout` seconds. It is measured and recorded rather than enforced: a
            # local run has no reason to abort on a budget the leaderboard will judge.
            self.last_act_seconds = time.perf_counter() - started

        return self.steps

    def rewards(self) -> list[float]:
        """Final money per player, as the interpreter wrote it on the last step."""
        final = self.steps[-1]
        return [float(entry.reward) for entry in final]


# ------------------------------------------------------------------ agents

BUILT_INS = ("pass", "random", "starter")


def resolve_agent(spec: str):
    """
    Turn a name or a path into a callable.

    A name resolves to one of the environment module's own three agents, so a comparison
    against `starter` is a comparison against Kaggle's baseline and not against a
    reimplementation of it. A path is loaded as a file with an `agent` function, which is
    the shape the competition accepts.
    """
    if spec in BUILT_INS:
        module = load_environment()
        return {"pass": module.pass_agent, "random": module.random_agent, "starter": module.starter_agent}[spec]

    path = Path(spec)
    if not path.is_absolute():
        candidate = (HERE / spec) if (HERE / spec).exists() else Path.cwd() / spec
        path = candidate
    if not path.exists():
        raise FileNotFoundError(f"no agent at {path}")

    module_name = f"agent_{path.stem}"
    load_spec = importlib.util.spec_from_file_location(module_name, path)
    if load_spec is None or load_spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(load_spec)
    sys.modules[module_name] = module
    load_spec.loader.exec_module(module)

    if not hasattr(module, "agent"):
        raise AttributeError(f"{path} defines no `agent` function")
    return module.agent


# ------------------------------------------------------------------ command line


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run one Kaggriculture episode locally.")
    parser.add_argument("--agent", default="main.py", help="a built-in name, or a path to a .py with agent()")
    parser.add_argument("--opponent", default="starter", help="a built-in name, or a path to a .py")
    parser.add_argument("--seed", type=int, default=None, help="episode seed; omitted means random")
    parser.add_argument("--steps", type=int, default=None, help="cap the episode early")
    parser.add_argument("--swap", action="store_true", help="play the agent as player 1 instead of 0")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    args = parser.parse_args()

    mine = resolve_agent(args.agent)
    theirs = resolve_agent(args.opponent)

    players = [theirs, mine] if args.swap else [mine, theirs]
    environment = Environment()
    started = time.perf_counter()
    environment.run(players, seed=args.seed)
    elapsed = time.perf_counter() - started

    rewards = environment.rewards()
    my_index = 1 if args.swap else 0
    result = {
        "seed": args.seed,
        "agent": args.agent,
        "opponent": args.opponent,
        "played_as": my_index,
        "turns": len(environment.steps),
        "rewards": rewards,
        "mine": rewards[my_index],
        "theirs": rewards[1 - my_index],
        "margin": rewards[my_index] - rewards[1 - my_index],
        "wall_seconds": round(elapsed, 2),
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"  agent    {args.agent} as player {my_index}")
        print(f"  opponent {args.opponent}")
        print(f"  seed     {args.seed}")
        print(f"  turns    {result['turns']}")
        print(f"  mine     {result['mine']:.0f}")
        print(f"  theirs   {result['theirs']:.0f}")
        print(f"  margin   {result['margin']:+.0f}")
        print(f"  wall     {result['wall_seconds']}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
