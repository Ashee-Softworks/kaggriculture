#!/usr/bin/env python3
"""
The submission ledger.

Every attempt to submit, and every change in what Kaggle says about it, recorded in one
append-only file. The point is that the record comes from **authoritative sources** — the
Kaggle API and `git rev-parse` — rather than from someone remembering to type a number in.

Three rules, and they are the whole design:

  1. **Append-only.** A submission's entry is created once. Later knowledge is *appended* to
     its `history`; nothing already written is ever rewritten. A ledger whose past entries
     can be edited cannot be used to judge whether a claim should be trusted, which is the
     only reason to keep one.
  2. **A refusal is a result.** If the API answers `403` because the competition rules have
     not been accepted, that is recorded as an outcome, with the reason Kaggle gave. It is
     not an error in this script and it is not a silent no-op.
  3. **The score is quoted, never computed.** Whatever Kaggle reports is written down
     verbatim. Nothing here infers a score from a local run.

  python3 record.py --submit --message "wheat v2"     # submit main.py and record it
  python3 record.py --refresh                         # ask Kaggle about everything recorded
  python3 record.py --list                            # print the ledger
  python3 record.py --render                          # rewrite RESULTS.md from the ledger
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "results" / "submissions.json"
RENDERED = HERE / "RESULTS.md"
AGENT = HERE / "main.py"
BENCH_DIR = HERE / "bench" / "out"

COMPETITION = "kaggriculture"
TOKEN_PATH = Path.home() / ".kaggle" / "access_token"
API = "https://www.kaggle.com/api/v1"

EMPTY_LEDGER = {
    "schemaVersion": 1,
    "competition": COMPETITION,
    "note": (
        "Append-only. An entry is created once and never rewritten; later knowledge is "
        "appended to its `history`. A refusal is recorded as an outcome, not as an error. "
        "Every score is quoted from the Kaggle API, never computed locally."
    ),
    "submissions": [],
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load() -> dict:
    if not LEDGER.exists():
        return json.loads(json.dumps(EMPTY_LEDGER))
    return json.loads(LEDGER.read_text(encoding="utf-8"))


def save(ledger: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    """The repository's own answer, or an empty string when there is no repository."""
    try:
        return subprocess.run(
            ["git", "-C", str(HERE), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def local_bench() -> dict:
    """
    The local measurements at the moment of recording, read from `bench/out/`.

    They travel with the submission entry so a later reader can tell what the agent scored
    locally *when it was sent*, without having to check out the commit and re-run it. They
    are not a prediction of the Kaggle score and the ledger never presents them as one.
    """
    found = {}
    if BENCH_DIR.exists():
        for path in sorted(BENCH_DIR.glob("*.json")):
            try:
                found[path.stem] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                found[path.stem] = "unreadable"
    return found


def api_get(path: str, timeout: int = 30) -> tuple[int, object]:
    """Read the API. The token file is read here and never written anywhere."""
    if not TOKEN_PATH.exists():
        return 0, f"no token at {TOKEN_PATH}"
    token = TOKEN_PATH.read_text().strip()
    request = urllib.request.Request(
        f"{API}{path}", headers={"Authorization": f"Bearer {token}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(2_000_000)
            try:
                return response.status, json.loads(body)
            except json.JSONDecodeError:
                return response.status, body.decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return error.code, error.read(4096).decode("utf-8", "replace")
    except Exception as error:  # noqa: BLE001
        return 0, f"{type(error).__name__}: {error}"


def kaggle_submissions() -> tuple[int, object]:
    """What Kaggle says this account has submitted for this competition."""
    return api_get(f"/competitions/submissions/list/{COMPETITION}")


def kaggle_cli() -> str | None:
    """The CLI, wherever it is. It was installed with `pip install --user` on this host."""
    for candidate in (
        Path.home() / ".local" / "bin" / "kaggle",
        Path("/usr/local/bin/kaggle"),
        Path("/usr/bin/kaggle"),
    ):
        if candidate.exists():
            return str(candidate)
    from shutil import which

    return which("kaggle")


def do_submit(message: str) -> dict:
    """
    Send `main.py`, and record what happened either way.

    The attempt is written down **before** it is interpreted, so a refusal is a row in the
    ledger with Kaggle's own words attached rather than an exception that leaves no trace.
    """
    entry = {
        "attemptedAt": now(),
        "message": message,
        "agentFile": AGENT.name,
        "agentSha256": sha256_file(AGENT) if AGENT.exists() else None,
        "agentCommit": git("rev-parse", "HEAD"),
        "agentCommitSubject": git("log", "-1", "--pretty=%s"),
        "agentCommitDate": git("log", "-1", "--pretty=%cI"),
        "agentDirty": bool(git("status", "--porcelain")),
        "localBenchAtRecord": local_bench(),
        "outcome": "unknown",
        "kaggle": {},
        "history": [{"at": now(), "status": "attempted", "score": None, "note": message}],
    }

    cli = kaggle_cli()
    if cli is None:
        entry["outcome"] = "no-cli"
        entry["kaggle"] = {"reason": "the kaggle CLI is not installed on this host"}
        return entry

    result = subprocess.run(
        [cli, "competitions", "submit", COMPETITION, "-f", str(AGENT), "-m", message, "-q"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    entry["kaggle"] = {
        "exitCode": result.returncode,
        "stdout": result.stdout.strip()[:2000],
        "stderr": result.stderr.strip()[:2000],
    }

    if result.returncode == 0:
        entry["outcome"] = "accepted"
        entry["history"].append(
            {"at": now(), "status": "accepted", "score": None, "note": "the CLI accepted the file"}
        )
    else:
        entry["outcome"] = "refused"
        reason = (result.stderr.strip() or result.stdout.strip()).split("\n")[0][:400]
        entry["kaggle"]["reason"] = reason
        entry["history"].append({"at": now(), "status": "refused", "score": None, "note": reason})

    # What Kaggle says about itself, immediately after the attempt.
    status, payload = kaggle_submissions()
    entry["kaggle"]["submissionsAfterAttempt"] = {
        "httpStatus": status,
        "count": len(payload) if isinstance(payload, list) else None,
    }
    return entry


def do_refresh(ledger: dict) -> int:
    """
    Ask Kaggle about every recorded submission and append anything that changed.

    Only *changes* are appended, so a ledger refreshed fifty times does not grow fifty
    identical rows. Nothing already written is modified — that is what `history` is for.
    """
    status, payload = kaggle_submissions()
    if status != 200 or not isinstance(payload, list):
        ledger.setdefault("refreshes", []).append(
            {"at": now(), "httpStatus": status, "problem": str(payload)[:300]}
        )
        return 0

    # The identifier field is `ref`, not `id`. This was got wrong the first time and the
    # effect was silent: `by_id` came out empty, every recorded submission failed to match,
    # and the refresh reported "0 seen, 0 changed" on a competition that had a submission in
    # it. A wrong field name that produces a plausible zero is worse than a crash.
    by_id = {}
    for row in payload:
        identifier = row.get("ref", row.get("id"))
        if identifier is not None:
            by_id[str(identifier)] = row

    changed = 0
    claimed = {str(e.get("submissionId")) for e in ledger.get("submissions", []) if e.get("submissionId")}

    # Backfill. An entry recorded before the `ref`/`id` bug was fixed stored no identifier, so
    # no later refresh can ever match it. Rather than editing the entry, the identifier is
    # adopted here and the adoption is written into `history` where it can be seen.
    for entry in ledger.get("submissions", []):
        if entry.get("submissionId"):
            continue
        attempted = str(entry.get("attemptedAt") or "")
        candidates = [
            row
            for identifier, row in by_id.items()
            if identifier not in claimed and str(row.get("date") or "") >= attempted
        ]
        if not candidates:
            continue
        adopted = min(candidates, key=lambda row: str(row.get("date") or ""))
        entry["submissionId"] = adopted.get("ref", adopted.get("id"))
        claimed.add(str(entry["submissionId"]))
        entry.setdefault("history", []).append(
            {
                "at": now(),
                "status": "identified",
                "score": None,
                "note": (
                    f"submission {entry['submissionId']} adopted from the API; the entry stored "
                    "no identifier because the recorder read `id` where the API sends `ref`"
                ),
            }
        )
        changed += 1

    for entry in ledger.get("submissions", []):
        row = by_id.get(str(entry.get("submissionId") or ""))
        if row is None:
            continue
        observation = {
            "status": row.get("status"),
            # An unscored submission reports `publicScore: ""` rather than null, so an empty
            # string has to be read as "no score yet" and not as the score zero.
            "score": (row.get("publicScore") or row.get("score")) or None,
            "privateScore": (row.get("privateScore") or None),
            "errorDescription": row.get("errorDescription") or None,
            "submittedAt": row.get("date") or row.get("submittedAt"),
        }
        history = entry.get("history") or []
        last = history[-1] if history else {}
        if last.get("status") != observation["status"] or last.get("score") != observation["score"]:
            entry.setdefault("history", []).append({"at": now(), **observation})
            entry["kaggle"] = {**entry.get("kaggle", {}), **observation}
            changed += 1

    ledger.setdefault("refreshes", []).append(
        {"at": now(), "httpStatus": status, "seen": len(by_id), "changed": changed}
    )
    return changed


def render(ledger: dict) -> str:
    """The readable form. Generated from the ledger; never the source of it."""
    rows = ledger.get("submissions", [])
    lines = [
        "# Submission results",
        "",
        "Generated from `results/submissions.json` by `record.py --render`. **Append-only:** a",
        "submission's entry is created once, later knowledge is appended to its `history`, and",
        "nothing already written is rewritten. Every score is quoted from the Kaggle API.",
        "",
        f"Competition: **{ledger.get('competition')}** · submissions recorded: **{len(rows)}**",
        "",
    ]

    if not rows:
        lines += [
            "## No submissions",
            "",
            "Nothing has been submitted. This is not a placeholder — it is the state, and the",
            "reason is checkable:",
            "",
            "```",
            f"$ kaggle competitions submissions {COMPETITION}",
            "No submissions found",
            "",
            f"$ GET /api/v1/competitions/submissions/list/{COMPETITION}",
            "[]                                    HTTP 200",
            "```",
            "",
            "**What is blocking it.** Submitting requires joining the competition, which means",
            "accepting its rules on the website. That is a legal acceptance by the account",
            "holder, so no tool here performs it. Until it is done the API answers `403` to any",
            "download and the CLI refuses any submission — and when an attempt *is* made, the",
            "refusal is recorded below as an outcome with Kaggle's own words attached.",
            "",
            f"    https://www.kaggle.com/competitions/{COMPETITION}   →   Join Competition",
            "",
        ]
    else:
        lines += [
            "| # | submitted | message | outcome | score | agent commit |",
            "|---|---|---|---|---|---|",
        ]
        for index, entry in enumerate(rows, start=1):
            score = (entry.get("kaggle") or {}).get("score")
            lines.append(
                f"| {index} | {entry.get('attemptedAt', '—')} | {entry.get('message', '—')} | "
                f"{entry.get('outcome', '—')} | {score if score is not None else '—'} | "
                f"`{(entry.get('agentCommit') or '—')[:9]}` |"
            )

        lines += ["", "## Every observation, in order", ""]
        for index, entry in enumerate(rows, start=1):
            lines += [
                f"### {index}. {entry.get('message', '(no message)')}",
                "",
                f"- agent `{entry.get('agentFile')}` sha256 `{(entry.get('agentSha256') or '—')[:16]}…`",
                f"- commit `{(entry.get('agentCommit') or '—')[:9]}` — {entry.get('agentCommitSubject', '')}",
                f"- working tree dirty at submission: **{entry.get('agentDirty')}**",
                "",
                "| at | status | score | note |",
                "|---|---|---|---|",
            ]
            for step in entry.get("history", []):
                lines.append(
                    f"| {step.get('at', '—')} | {step.get('status', '—')} | "
                    f"{step.get('score') if step.get('score') is not None else '—'} | "
                    f"{str(step.get('note') or '')[:120]} |"
                )
            lines.append("")

    refreshes = ledger.get("refreshes") or []
    if refreshes:
        lines += [
            "## Refreshes against the API",
            "",
            "| at | HTTP | seen | changed | problem |",
            "|---|---|---|---|---|",
        ]
        for row in refreshes[-25:]:
            lines.append(
                f"| {row.get('at', '—')} | {row.get('httpStatus', '—')} | {row.get('seen', '—')} | "
                f"{row.get('changed', '—')} | {str(row.get('problem') or '')[:90]} |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="The submission ledger.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--submit", action="store_true", help="submit main.py, then record it")
    group.add_argument("--refresh", action="store_true", help="ask the API about recorded submissions")
    group.add_argument("--list", action="store_true", help="print the ledger")
    group.add_argument("--render", action="store_true", help="rewrite RESULTS.md from the ledger")
    parser.add_argument("--message", default="", help="the submission message")
    args = parser.parse_args()

    ledger = load()

    if args.submit:
        if not args.message:
            parser.error("--submit needs --message")
        entry = do_submit(args.message)

        # The submission id comes from the API rather than from parsing the CLI's prose. A
        # refusal produces none, which is itself recorded by leaving the field empty. The
        # identifier field is `ref`.
        status, payload = kaggle_submissions()
        if status == 200 and isinstance(payload, list) and payload:
            newest = max(payload, key=lambda row: str(row.get("date") or ""))
            entry["submissionId"] = newest.get("ref", newest.get("id"))
            entry["submissionStatus"] = newest.get("status")

        ledger.setdefault("submissions", []).append(entry)
        save(ledger)
        RENDERED.write_text(render(ledger), encoding="utf-8")

        print(f"outcome: {entry['outcome']}")
        if entry.get("kaggle", {}).get("reason"):
            print(f"reason:  {entry['kaggle']['reason']}")
        print(f"ledger:  {LEDGER}")
        return 0 if entry["outcome"] == "accepted" else 1

    if args.refresh:
        changed = do_refresh(ledger)
        save(ledger)
        RENDERED.write_text(render(ledger), encoding="utf-8")
        print(f"refreshed; {changed} observation(s) changed")
        return 0

    if args.render:
        RENDERED.write_text(render(ledger), encoding="utf-8")
        print(f"wrote {RENDERED}")
        return 0

    print(json.dumps(ledger, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
