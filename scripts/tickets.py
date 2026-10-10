"""Mirror plan tasks into GitHub milestones and issues of the private dev repo.

    uv run python scripts/tickets.py sync [--dry-run] [--repo OWNER/NAME]

Plans (`docs/plans/plan-NN-*.md`) stay the source of truth: one milestone per plan, one issue per
`### Task N:` heading, labels from the task's `**Labels:**` line plus `plan:PNN`. The issue body
carries the plan goal, the acceptance line and the full task text (owner decision 2026-10-10:
plans may live on GitHub). Re-running is idempotent: issues are matched by their key prefix
(`P02.T3`) and updated when the task text changed. A task heading ending in `(done)` closes its
issue.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLANS_DIR = ROOT / "docs" / "plans"
DEFAULT_REPO = "aliildan/country-code-decide-dev"
FOOTER = "_Mirrored by `scripts/tickets.py` from the plan. Edit the plan, then re-sync._"

LABEL_COLORS = {
    "area:": "1d76db",
    "type:": "5319e7",
    "plan:": "c5def5",
    "user-action": "d93f0b",
}

H1 = re.compile(r"^# Plan (\d+) — (.+)$", re.M)
TASK = re.compile(r"^### Task (\d+): (.+)$", re.M)
DONE = re.compile(r"\s*\(done\)\s*$")
SECTION_END = re.compile(r"^---\s*$|^## ", re.M)


@dataclass
class Task:
    key: str
    title: str
    labels: list[str]
    acceptance: str
    number: int
    block: str
    done: bool = False


@dataclass
class Plan:
    number: int
    title: str
    milestone: str
    goal: str
    path: Path
    tasks: list[Task] = field(default_factory=list)


def _field(block: str, name: str) -> str:
    m = re.search(rf"^\*\*{name}:\*\*\s*(.+)$", block, re.M)
    return m.group(1).strip() if m else ""


def _fence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of fenced code blocks (``` or ````), so headings inside them are ignored."""
    spans, start, fence = [], 0, ""
    for m in re.finditer(r"^(`{3,})[^\n]*$", text, re.M):
        if not fence:
            start, fence = m.start(), m.group(1)
        elif m.group(1) == fence:
            spans.append((start, m.end()))
            fence = ""
    return spans


def parse_plan(text: str, path: Path) -> Plan:
    h1 = H1.search(text)
    if not h1:
        raise ValueError(f"{path}: missing '# Plan NN — Title' heading")
    number = int(h1.group(1))
    title = h1.group(2).strip()
    short = re.split(r"[:(]", title, maxsplit=1)[0].strip()
    plan = Plan(number, title, f"P{number:02d} {short}", _field(text, "Goal"), path)
    fences = _fence_spans(text)
    heads = [m for m in TASK.finditer(text) if not any(a <= m.start() < b for a, b in fences)]
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        block = text[m.start() : end]
        stop = SECTION_END.search(block, pos=len(m.group(0)))
        if stop:
            block = block[: stop.start()]
        n = int(m.group(1))
        raw_title = m.group(2).strip()
        key = f"P{number:02d}.T{n}"
        labels = [s.strip() for s in _field(block, "Labels").split(",") if s.strip()]
        plan.tasks.append(
            Task(
                key=key,
                title=f"{key} {DONE.sub('', raw_title)}",
                labels=[*labels, f"plan:P{number:02d}"],
                acceptance=_field(block, "Acceptance"),
                number=n,
                block=block.strip(),
                done=bool(DONE.search(raw_title)),
            )
        )
    return plan


def issue_body(plan: Plan, task: Task) -> str:
    rel = plan.path.relative_to(ROOT) if plan.path.is_absolute() else plan.path
    return (
        f"**Plan:** `{rel}` (Plan {plan.number:02d} — {plan.title}), Task {task.number}\n"
        f"**Plan goal:** {plan.goal}\n\n"
        f"**Acceptance:** {task.acceptance or '(see task)'}\n\n"
        f"<details><summary>Task text</summary>\n\n{task.block}\n\n</details>\n\n---\n{FOOTER}\n"
    )


class Gh:
    def __init__(self, repo: str, dry_run: bool) -> None:
        self.repo = repo
        self.dry_run = dry_run

    def run(self, *args: str, mutate: bool = True) -> str:
        if mutate and self.dry_run:
            print("DRY", "gh", *args[:4])
            return ""
        res = subprocess.run(["gh", *args], check=True, capture_output=True, text=True)
        return res.stdout

    def ensure_label(self, name: str) -> None:
        color = next((c for p, c in LABEL_COLORS.items() if name.startswith(p)), "ededed")
        self.run("label", "create", name, "--repo", self.repo, "--color", color, "--force")

    def milestones(self) -> dict[str, int]:
        out = self.run("api", f"repos/{self.repo}/milestones?state=all&per_page=100", mutate=False)
        return {m["title"]: m["number"] for m in json.loads(out or "[]")}

    def ensure_milestone(self, title: str, description: str, existing: dict[str, int]) -> None:
        if title in existing:
            return
        self.run(
            "api",
            f"repos/{self.repo}/milestones",
            "-f",
            f"title={title}",
            "-f",
            f"description={description}",
        )
        existing[title] = -1

    def issues(self) -> dict[str, dict]:
        out = self.run(
            "issue",
            "list",
            "--repo",
            self.repo,
            "--state",
            "all",
            "--limit",
            "1000",
            "--json",
            "number,title,state,body",
            mutate=False,
        )
        return {i["title"].split(" ", 1)[0]: i for i in json.loads(out or "[]")}


def sync(repo: str, dry_run: bool) -> int:
    gh = Gh(repo, dry_run)
    plans = [
        parse_plan(p.read_text(encoding="utf-8"), p) for p in sorted(PLANS_DIR.glob("plan-*.md"))
    ]
    for lab in sorted({lab for p in plans for t in p.tasks for lab in t.labels}):
        gh.ensure_label(lab)
    existing_ms = gh.milestones()
    for p in plans:
        gh.ensure_milestone(p.milestone, p.goal[:250], existing_ms)
    existing = gh.issues()

    created = updated = closed = 0
    for p in plans:
        for t in p.tasks:
            body = issue_body(p, t)
            issue = existing.get(t.key)
            if issue is None:
                args = ["issue", "create", "--repo", repo, "--title", t.title, "--body", body]
                args += ["--milestone", p.milestone]
                for lab in t.labels:
                    args += ["--label", lab]
                url = gh.run(*args).strip()
                created += 1
                issue = {"number": url.rsplit("/", 1)[-1] if url else "", "state": "OPEN"}
            elif issue["body"].strip() != body.strip() or issue["title"] != t.title:
                gh.run(
                    "issue",
                    "edit",
                    str(issue["number"]),
                    "--repo",
                    repo,
                    "--body",
                    body,
                    "--title",
                    t.title,
                )
                updated += 1
            if t.done and issue.get("state") == "OPEN" and issue.get("number"):
                gh.run(
                    "issue",
                    "close",
                    str(issue["number"]),
                    "--repo",
                    repo,
                    "--comment",
                    "Done; see the plan's ledger and `docs/implement/` (private).",
                )
                closed += 1

    n_tasks = sum(len(p.tasks) for p in plans)
    print(
        f"tickets: {len(plans)} plans, {n_tasks} tasks; created {created}, updated {updated}, "
        f"closed {closed}{' (dry run)' if dry_run else ''}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sync")
    s.add_argument("--repo", default=DEFAULT_REPO)
    s.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    return sync(args.repo, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
