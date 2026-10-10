import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("tk", Path("scripts/tickets.py"))
assert _spec and _spec.loader
tk = importlib.util.module_from_spec(_spec)
sys.modules["tk"] = tk
_spec.loader.exec_module(tk)

PLAN = """# Plan 07 — Example plan

**Goal:** Do the example.

### Task 1: First thing

**Labels:** area:x, type:feature
**Acceptance:** It works.

- [ ] **Step 1: Write the failing test**

### Task 2: Second thing (done)

**Labels:** area:y
**Acceptance:** Done already.
"""


def test_parse_plan():
    p = tk.parse_plan(PLAN, Path("docs/plans/plan-07-example.md"))
    assert p.milestone == "P07 Example plan" and p.goal == "Do the example."
    assert [t.key for t in p.tasks] == ["P07.T1", "P07.T2"]
    assert p.tasks[0].labels == ["area:x", "type:feature", "plan:P07"]
    assert p.tasks[0].acceptance == "It works."
    assert p.tasks[1].done and p.tasks[1].title == "P07.T2 Second thing"


def test_issue_body_carries_task_text():
    p = tk.parse_plan(PLAN, Path("docs/plans/plan-07-example.md"))
    body = tk.issue_body(p, p.tasks[0])
    assert "Do the example." in body and "**Step 1: Write the failing test**" in body
    assert "plan-07-example.md" in body


def test_headings_inside_code_fences_are_not_tasks():
    text = PLAN + '\n```python\nX = """\n### Task 9: Not a task\n"""\n```\n'
    p = tk.parse_plan(text, Path("docs/plans/plan-07-example.md"))
    assert [t.key for t in p.tasks] == ["P07.T1", "P07.T2"]
