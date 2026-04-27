import json
import subprocess
from pathlib import Path

import pytest

from spoke.epistaxis_operator import EpistaxisOperator, EpistaxisOperatorError, main


def _make_epistaxis_root(tmp_path: Path) -> Path:
    root = tmp_path / "epistaxis-worktree"
    root.mkdir()
    (root / ".git").write_text("gitdir: /fake/common.git/worktrees/test\n", encoding="utf-8")
    (root / "policy" / "codex").mkdir(parents=True)
    (root / "policy" / "codex" / "agents.md").write_text("# policy\n", encoding="utf-8")
    (root / "reviews").mkdir()
    (root / "projects" / "spoke").mkdir(parents=True)
    (root / "projects" / "spoke" / "epistaxis.md").write_text(
        "# Spoke Epistaxis\n\n"
        "## Review Notes\n"
        "Full review documents and review tickets live in `reviews/`.\n\n"
        "## Open Questions\n"
        "- placeholder\n",
        encoding="utf-8",
    )
    return root


def _completed(*, stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def test_refuses_inert_main_checkout(tmp_path, monkeypatch):
    root = _make_epistaxis_root(tmp_path)
    monkeypatch.setattr("spoke.epistaxis_operator._INERT_EPISTAXIS_MAIN", root)
    operator = EpistaxisOperator(root, "spoke")

    with pytest.raises(EpistaxisOperatorError, match="inert Epistaxis main checkout"):
        operator.execute_plan([{"op": "git_status"}])


def test_refuses_main_branch(tmp_path, monkeypatch):
    root = _make_epistaxis_root(tmp_path)
    operator = EpistaxisOperator(root, "spoke")

    def _run(cmd, capture_output, text, check):
        assert cmd[:3] == ["git", "-C", str(root)]
        return _completed(stdout="main\n")

    monkeypatch.setattr(subprocess, "run", _run)

    with pytest.raises(EpistaxisOperatorError, match="branch 'main'"):
        operator.execute_plan([{"op": "git_status"}])


def test_refuses_non_epistaxis_checkout(tmp_path):
    root = _make_epistaxis_root(tmp_path)
    (root / "policy" / "codex" / "agents.md").unlink()
    operator = EpistaxisOperator(root, "spoke")

    with pytest.raises(EpistaxisOperatorError, match="refusing to run outside an Epistaxis checkout"):
        operator.execute_plan([{"op": "git_status"}])


def test_write_ticket_and_append_pointer(tmp_path, monkeypatch):
    root = _make_epistaxis_root(tmp_path)
    operator = EpistaxisOperator(root, "spoke")

    def _run(cmd, capture_output, text, check):
        if cmd[-3:] == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return _completed(stdout="codex/spoke-demo\n")
        raise AssertionError(f"unexpected git command: {cmd}")

    monkeypatch.setattr(subprocess, "run", _run)

    results = operator.execute_plan(
        [
            {
                "op": "write_review_ticket",
                "ticket_name": "spoke_demo-review-ticket_2026-03-29.md",
                "content": "# Demo\n",
            },
            {
                "op": "append_review_pointer",
                "entry_heading": "2026-03-29 Review ticket: demo",
                "bullets": [
                    "`reviews/spoke_demo-review-ticket_2026-03-29.md`",
                    "Prototype pointer entry",
                ],
            },
        ]
    )

    ticket_path = root / "reviews" / "spoke_demo-review-ticket_2026-03-29.md"
    assert results[0]["path"] == str(ticket_path)
    assert ticket_path.read_text(encoding="utf-8") == "# Demo\n"
    note = (root / "projects" / "spoke" / "epistaxis.md").read_text(encoding="utf-8")
    assert "### 2026-03-29 Review ticket: demo" in note
    assert "- `reviews/spoke_demo-review-ticket_2026-03-29.md`" in note
    assert "## Open Questions" in note


def test_stage_commit_and_push_current_branch(tmp_path, monkeypatch):
    root = _make_epistaxis_root(tmp_path)
    ticket_name = "spoke_demo-review-ticket_2026-03-29.md"
    (root / "reviews" / ticket_name).write_text("# Demo\n", encoding="utf-8")
    operator = EpistaxisOperator(root, "spoke")
    calls: list[list[str]] = []

    def _run(cmd, capture_output, text, check):
        calls.append(cmd)
        if cmd[-3:] == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return _completed(stdout="codex/spoke-demo\n")
        if cmd[-3:] == ["rev-parse", "--short", "HEAD"]:
            return _completed(stdout="abc1234\n")
        return _completed(stdout="")

    monkeypatch.setattr(subprocess, "run", _run)

    results = operator.execute_plan(
        [
            {"op": "stage_review_artifacts", "ticket_names": [ticket_name]},
            {"op": "git_commit", "commit_message": "Demo ticket"},
            {"op": "git_push_current_branch"},
        ]
    )

    assert results[0]["paths"] == [
        "projects/spoke/epistaxis.md",
        f"reviews/{ticket_name}",
    ]
    assert results[1]["commit"] == "abc1234"
    assert results[2]["branch"] == "codex/spoke-demo"
    assert ["git", "-C", str(root), "add", "projects/spoke/epistaxis.md", f"reviews/{ticket_name}"] in calls
    assert ["git", "-C", str(root), "commit", "-m", "Demo ticket"] in calls
    assert ["git", "-C", str(root), "push", "-u", "origin", "codex/spoke-demo"] in calls


def test_rejects_bad_ticket_name(tmp_path, monkeypatch):
    root = _make_epistaxis_root(tmp_path)
    operator = EpistaxisOperator(root, "spoke")

    def _run(cmd, capture_output, text, check):
        return _completed(stdout="codex/spoke-demo\n")

    monkeypatch.setattr(subprocess, "run", _run)

    with pytest.raises(EpistaxisOperatorError, match="must start with 'spoke_'"):
        operator.execute_plan(
            [
                {
                    "op": "write_review_ticket",
                    "ticket_name": "wrongprefix_demo.md",
                    "content": "# Demo\n",
                }
            ]
        )


def test_main_cli_reads_json_plan_and_prints_results(tmp_path, monkeypatch, capsys):
    root = _make_epistaxis_root(tmp_path)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"operations": [{"op": "list_reviews"}]}), encoding="utf-8")
    operator = EpistaxisOperator(root, "spoke")
    (root / "reviews" / "spoke_demo_2026-03-29.md").write_text("# Demo\n", encoding="utf-8")

    def _run(cmd, capture_output, text, check):
        return _completed(stdout="codex/spoke-demo\n")

    monkeypatch.setattr(subprocess, "run", _run)
    monkeypatch.setattr(
        "spoke.epistaxis_operator.EpistaxisOperator",
        lambda epistaxis_root, target_repo: operator,
    )

    assert main(
        [
            "--epistaxis-root",
            str(root),
            "--target-repo",
            "spoke",
            "--plan",
            str(plan_path),
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["target_repo"] == "spoke"
    assert payload["operations"][0]["reviews"] == ["spoke_demo_2026-03-29.md"]
