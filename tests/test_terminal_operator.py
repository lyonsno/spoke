import subprocess
from unittest.mock import patch

import pytest


class TestTerminalOperator:
    def test_tool_schema_exposes_argv_and_policy_contract(self):
        from spoke.terminal_operator import tool_schema

        schema = tool_schema()
        params = schema["function"]["parameters"]["properties"]
        assert schema["function"]["name"] == "run_terminal_command"
        assert "argv" in params
        assert params["argv"]["type"] == "array"
        assert "cwd" in params
        assert "timeout_seconds" in params
        assert "argv" in schema["function"]["parameters"]["required"]
        assert "Pass each token as a separate argv entry" in schema["function"]["description"]

    def test_execute_runs_allowlisted_command(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator

        def fake_run(cmd, capture_output, cwd, text, timeout):
            assert cmd == ["git", "status", "--short"]
            assert cwd == str(tmp_path)
            assert timeout == 12
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=" M spoke/tool_dispatch.py\n",
                stderr="",
            )

        with patch("subprocess.run", side_effect=fake_run):
            result = TerminalOperator().execute_command(
                ["git", "status", "--short"],
                cwd=str(tmp_path),
                timeout_seconds=12,
            )

        assert result["decision"] == "allow"
        assert result["executed"] is True
        assert result["exit_code"] == 0
        assert "spoke/tool_dispatch.py" in result["stdout"]
        assert result["stderr"] == ""
        assert result["stdout_truncated"] is False
        assert result["stderr_truncated"] is False

    def test_execute_blocks_approval_required_command(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator

        with patch("subprocess.run") as mock_run:
            result = TerminalOperator().execute_command(
                ["git", "commit", "-m", "hello"],
                cwd=str(tmp_path),
            )

        assert result["decision"] == "approval_required"
        assert result["executed"] is False
        assert "requires approval" in result["reason"]
        mock_run.assert_not_called()

    def test_execute_blocks_denied_command(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator

        with patch("subprocess.run") as mock_run:
            result = TerminalOperator().execute_command(
                ["rm", "-rf", "/tmp/nope"],
                cwd=str(tmp_path),
            )

        assert result["decision"] == "deny"
        assert result["executed"] is False
        assert "denied" in result["reason"]
        mock_run.assert_not_called()

    def test_execute_rejects_shell_syntax_tokens(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator

        result = TerminalOperator().execute_command(
            ["git", "status", "|", "cat"],
            cwd=str(tmp_path),
        )

        assert result["decision"] == "deny"
        assert result["executed"] is False
        assert "shell syntax" in result["reason"]

    def test_execute_truncates_large_output(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator

        big_stdout = "x" * 9000

        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=["git", "status"],
                returncode=0,
                stdout=big_stdout,
                stderr="",
            ),
        ):
            result = TerminalOperator(max_output_chars=128).execute_command(
                ["git", "status"],
                cwd=str(tmp_path),
            )

        assert result["decision"] == "allow"
        assert result["executed"] is True
        assert len(result["stdout"]) <= 128
        assert result["stdout_truncated"] is True

    def test_execute_reports_timeout_for_allowlisted_command(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator

        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["git", "status"], timeout=10),
        ):
            result = TerminalOperator().execute_command(
                ["git", "status"],
                cwd=str(tmp_path),
            )

        assert result["decision"] == "allow"
        assert result["executed"] is True
        assert result["timed_out"] is True
        assert result["exit_code"] is None

    def test_execute_rejects_non_string_argv_entries(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator

        result = TerminalOperator().execute_command(
            ["git", 3],  # type: ignore[list-item]
            cwd=str(tmp_path),
        )

        assert result["decision"] == "deny"
        assert result["executed"] is False
        assert "strings" in result["reason"]
