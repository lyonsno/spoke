import os
import subprocess
from pathlib import Path
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

    def test_execute_runs_allowlisted_command_with_bounded_exec_env(self, tmp_path):
        from spoke import terminal_operator as terminal_operator

        def fake_run(cmd, capture_output, cwd, text, timeout, env):
            assert cmd == ["/usr/bin/git", "status", "--short"]
            assert cwd == str(tmp_path)
            assert text is False
            assert timeout == 12
            assert env["PATH"] == terminal_operator._EXECUTABLE_SEARCH_PATH
            assert env["HOME"] == "/tmp/git-home"
            assert env["XDG_CONFIG_HOME"] == "/tmp/git-xdg"
            assert "GIT_CONFIG_GLOBAL" not in env
            assert "GIT_CONFIG_NOSYSTEM" not in env
            assert "GIT_CONFIG_SYSTEM" not in env
            assert env["GIT_PAGER"] == "cat"
            assert env["GIT_TERMINAL_PROMPT"] == "0"
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=b" M spoke/tool_dispatch.py\n",
                stderr=b"",
            )

        with (
            patch.dict(
                os.environ,
                {
                    "PATH": "/tmp/shadow-bin",
                    "HOME": "/tmp/git-home",
                    "XDG_CONFIG_HOME": "/tmp/git-xdg",
                    "GIT_CONFIG_GLOBAL": "/tmp/evil-gitconfig",
                },
                clear=False,
            ),
            patch("shutil.which", return_value="/usr/bin/git"),
            patch("subprocess.run", side_effect=fake_run),
        ):
            result = terminal_operator.TerminalOperator().execute_command(
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

    def test_epistaxis_helper_resolves_from_user_local_bin_after_approval(self, tmp_path):
        from spoke import terminal_operator as terminal_operator

        user_local_bin = str(Path.home() / ".local" / "bin")
        resolved = f"{user_local_bin}/epistaxis"

        def fake_which(name, path):
            assert name == "epistaxis"
            assert user_local_bin in path.split(os.pathsep)
            return resolved

        def fake_run(cmd, capture_output, cwd, text, timeout, env):
            assert cmd == [resolved, "zetesis", "what is current state?"]
            assert user_local_bin in env["PATH"].split(os.pathsep)
            assert env["HOME"] == "/tmp/epistaxis-home"
            assert env["CODEX_THREAD_ID"] == "thread-123"
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=b"state\n",
                stderr=b"",
            )

        with (
            patch.dict(
                os.environ,
                {
                    "HOME": "/tmp/epistaxis-home",
                    "CODEX_THREAD_ID": "thread-123",
                },
                clear=False,
            ),
            patch("shutil.which", side_effect=fake_which),
            patch("subprocess.run", side_effect=fake_run),
        ):
            result = terminal_operator.TerminalOperator().execute_command(
                ["epistaxis", "zetesis", "what is current state?"],
                cwd=str(tmp_path),
                approval_granted=True,
            )

        assert result["decision"] == "allow"
        assert result["executed"] is True
        assert result["policy_decision"] == "approval_required"
        assert result["approval_state"] == "granted"
        assert result["stdout"] == "state\n"

    def test_execute_decodes_invalid_binary_output_without_crashing(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator

        def fake_run(cmd, capture_output, cwd, text, timeout, env):
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=b"\xff",
                stderr=b"",
            )

        with (
            patch("shutil.which", return_value="/bin/cat"),
            patch("subprocess.run", side_effect=fake_run),
        ):
            result = TerminalOperator().execute_command(
                ["cat", "README.md"],
                cwd=str(tmp_path),
            )

        assert result["decision"] == "allow"
        assert result["executed"] is True
        assert result["exit_code"] == 0
        assert result["stdout"] == "\ufffd"
        assert result["stderr"] == ""

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

    def test_pending_approval_result_exposes_policy_and_approval_state(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator

        with patch("subprocess.run") as mock_run:
            result = TerminalOperator().execute_command(
                ["git", "commit", "-m", "hello"],
                cwd=str(tmp_path),
            )

        assert result["decision"] == "approval_required"
        assert result["executed"] is False
        assert result["policy_decision"] == "approval_required"
        assert result["policy_reason"] == result["reason"]
        assert result["approval_state"] == "pending"
        mock_run.assert_not_called()

    def test_pending_approval_message_matches_enter_delete_contract(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator

        with patch("subprocess.run") as mock_run:
            result = TerminalOperator().execute_command(
                ["git", "commit", "-m", "hello"],
                cwd=str(tmp_path),
            )

        message = result["approval_request"]["message"]
        assert "Enter to run" in message
        assert "Delete to cancel" in message
        assert "speak or type to revise" in message
        assert "space to run" not in message
        mock_run.assert_not_called()

    def test_approved_run_separates_policy_reason_from_runtime_state(self, tmp_path):
        from spoke import terminal_operator as terminal_operator

        def fake_run(cmd, capture_output, cwd, text, timeout, env):
            assert cmd == ["/usr/bin/git", "commit", "-m", "hello"]
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=b"[ok]\n",
                stderr=b"",
            )

        with (
            patch("shutil.which", return_value="/usr/bin/git"),
            patch("subprocess.run", side_effect=fake_run),
        ):
            result = terminal_operator.TerminalOperator().execute_command(
                ["git", "commit", "-m", "hello"],
                cwd=str(tmp_path),
                approval_granted=True,
            )

        assert result["decision"] == "allow"
        assert result["executed"] is True
        assert result["policy_decision"] == "approval_required"
        assert result["policy_reason"] == "command requires approval: git commit -m"
        assert result["approval_state"] == "granted"
        assert "reason" not in result

    def test_execute_requires_approval_for_find_delete(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator

        with patch("subprocess.run") as mock_run:
            result = TerminalOperator().execute_command(
                ["find", ".", "-delete"],
                cwd=str(tmp_path),
            )

        assert result["decision"] == "approval_required"
        assert result["executed"] is False
        assert "requires approval" in result["reason"]
        mock_run.assert_not_called()

    def test_execute_requires_approval_for_sed_family(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator

        with patch("subprocess.run") as mock_run:
            result = TerminalOperator().execute_command(
                ["sed", "-n", "w", "/tmp/out", "file.txt"],
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

    def test_execute_blocks_denied_command_by_absolute_path(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator

        with patch("subprocess.run") as mock_run:
            result = TerminalOperator().execute_command(
                ["/bin/rm", "-rf", "/tmp/nope"],
                cwd=str(tmp_path),
            )

        assert result["decision"] == "deny"
        assert result["executed"] is False
        assert "denied" in result["reason"]
        mock_run.assert_not_called()

    def test_execute_rejects_explicit_executable_paths(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator

        with patch("subprocess.run") as mock_run:
            result = TerminalOperator().execute_command(
                ["/tmp/git", "status"],
                cwd=str(tmp_path),
            )

        assert result["decision"] == "deny"
        assert result["executed"] is False
        assert "bare executable name" in result["reason"]
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
        assert result["output_complete"] is False
        assert result["stdout_truncated"] is True
        assert (
            result["truncation_message"]
            == "tool output truncated before it reached the assistant: stdout truncated to 128 chars"
        )

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

    def test_execute_requires_approval_for_git_diff_output_flag(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator

        with patch("subprocess.run") as mock_run:
            result = TerminalOperator().execute_command(
                ["git", "diff", "--output=/tmp/out.patch"],
                cwd=str(tmp_path),
            )

        assert result["decision"] == "approval_required"
        assert result["executed"] is False
        assert "requires approval" in result["reason"]
        mock_run.assert_not_called()

    def test_execute_requires_approval_for_git_diff_no_index(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator

        with patch("subprocess.run") as mock_run:
            result = TerminalOperator().execute_command(
                ["git", "diff", "--no-index", "/etc/passwd", "/dev/null"],
                cwd=str(tmp_path),
            )

        assert result["decision"] == "approval_required"
        assert result["executed"] is False
        assert "requires approval" in result["reason"]
        mock_run.assert_not_called()

    def test_execute_requires_approval_for_git_diff_ext_diff(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator

        with patch("subprocess.run") as mock_run:
            result = TerminalOperator().execute_command(
                ["git", "diff", "--ext-diff"],
                cwd=str(tmp_path),
            )

        assert result["decision"] == "approval_required"
        assert result["executed"] is False
        assert "requires approval" in result["reason"]
        mock_run.assert_not_called()

    def test_execute_requires_approval_for_git_show_textconv(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator

        with patch("subprocess.run") as mock_run:
            result = TerminalOperator().execute_command(
                ["git", "show", "--textconv", "HEAD:README.md"],
                cwd=str(tmp_path),
            )

        assert result["decision"] == "approval_required"
        assert result["executed"] is False
        assert "requires approval" in result["reason"]
        mock_run.assert_not_called()

    def test_execute_requires_approval_for_git_blame_contents(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator

        with patch("subprocess.run") as mock_run:
            result = TerminalOperator().execute_command(
                ["git", "blame", "--contents", "/tmp/outside", "README.md"],
                cwd=str(tmp_path),
            )

        assert result["decision"] == "approval_required"
        assert result["executed"] is False
        assert "requires approval" in result["reason"]
        mock_run.assert_not_called()

    def test_execute_requires_approval_for_git_blame_contents_equals(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator

        with patch("subprocess.run") as mock_run:
            result = TerminalOperator().execute_command(
                ["git", "blame", "--contents=/tmp/outside", "README.md"],
                cwd=str(tmp_path),
            )

        assert result["decision"] == "approval_required"
        assert result["executed"] is False
        assert "requires approval" in result["reason"]
        mock_run.assert_not_called()

    def test_execute_requires_approval_for_rg_e_outside_root(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator

        with patch("subprocess.run") as mock_run:
            result = TerminalOperator().execute_command(
                ["rg", "-e", "needle", "/etc/passwd"],
                cwd=str(tmp_path),
            )

        assert result["decision"] == "approval_required"
        assert result["executed"] is False
        assert "requires approval" in result["reason"]
        mock_run.assert_not_called()

    def test_execute_requires_approval_for_rg_files_outside_root(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator

        with patch("subprocess.run") as mock_run:
            result = TerminalOperator().execute_command(
                ["rg", "--files", "/etc"],
                cwd=str(tmp_path),
            )

        assert result["decision"] == "approval_required"
        assert result["executed"] is False
        assert "requires approval" in result["reason"]
        mock_run.assert_not_called()

    def test_execute_requires_approval_for_rg_pre_helper(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator

        with patch("subprocess.run") as mock_run:
            result = TerminalOperator().execute_command(
                ["rg", "--pre", "python3", "needle", "."],
                cwd=str(tmp_path),
            )

        assert result["decision"] == "approval_required"
        assert result["executed"] is False
        assert "requires approval" in result["reason"]
        mock_run.assert_not_called()

    def test_execute_requires_approval_for_rg_pre_equals_helper(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator

        with patch("subprocess.run") as mock_run:
            result = TerminalOperator().execute_command(
                ["rg", "--pre=python3", "needle", "."],
                cwd=str(tmp_path),
            )

        assert result["decision"] == "approval_required"
        assert result["executed"] is False
        assert "requires approval" in result["reason"]
        mock_run.assert_not_called()

    def test_execute_requires_approval_for_rg_file_equals_outside_root(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator

        with patch("subprocess.run") as mock_run:
            result = TerminalOperator().execute_command(
                ["rg", "--file=/etc/passwd", "."],
                cwd=str(tmp_path),
            )

        assert result["decision"] == "approval_required"
        assert result["executed"] is False
        assert "requires approval" in result["reason"]
        mock_run.assert_not_called()

    def test_execute_requires_approval_for_rg_ignore_file(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator

        with patch("subprocess.run") as mock_run:
            result = TerminalOperator().execute_command(
                ["rg", "--ignore-file", "/etc/passwd", "needle", "."],
                cwd=str(tmp_path),
            )

        assert result["decision"] == "approval_required"
        assert result["executed"] is False
        assert "requires approval" in result["reason"]
        mock_run.assert_not_called()

    def test_execute_requires_approval_for_rg_attached_short_file_flag(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator

        with patch("subprocess.run") as mock_run:
            result = TerminalOperator().execute_command(
                ["rg", "-f/etc/passwd", "needle", "."],
                cwd=str(tmp_path),
            )

        assert result["decision"] == "approval_required"
        assert result["executed"] is False
        assert "requires approval" in result["reason"]
        mock_run.assert_not_called()

    def test_execute_requires_approval_for_rg_attached_short_regexp_flag(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator

        with patch("subprocess.run") as mock_run:
            result = TerminalOperator().execute_command(
                ["rg", "-eneedle", "/etc/passwd"],
                cwd=str(tmp_path),
            )

        assert result["decision"] == "approval_required"
        assert result["executed"] is False
        assert "requires approval" in result["reason"]
        mock_run.assert_not_called()

    def test_execute_requires_approval_for_rg_clustered_short_file_flag(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator

        with patch("subprocess.run") as mock_run:
            result = TerminalOperator().execute_command(
                ["rg", "-nf/etc/passwd", "/etc/hosts"],
                cwd=str(tmp_path),
            )

        assert result["decision"] == "approval_required"
        assert result["executed"] is False
        assert "requires approval" in result["reason"]
        mock_run.assert_not_called()

    def test_execute_requires_approval_for_ps_eww(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator

        with patch("subprocess.run") as mock_run:
            result = TerminalOperator().execute_command(
                ["ps", "eww", "123"],
                cwd=str(tmp_path),
            )

        assert result["decision"] == "approval_required"
        assert result["executed"] is False
        assert "requires approval" in result["reason"]
        mock_run.assert_not_called()

    def test_execute_raises_bounded_error_for_spawn_failure(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator, TerminalOperatorError

        with patch("subprocess.run", side_effect=FileNotFoundError("rg not found")):
            with pytest.raises(TerminalOperatorError, match="failed to start command"):
                TerminalOperator().execute_command(
                    ["rg", "needle"],
                    cwd=str(tmp_path),
                )

    def test_execute_rejects_non_string_argv_entries(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator

        result = TerminalOperator().execute_command(
            ["git", 3],  # type: ignore[list-item]
            cwd=str(tmp_path),
        )

        assert result["decision"] == "deny"
        assert result["executed"] is False
        assert "strings" in result["reason"]

    def test_execute_rejects_cwd_outside_allowed_roots(self):
        from spoke.terminal_operator import TerminalOperator, TerminalOperatorError

        with pytest.raises(TerminalOperatorError, match="outside allowed local roots"):
            TerminalOperator().execute_command(
                ["git", "status"],
                cwd="/etc",
            )

    def test_execute_rejects_home_root_as_cwd(self):
        from spoke.terminal_operator import TerminalOperator, TerminalOperatorError

        with pytest.raises(TerminalOperatorError, match="outside allowed local roots"):
            TerminalOperator().execute_command(
                ["git", "status"],
                cwd=str(Path.home()),
            )

    def test_execute_requires_approval_for_sensitive_home_path(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator

        with patch("subprocess.run") as mock_run:
            result = TerminalOperator().execute_command(
                ["cat", str(Path.home() / ".ssh" / "id_rsa")],
                cwd=str(tmp_path),
            )

        assert result["decision"] == "approval_required"
        assert result["executed"] is False
        assert "requires approval" in result["reason"]
        mock_run.assert_not_called()

    def test_execute_requires_approval_for_bare_name_symlink_operand(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator

        link_path = tmp_path / "secret_link"
        link_path.symlink_to(Path("/etc/hosts"))

        with patch("subprocess.run") as mock_run:
            result = TerminalOperator().execute_command(
                ["cat", "secret_link"],
                cwd=str(tmp_path),
            )

        assert result["decision"] == "approval_required"
        assert result["executed"] is False
        assert "requires approval" in result["reason"]
        mock_run.assert_not_called()

    def test_execute_requires_approval_for_bare_name_symlink_after_option_terminator(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator

        link_path = tmp_path / "-secret_link"
        link_path.symlink_to(Path("/etc/hosts"))

        with patch("subprocess.run") as mock_run:
            result = TerminalOperator().execute_command(
                ["cat", "--", "-secret_link"],
                cwd=str(tmp_path),
            )

        assert result["decision"] == "approval_required"
        assert result["executed"] is False
        assert "requires approval" in result["reason"]
        mock_run.assert_not_called()

    def test_execute_rejects_non_integer_timeout(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator, TerminalOperatorError

        with pytest.raises(TerminalOperatorError, match="must be an integer"):
            TerminalOperator().execute_command(
                ["git", "status"],
                cwd=str(tmp_path),
                timeout_seconds="abc",
            )

    def test_execute_rejects_out_of_range_timeout(self, tmp_path):
        from spoke.terminal_operator import TerminalOperator, TerminalOperatorError

        with pytest.raises(TerminalOperatorError, match="between 1 and 30"):
            TerminalOperator().execute_command(
                ["git", "status"],
                cwd=str(tmp_path),
                timeout_seconds=99,
            )
