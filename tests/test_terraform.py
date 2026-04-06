"""Tests for the Terraform epistaxis topoi parser."""

from spoke.terraform import (
    Topos, parse_topoi, format_topos_summary, sort_topoi, filter_topoi,
    disambiguated_name, _is_tag_semeion, _clean_status,
)


_SAMPLE_NOTE = """\
# Spoke Epistaxis

## Status

- **`origin/main-next` at `405a0e1`.**

## Scoped Local State

### cc-ham-hogg-0402
- Machine: `MacBook-Pro-2.local` | Tool: Claude Code (Opus 4.6)
- Checkout: `/Users/dev/donttype` | Branch: `ham-hogg-0402` | Worktree: `/tmp/spoke-ham-hogg-0402`
- Continuation: `codex resume 019d4bf1-1303-7053-9c37-f8f36fc5d720`
- Temperature: `hot`
- Attractors: `support-spoke-process-heartbeat_2026-04-02`, ~~`stop-test-killing-live_2026-04-02`~~ (test-kills-live → **katástasis**)
- [Sēmeion: `Operation Ham-Hogg`]
- [Sēmeion: `Operation Wet Nurse` — triage squadron for Careless Whisper fallout]
- Status: **Active.** Heartbeat + model TTL landed on `dev-0402`.

### cc-panic-switch-0404
- Session ID: `214778e9` | Machine: `MacBook-Pro-2.local` | Tool: Claude Code (Opus 4.6)
- Checkout: `/Users/dev/donttype` | Branch: `cc/panic-switch-0404` | Worktree: `/tmp/spoke-panic-switch-0404`
- Continuation: `claude -r 214778e9-5203-47f6-b791-f436dcbc2694`
- Temperature: `hot`
- Attractors: `stop-tool-call-parser-drop_2026-04-03`, `support-cancel-generation_2026-04-03`
- [Sēmeion: `Operation Panic Switch`]
- Status: **Active.** Fix for silent tool-call drop plus cancel chord landed.

### project-friendly-snoop-0402
- Session ID: `session-386e5f84` | Machine: `darwin` | Tool: `Gemini CLI`
- Temperature: **katástasis**
- Attractors: ~~`allow-operator-shell-to-query-gmail_2026-03-29`~~ (satisfied)
- [Sēmeion: Project Friendly Snoop]
- Status: **Katástasis.** Gmail operator merged.

## Decisions

- Some decisions here.
"""


def test_parse_topoi_count():
    topoi = parse_topoi(_SAMPLE_NOTE)
    assert len(topoi) == 3


def test_parse_topos_id():
    topoi = parse_topoi(_SAMPLE_NOTE)
    assert topoi[0].id == "cc-ham-hogg-0402"
    assert topoi[1].id == "cc-panic-switch-0404"
    assert topoi[2].id == "project-friendly-snoop-0402"


def test_parse_semeion():
    topoi = parse_topoi(_SAMPLE_NOTE)
    assert topoi[0].semeion == "Operation Ham-Hogg"
    assert topoi[1].semeion == "Operation Panic Switch"
    assert topoi[2].semeion == "Project Friendly Snoop"


def test_parse_all_semeions():
    topoi = parse_topoi(_SAMPLE_NOTE)
    assert topoi[0].all_semeions == ["Operation Ham-Hogg", "Operation Wet Nurse"]


def test_parse_branch():
    topoi = parse_topoi(_SAMPLE_NOTE)
    assert topoi[0].branch == "ham-hogg-0402"
    assert topoi[1].branch == "cc/panic-switch-0404"


def test_parse_resume_cmd():
    topoi = parse_topoi(_SAMPLE_NOTE)
    assert topoi[0].resume_cmd == "codex resume 019d4bf1-1303-7053-9c37-f8f36fc5d720"
    assert topoi[1].resume_cmd == "claude -r 214778e9-5203-47f6-b791-f436dcbc2694"


def test_parse_temperature():
    topoi = parse_topoi(_SAMPLE_NOTE)
    assert topoi[0].temperature == "hot"
    assert topoi[2].temperature == "katástasis"


def test_parse_machine():
    topoi = parse_topoi(_SAMPLE_NOTE)
    assert topoi[0].machine == "MacBook-Pro-2.local"


def test_parse_tool():
    topoi = parse_topoi(_SAMPLE_NOTE)
    assert topoi[0].tool == "Claude Code (Opus 4.6)"
    assert topoi[2].tool == "Gemini CLI"


def test_parse_status():
    topoi = parse_topoi(_SAMPLE_NOTE)
    assert topoi[0].status.startswith("Active.")
    assert "Katástasis." in topoi[2].status


def test_parse_attractors():
    topoi = parse_topoi(_SAMPLE_NOTE)
    assert "support-spoke-process-heartbeat_2026-04-02" in topoi[0].attractors
    # Strikethrough attractors are still parsed (just without ~~)
    assert "stop-test-killing-live_2026-04-02" in topoi[0].attractors


def test_format_topos_summary_with_semeion():
    topos = Topos(
        id="cc-ham-hogg-0402",
        semeion="Operation Ham-Hogg",
        temperature="hot",
        status="Active. Heartbeat landed.",
    )
    summary = format_topos_summary(topos)
    assert "Operation Ham-Hogg" in summary
    assert "[hot]" in summary
    assert "Active" in summary


def test_format_topos_summary_without_semeion():
    topos = Topos(id="cc-something-0404")
    summary = format_topos_summary(topos)
    assert summary == "cc-something-0404"


# -- Tag semeion detection --

def test_is_tag_semeion_reboot():
    assert _is_tag_semeion("reboot")
    assert _is_tag_semeion("reboot-pending")


def test_is_tag_semeion_consult():
    assert _is_tag_semeion("consult metadosis/spoke_something.md")


def test_is_tag_semeion_real_name():
    assert not _is_tag_semeion("Operation Ham-Hogg")
    assert not _is_tag_semeion("Project Finger Flounder")
    assert not _is_tag_semeion("Baboon")


def test_tag_semeion_falls_back_to_id():
    """When all semeions are tags, semeion should be None (falls back to id)."""
    text = """\
# Spoke

## Scoped Local State

### gemini-careless-whisper-0402
- [Sēmeion: reboot — blocked on system restart]
- [Sēmeion: consult metadosis/something.md — pull commit abc]
- Status: **Active.**
"""
    topoi = parse_topoi(text)
    assert len(topoi) == 1
    assert topoi[0].semeion is None  # all semeions are tags
    assert topoi[0].all_semeions == ["reboot", "consult metadosis/something.md"]


def test_tag_semeion_skips_to_real_name():
    """When first semeion is a tag but second is a name, use the name."""
    text = """\
# Spoke

## Scoped Local State

### cc-something-0402
- [Sēmeion: reboot — blocked]
- [Sēmeion: `Operation Cool Name` — the real one]
- Status: **Active.**
"""
    topoi = parse_topoi(text)
    assert topoi[0].semeion == "Operation Cool Name"


# -- Status cleaning --

def test_clean_status_strips_backticks():
    assert _clean_status("`main-next`-adjacent work") == "main-next-adjacent work"


def test_clean_status_strips_bold_and_strikethrough():
    assert _clean_status("**Active.** Some ~~old~~ thing") == "Active. Some old thing"


# -- Temperature inference from status --

def test_temperature_inferred_from_status():
    text = """\
# Spoke

## Scoped Local State

### cc-ham-hogg-0402
- Status: **Κατάστασις (2026-04-05)** — Settled.
"""
    topoi = parse_topoi(text)
    assert topoi[0].temperature == "katástasis"


# -- Disambiguated names --

def test_disambiguated_name_with_machine_and_tool():
    t = Topos(id="x", semeion="Finger Flounder", machine="MacBook-Pro-2.local", tool="Codex")
    assert disambiguated_name(t) == "Finger Flounder  (MacBook-Pro-2, Codex)"


def test_disambiguated_name_strips_tool_version():
    t = Topos(id="x", semeion="Ham-Hogg", machine="nlm2pr.local", tool="Claude Code (Opus 4.6)")
    assert disambiguated_name(t) == "Ham-Hogg  (nlm2pr, Claude Code)"


def test_disambiguated_name_no_metadata():
    t = Topos(id="cc-something-0404")
    assert disambiguated_name(t) == "cc-something-0404"


def test_parse_empty_scoped_state():
    text = """\
# Spoke Epistaxis

## Scoped Local State

_No active lanes._

## Decisions
"""
    topoi = parse_topoi(text)
    assert topoi == []


def test_parse_no_scoped_state_section():
    text = "# Spoke Epistaxis\n\n## Decisions\n\n- stuff\n"
    topoi = parse_topoi(text)
    assert topoi == []


# -- Sorting tests --

def _make_topoi():
    return [
        Topos(id="a", semeion="Alpha", temperature="warm", machine="box-1", tool="Claude Code"),
        Topos(id="b", semeion="Beta", temperature="hot", machine="box-2", tool="Codex"),
        Topos(id="c", semeion="Gamma", temperature="katástasis", machine="box-1", tool="Gemini CLI"),
        Topos(id="d", semeion="Delta", temperature="cool", machine="box-2", tool="Claude Code"),
        Topos(id="e", temperature="cold", machine="box-1"),
    ]


def test_sort_by_temperature():
    topoi = sort_topoi(_make_topoi(), key="temperature")
    temps = [t.temperature for t in topoi]
    assert temps == ["hot", "warm", "cool", "cold", "katástasis"]


def test_sort_by_semeion():
    topoi = sort_topoi(_make_topoi(), key="semeion")
    names = [t.semeion or t.id for t in topoi]
    assert names == ["Alpha", "Beta", "Delta", "e", "Gamma"]


def test_sort_by_machine():
    topoi = sort_topoi(_make_topoi(), key="machine")
    # box-1 group first (alphabetical), then box-2, sorted by temp within
    machines = [(t.machine, t.temperature) for t in topoi]
    assert machines[0] == ("box-1", "warm")
    assert machines[1] == ("box-1", "cold")
    assert machines[2] == ("box-1", "katástasis")
    assert machines[3][0] == "box-2"


# -- Filtering tests --

def test_filter_hide_katastasis():
    result = filter_topoi(_make_topoi(), hide_katastasis=True)
    assert all(t.temperature != "katástasis" for t in result)
    assert len(result) == 4


def test_filter_by_machine():
    result = filter_topoi(_make_topoi(), machine="box-1")
    assert all("box-1" in t.machine for t in result)
    assert len(result) == 3


def test_filter_by_tool():
    result = filter_topoi(_make_topoi(), tool="claude")
    assert len(result) == 2
    assert all("Claude" in t.tool for t in result)


def test_filter_by_temperature():
    result = filter_topoi(_make_topoi(), temperature="hot")
    assert len(result) == 1
    assert result[0].id == "b"


def test_filter_combined():
    result = filter_topoi(_make_topoi(), machine="box-1", hide_katastasis=True)
    assert len(result) == 2
    assert all(t.machine == "box-1" for t in result)
    assert all(t.temperature != "katástasis" for t in result)
