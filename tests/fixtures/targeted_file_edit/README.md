Targeted file edit fixtures for Project Sniper Lane T.

Each fixture directory witnesses one bounded contract case without importing
the implementation. The runner discovers the eventual tool by schema shape
(`file`, `old_string`, `new_string`) rather than by hardcoded tool name so
Lane T can stay black-box relative to Lane A.

Whitespace- and newline-sensitive fixtures may store `source` and
`expected_file` as `*.hex.json` sidecars instead of plain `*.txt` payloads.
That keeps the witness honest for CRLF, trailing whitespace, and missing-final-
newline cases that a text-only harness would accidentally normalize away.
