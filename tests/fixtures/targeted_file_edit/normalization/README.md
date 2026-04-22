Normalization fixtures witness deterministic matching and writeback behavior
without importing implementation internals.

These fixtures are byte-sensitive: some use `*.hex.json` payloads so the witness
can preserve CRLF, trailing whitespace, and final-newline state exactly.
