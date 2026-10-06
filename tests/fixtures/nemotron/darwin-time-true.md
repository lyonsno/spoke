# Darwin Resource Accounting Fixture

Observed on MacBook-Pro-2.local, macOS, October6 2026. Command:
`/usr/bin/time -l -o <output> /usr/bin/true`. Raw unedited output is
`darwin-time-true.txt`. Counts are OS per-command child resource totals;
maximum resident set size is bytes on Darwin. The fixture tests the installed
format, not inference performance. The native test repeats this zero-inference
command and verifies semantic fields, not incidental exact counts.
