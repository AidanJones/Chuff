# Selected-feature foreground-removal diagnostic

Observed selected-feature intervention with actual prior exposure; regions can change; no semantic causality or new held-out policy

10000 seed-bootstrap draws; aggregate counts across suites within seed, then equally weight seeds; conditional on five fixed foreground recordings

Actual feature must match its stored vector. Zero/unmatchable actuals are excluded explicitly. Undefined seed denominators make the mean and interval null.

Birth, confirmation and adjacent partition all observations; admission overlaps confirmation. Never-admitted candidates are included.

| Configuration | Target-window births: bed match / eligible | Equal-seed % [95% interval] | Target-window admissions: bed match / eligible | Pure-control pairs reproduced at both ends / all |
|---|---:|---:|---:|---:|
| adaptive_gain | 174/1015 | 17.05 [15.13, 19.06] | 72/238 | 124/124 |
| mcra128-f0-l3.5-b1.5 | 233/603 | 39.10 [34.45, 43.54] | 98/140 | 148/148 |
| mcra128-f1-l3.5-b1.5 | 522/1009 | 51.47 [46.72, 56.13] | 130/176 | 256/256 |
| pcen160-f0-l2.5-b1.5 | 167/712 | 23.72 [18.99, 28.32] | 72/183 | 82/82 |
| pcen160-f0-l3.5-b1.5 | 155/540 | 29.08 [23.33, 35.03] | 57/135 | 70/70 |
| pcen160-f1-l2.5-b1.15 | 814/1717 | 47.43 [44.93, 49.57] | 178/279 | 314/314 |
| quiet128-f0-l3.5-b1.5 | 191/881 | 21.84 [18.89, 24.46] | 99/228 | 128/128 |
| quiet128-f1-l3.5-b1.5 | 495/1441 | 34.57 [31.13, 37.65] | 177/336 | 250/250 |
| quiet160-f0-l3.5-b1.5 | 201/902 | 22.42 [19.87, 24.63] | 91/221 | 121/121 |
| quiet160-f1-l3.5-b1.5 | 518/1485 | 35.10 [31.42, 38.47] | 172/331 | 253/253 |

A bed match can select a different region. A failed match can mean no detection or an outside-radius feature; inspect the separate counts in report.json. Pure controls are identical-audio negative controls, not a source-deletion test. Source stems are evaluator-only.
