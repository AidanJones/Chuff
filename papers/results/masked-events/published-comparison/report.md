# Held-out frontend comparison

10000 paired seed-bootstrap draws jointly across suites; conditional on five fixed foreground recordings, not corpus-level confidence

Post-run descriptive aggregation; no retuning or new selection. Contrast intervals are exploratory and unadjusted for multiple comparisons.

| Configuration | Strict identity % [95% interval] | Background admissions / 210s [95% interval] | Wrong identity % | Observe ms |
|---|---:|---:|---:|---:|
| adaptive_gain | 26.75 [21.25, 32.25] | 6.20 [5.30, 7.10] | 72.25 | 1.58 |
| mcra128-f0-l3.5-b1.5 | 4.50 [1.50, 8.00] | 7.40 [6.65, 8.15] | 81.50 | 4.08 |
| mcra128-f1-l3.5-b1.5 | 2.25 [1.00, 3.75] | 12.80 [12.10, 13.50] | 69.50 | 3.98 |
| pcen160-f0-l2.5-b1.5 | 18.25 [12.25, 23.75] | 4.10 [3.30, 4.85] | 53.75 | 1.73 |
| pcen160-f0-l3.5-b1.5 | 12.00 [7.75, 16.00] | 3.50 [2.95, 4.05] | 50.00 | 1.67 |
| pcen160-f1-l2.5-b1.15 | 11.75 [8.75, 15.50] | 15.70 [14.75, 16.65] | 62.50 | 1.72 |
| quiet128-f0-l3.5-b1.5 | 20.25 [14.75, 25.25] | 6.40 [5.85, 6.95] | 91.00 | 1.41 |
| quiet128-f1-l3.5-b1.5 | 21.25 [15.00, 26.75] | 12.50 [11.45, 13.45] | 82.00 | 1.32 |
| quiet160-f0-l3.5-b1.5 | 21.25 [15.75, 26.25] | 6.05 [5.20, 6.95] | 91.25 | 1.27 |
| quiet160-f1-l3.5-b1.5 | 20.50 [15.00, 24.75] | 12.65 [11.70, 13.55] | 80.50 | 1.21 |

Selected raw/relative pipelines may have different thresholds, especially PCEN; their difference is not an isolated stored-feature effect.

Mean warmed actual-observe milliseconds across six equally weighted suites per seed; desktop shared, not CPU-isolated
