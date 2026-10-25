# Candidate birth/admission label sensitivity

Observed-data label sensitivity, not a replacement accuracy estimate or a newly held-out method

10000 paired seed-bootstrap draws, jointly across positive suites; conditional on five fixed foreground recordings

| Configuration | Mixed-stream admissions | FG to BG | BG to FG | Different recording | Admission strict % | Birth strict % | Difference pp [95% interval] |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive_gain | 388 | 36 | 52 | 46 | 26.75 | 32.50 | +5.75 [-0.50, +12.25] |
| mcra128-f0-l3.5-b1.5 | 306 | 45 | 75 | 34 | 4.50 | 4.50 | +0.00 [-2.75, +2.25] |
| mcra128-f1-l3.5-b1.5 | 458 | 94 | 97 | 48 | 2.25 | 3.50 | +1.25 [-1.25, +3.25] |
| pcen160-f0-l2.5-b1.5 | 284 | 25 | 45 | 30 | 18.25 | 20.00 | +1.75 [-0.25, +3.50] |
| pcen160-f0-l3.5-b1.5 | 218 | 21 | 40 | 21 | 12.00 | 14.00 | +2.00 [+0.50, +3.75] |
| pcen160-f1-l2.5-b1.15 | 579 | 81 | 95 | 77 | 11.75 | 14.50 | +2.75 [-0.50, +5.75] |
| quiet128-f0-l3.5-b1.5 | 366 | 33 | 69 | 44 | 20.25 | 24.50 | +4.25 [+1.00, +8.00] |
| quiet128-f1-l3.5-b1.5 | 616 | 75 | 111 | 55 | 21.25 | 24.00 | +2.75 [+0.25, +5.25] |
| quiet160-f0-l3.5-b1.5 | 357 | 32 | 69 | 36 | 21.25 | 24.75 | +3.50 [+0.50, +6.75] |
| quiet160-f1-l3.5-b1.5 | 622 | 88 | 115 | 50 | 20.50 | 23.50 | +3.00 [+0.50, +5.75] |

Birth and admission window labels do not identify the acoustic cause of a stored vector. Background births with prior source tails are explicit.
