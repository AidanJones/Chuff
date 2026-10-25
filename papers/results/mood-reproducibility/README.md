# Seeded mood/dream replay — ch-42z, 25 October 2026

`run-a` and `run-b` contain identical `trajectory.json` results from two
independent invocations, including seven dreams and all 45 mood transitions.
Their source hashes were independently checked against each archived file.
Both runs used:

```powershell
uv run chuff-sim-mood --ticks 45 --seed 13 --tick-duration 180 --sound-probability 0.2 --ack-ticks 5 9 --output NEW_DIRECTORY
```

Observed state counts: contentment 11, dreaming 21, new sound learned 5,
acknowledgement 2, song recognised 2, surprise 4. The two ACKs are explicit
synthetic protocol inputs. This run does not demonstrate nine naturally
visited states or acoustic acknowledgement detection. Unit fixtures verify
the nine-state API contract and priority rules separately. No historical
five-seed aggregate is retroactively reproduced by this small replay check.

Environment, agent, schedule and dream streams use seeds 13, 14, 15 and 16.
Source snapshots and dependency versions are in each manifest. Same-host
trajectory equality is verified; cross-platform bitwise equality and
resuming a saved RNG state mid-run are not established.
