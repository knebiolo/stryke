# Route Aggregation and Survival Reporting (Policy B)

This document explains the recent changes to the report aggregation logic and survival extraction.

What changed

- Policy B (average across iterations) is now the default aggregation for route counts in the report.
  - We deduplicate passage events by `(fish_id, iteration, day)` because fish IDs are assigned per-day in the simulation.
  - Per-iteration route counts are computed by counting unique fish instances per iteration (i.e., sum of unique fish per day within each iteration).
  - The final report shows the mean per-iteration counts across all iterations.

- Survival extraction is now performed using explicit `survival_*` columns written to the `/simulations/...` DataFrame.
  - For a passage event that occurs at `state_k`, the corresponding survival flag is `survival_{k-1}` when available (i.e., survival before entering the state).
  - If survival columns are missing, the code falls back to `None` for the survived field.

Why

- Earlier heuristics attempted to read survival from numeric blocks in the HDF; that method was unreliable and sometimes produced incorrect values (e.g., 100% survival). The new approach uses explicit survival columns already present in the DataFrame, making the report accurate and auditable.

How to run the tests

From the project root run:

```
python -m pip install -r requirements.txt  # ensure test deps are present (pytest, pandas)
pytest -q
```

Files changed

- `webapp/app.py` — uses explicit `survival_*` columns and Policy B dedupe for route aggregation
- `tests/test_route_stats.py` — small unit test for aggregation logic
- `docs/ROUTE_AGG_README.md` — this file

Notes

- If you re-run simulations with multiple iterations, the report will provide mean per-iteration counts.
- If you observe unexpected survival rates, verify that `survival_*` columns are being populated correctly by the simulation engine; I can add additional checks or unit tests for that as well.
