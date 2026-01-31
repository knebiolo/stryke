# Test Plan - Stryke

## Goals
- Catch routing/flow allocation regressions (env/bypass/spill/units).
- Validate survival calculations (blade strike, impingement, barotrauma).
- Ensure unit conversions and reporting outputs are consistent for metric/imperial.
- Verify webapp import/export and model summary/report rendering.

## Non-Goals
- Full UI automation coverage of all pages (phase 1 focuses on critical paths).
- Performance benchmarking on large Monte Carlo runs (tracked separately).

## Test Environments
- Windows + conda env (primary).
- Optional: Linux CI (if/when added).

## Test Data Fixtures
- Minimal .stryke files:
  - Single facility, 2 units, env_flow + bypass_flow + spillway nodes.
  - Existing multi-unit Kakabeka files (already in Data/).
- Minimal hydrograph CSV (3-5 days).
- Facilities CSV with nonzero Env_Flow and Bypass_Flow.
- Unit params CSV with barotrauma fields present and missing (for error tests).

## Unit Tests (Core Logic)
### Routing and Flow Allocation
- movement(): env_flow probability equals env_Q / curr_Q when env node exists.
- movement(): bypass_flow probability equals bypass_Q / curr_Q when bypass node exists.
- movement(): spill residual excludes env + bypass + unit flows.
- compute_unit_flow_allocations(): respects min_op_flow threshold for unit commitment.
- daily_hours(): hours_dict keys map to unit identifiers consistently.

### Survival and Mortality
- node_surv_rate():
  - blocked_by_rack True -> blade strike skipped, barotrauma skipped.
  - impingement logic triggers when width > rack_spacing and U_crit < intake_vel.
  - barotrauma requires fb_depth and elevation_head/submergence_depth (fail loud).
- Kaplan/Propeller/Francis/Pump survival functions return values in [0, 1].

### Unit Conversion
- Metric inputs converted to imperial for internal calcs, then back for reports:
  - H, D, Qopt, Qcap, ps_* fields, fb_depth, elevation_head.
- Driver diagnostics output uses metric when units == metric.

## Integration Tests (Simulation Output)
### HDF Outputs
- Route_Flows includes env_flow and bypass_flow rows when flows are nonzero.
- Unit_Hours includes discharge_cfs and hours for each unit.
- Driver_Diagnostics includes mean_abs_pct_off_qopt, pct_hours_outside_qopt_band.

### Routing Consistency
- Sum of route discharge fractions equals 1.0 (within tolerance) per day.
- Env flow present even when curr_Q < station capacity.

### Barotrauma Enforcement
- When barotrauma inputs exist, simulation uses barotrauma for all facilities.
- Missing required barotrauma inputs throws descriptive errors.

## Web App Tests
- Load .stryke file -> graph_summary nodes/edges match file.
- Model summary page shows passage routes list including env_flow/bypass_flow.
- Facilities page correctly saves Env_Flow and Bypass_Flow (metric + imperial).
- Report shows nonzero mean discharge for env_flow/bypass_flow when set.

## Regression Tests
- Legacy project with no env/bypass nodes still runs without crash.
- Single-unit survival-only model runs without graph edges.

## Performance/Scale Tests (Optional)
- 10k iterations, 365 days: runtime within expected envelope.
- HDF size growth monitored for Route_Flows and Unit_Hours tables.

## Error Handling Tests (Fail Loud)
- Missing rack spacing -> raise descriptive error if used in impingement check.
- Missing U_crit -> raise or default? (decide and codify).
- Env/bypass nodes present but Env_Flow/Bypass_Flow missing -> raise error.

## Reporting/Diagnostics Validation
- Discharge by passage route matches Route_Flows aggregation.
- Driver diagnostics flow_share sums to 1.0 for unit routes.
- Qopt deviation metrics in diagnostics are computed from Unit_Hours when present.

## Phased Implementation
1) Unit tests for routing + survival edge cases.
2) Integration tests around HDF outputs.
3) Webapp tests (Flask test client).
4) Performance checks (optional).

## Ownership and Maintenance
- New features must add/update tests in the relevant section.
- Keep fixtures small and deterministic; avoid full Monte Carlo runs in CI.
