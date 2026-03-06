# Stryke Testing Strategy

## Goals
- Catch correctness regressions in routing, survival, and summary outputs.
- Catch performance/memory regressions before they hit user runs.
- Keep fast feedback for local development, with deeper checks for release candidates.

## Test Pyramid
- Unit tests (`tests/test_*.py`): pure logic and validation rules.
- Component tests: HDF read/write and summary/report assembly on synthetic datasets.
- Scenario smoke tests: short end-to-end simulation runs with representative model inputs.

## Test Suites

### 1) Fast Unit Suite (target: < 30s)
- Command:
  - `PYTHONPATH=. pytest -q tests/test_compute_unit_flow_allocations.py tests/test_daily_hours.py tests/test_population_sim_caps.py tests/test_mortality_component_counters.py tests/test_unit_params_validation.py tests/test_barotrauma_validation.py`
- Purpose:
  - Core hydraulic allocation behavior.
  - Population scaling and fail-fast limits.
  - Mortality accounting implementation details.
  - Unit parameter and barotrauma validations.

### 2) Summary/Storage Suite (target: < 60s)
- Command:
  - `PYTHONPATH=. pytest -q tests/test_summary_state_daily.py tests/test_driver_diagnostics.py tests/test_route_stats.py`
- Purpose:
  - Validate summary correctness using lightweight `State_Daily` aggregates.
  - Ensure diagnostics remain available when full fish-level sim table is not persisted.
  - Verify route aggregation and report-facing outputs.

### 3) Routing Behavior Suite (target: < 60s)
- Command:
  - `PYTHONPATH=. pytest -q tests/test_movement_multiple_routes.py tests/test_routing_env_bypass.py tests/test_routing_units_env_bypass.py tests/test_rack_spacing.py`
- Purpose:
  - Neighbor selection, route flow splits, rack effects, and survival pathway behavior.

### 4) Full Regression Suite (pre-release)
- Command:
  - `PYTHONPATH=. pytest -q tests`
- Purpose:
  - Run all tests before merge/release.
  - Baseline for acceptance after major simulation-core changes.

## Performance & Memory Checks
- Add a lightweight runtime check to release validation:
  - Run one known model with low iteration count and verify:
    - No unbounded RAM growth across days.
    - No `Population size n=` guard failures unless expected.
    - Stable completion time trend across iterations.
- Recommended environment defaults for production stability:
  - `STRYKE_STORE_SIM_TABLE=0`
  - `STRYKE_STORE_AGENT_TRACES=0`
  - `STRYKE_MAX_DAILY_FISH=100000` (raise only with deliberate approval)

## Quality Gates
- Every simulation-core change must pass:
  - Fast Unit Suite
  - Summary/Storage Suite
- Any change to routing or movement must also pass:
  - Routing Behavior Suite
- Release candidate must pass:
  - Full Regression Suite

## Test Coverage Roadmap
- Near-term additions:
  - Direct memory-growth sentinel test around long synthetic runs (bounded rows and object counts).
  - Route-level summary parity test comparing `State_Daily` vs full sim-table outputs.
  - Input fuzz tests for malformed hydrograph and operating-scenario payloads.
- Medium-term:
  - CI matrix across Python versions used by production.
  - Scheduled nightly smoke test with representative `.stryke` models.

