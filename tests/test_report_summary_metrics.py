import ast
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


def _load_generate_report():
    source = Path("webapp/app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    func_node = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "generate_report"
    )
    module = ast.Module(body=[func_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(
        compile(module, "webapp/app.py", "exec"),
        {
            "__name__": "webapp.app",
            "np": np,
            "json": json,
            "defaultdict": defaultdict,
        },
        namespace,
    )
    return namespace["generate_report"]


def _build_single_iteration_report_hdf(hdf_path):
    pop = pd.DataFrame([{"Species": "TestFish"}])
    unit_params = pd.DataFrame(
        [{"H": 10.0, "Qopt": 100.0, "Qcap": 150.0, "RPM": 100.0, "D": 5.0}],
        index=pd.Index(["UnitA"], name="Unit"),
    )
    yearly = pd.DataFrame(
        [
            {
                "species": "TestFish",
                "scenario": "ScenarioA",
                "prob_entrainment": 1.0,
                "mean_yearly_entrainment": 368.0,
                "mean_yearly_mortality": 98.0,
            }
        ]
    )
    daily = pd.DataFrame(
        [
            {
                "species": "TestFish",
                "scenario": "ScenarioA",
                "season": "spring",
                "iteration": 0,
                "day": pd.Timestamp("2026-01-01"),
                "flow": 100.0,
                "pop_size": 0.0,
                "num_entrained": 0.0,
                "num_survived": 0.0,
            },
            {
                "species": "TestFish",
                "scenario": "ScenarioA",
                "season": "spring",
                "iteration": 0,
                "day": pd.Timestamp("2026-01-02"),
                "flow": 110.0,
                "pop_size": 200.0,
                "num_entrained": 200.0,
                "num_survived": 10.0,
            },
            {
                "species": "TestFish",
                "scenario": "ScenarioA",
                "season": "spring",
                "iteration": 0,
                "day": pd.Timestamp("2026-01-03"),
                "flow": 120.0,
                "pop_size": 168.0,
                "num_entrained": 168.0,
                "num_survived": 1.0,
            },
        ]
    )
    state_daily = pd.DataFrame(
        [
            {
                "scenario": "ScenarioA",
                "species": "TestFish",
                "iteration": 0,
                "day": pd.Timestamp("2026-01-02"),
                "move": 0,
                "state": "river_node_0",
                "successes": 200.0,
                "count": 200.0,
            },
            {
                "scenario": "ScenarioA",
                "species": "TestFish",
                "iteration": 0,
                "day": pd.Timestamp("2026-01-03"),
                "move": 0,
                "state": "river_node_0",
                "successes": 168.0,
                "count": 168.0,
            },
            {
                "scenario": "ScenarioA",
                "species": "TestFish",
                "iteration": 0,
                "day": pd.Timestamp("2026-01-02"),
                "move": 1,
                "state": "UnitA",
                "successes": 150.0,
                "count": 200.0,
            },
            {
                "scenario": "ScenarioA",
                "species": "TestFish",
                "iteration": 0,
                "day": pd.Timestamp("2026-01-03"),
                "move": 1,
                "state": "UnitA",
                "successes": 120.0,
                "count": 168.0,
            },
        ]
    )

    with pd.HDFStore(hdf_path, mode="w") as store:
        store["Population"] = pop
        store["Unit_Parameters"] = unit_params
        store["Yearly_Summary"] = yearly
        store["Daily"] = daily
        store["State_Daily"] = state_daily


def test_generate_report_uses_full_daily_population_and_state_daily_survival(local_tmp_path):
    hdf_path = local_tmp_path / "single_iteration_report.h5"
    _build_single_iteration_report_hdf(hdf_path)

    generate_report = _load_generate_report()
    sim = SimpleNamespace(
        proj_dir=str(local_tmp_path),
        output_name="single_iteration_report",
        output_units="imperial",
        project_name="Test Project",
        project_notes="N/A",
        model_setup="N/A",
    )

    html = generate_report(sim)

    assert "Whole-Project Survival" in html
    assert "73.4%" in html
    assert "Total fish simulated (from /Daily.pop_size): <strong>368</strong>" in html
    assert "Entrained fish surviving first turbine encounter (from /Daily.num_survived): <strong>11</strong>" in html
    assert "Whole-project survivors (from /State_Daily final move): <strong>270</strong>" in html
    assert "all fish that completed passage" not in html
