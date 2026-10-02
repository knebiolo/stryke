import ast
import io
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _load_generate_report():
    source = Path("webapp/app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {"generate_report", "_escapement_and_length_html"}
    func_nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    module = ast.Module(body=func_nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
            "__name__": "webapp.app",
            "np": np,
            "json": json,
            "defaultdict": defaultdict,
            "io": io,
            "pd": pd,
            "plt": plt,
    }
    exec(compile(module, "webapp/app.py", "exec"), namespace)
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


def test_escapement_and_length_section():
    generate_report = _load_generate_report()
    helper = generate_report.__globals__["_escapement_and_length_html"]
    daily = pd.DataFrame({
        "iteration": [0, 0, 1],
        "num_entrained": [6, 2, 8],
        "num_escaped": [1, 1, 2],
    })
    length = pd.DataFrame({
        "scenario": ["s"] * 3,
        "species": ["Walleye"] * 3,
        "length_bin_lower_cm": [3.0, 4.0, 26.0],
        "mean_num_entrained": [5.0, 3.0, 0.0],
        "mean_num_mortality": [1.0, 1.0, 0.0],
        "mean_mortality_impingement": [0.0, 0.0, 0.0],
        "mean_mortality_blade_strike": [1.0, 0.5, 0.0],
        "mean_mortality_barotrauma": [0.0, 0.5, 0.0],
        "mean_mortality_other": [0.0, 0.0, 0.0],
        "mean_num_escaped": [0.0, 0.0, 2.0],
    })
    html = helper(daily, length)
    assert "Rack Escapement" in html
    assert "20.0%" in html  # 2 escaped / (8 entrained + 2 escaped)
    assert "no spillway mapped" in html
    assert "facility's spillway" not in html
    assert "Mortality by Fish Length" in html
    assert ">0-5<" in html and ">25-30<" in html
    assert "25.0%" in html  # 2 deaths / 8 entrained in the 0-5 cm class
    assert helper(None, None) == ""

    html_with_spill = helper(daily, length, has_spillway_route=True)
    assert "downstream via the facility's spillway" in html_with_spill
    assert "no spillway mapped" not in html_with_spill
