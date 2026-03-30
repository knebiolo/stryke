import numpy as np
import pandas as pd
import pytest

from Stryke import stryke as stryke_module
from Stryke.stryke import simulation


def _make_sim():
    sim = simulation.__new__(simulation)
    sim.unit_params = pd.DataFrame(
        [
            {"Facility": "FacilityA", "op_order": 1, "Qcap": 50.0},
            {"Facility": "FacilityA", "op_order": 2, "Qcap": 50.0},
        ],
        index=pd.Index([1, 2], name="Unit"),
    )
    sim.facility_params = pd.DataFrame(
        [
            {
                "Operations": "run-of-river",
                "Min_Op_Flow": 0.0,
                "Env_Flow": 10.0,
                "Bypass_Flow": 0.0,
            }
        ],
        index=pd.Index(["FacilityA"], name="Facility"),
    )
    sim.operating_scenarios_df = pd.DataFrame(
        [
            {"Scenario": "ScenarioA", "Facility": "FacilityA", "Unit": 1, "Hours": 24.0},
            {"Scenario": "ScenarioA", "Facility": "FacilityA", "Unit": 2, "Hours": 24.0},
        ]
    )
    return sim


def test_daily_hours_run_of_river_keys_and_flow():
    sim = _make_sim()
    Q_dict = {
        "curr_Q": 60.0,
        "min_Q": {"FacilityA": 0.0},
        "env_Q": {"FacilityA": 10.0},
        "bypass_Q": {"FacilityA": 0.0},
        "sta_cap": {"FacilityA": 100.0},
    }
    tot_hours, tot_flow, hours_dict, flow_dict = sim.daily_hours(Q_dict, "ScenarioA")

    assert "1" in hours_dict
    assert "2" in hours_dict
    assert hours_dict["1"] == 24.0
    assert hours_dict["2"] == 24.0
    assert flow_dict["1"] == 50.0 * 24.0 * 3600.0
    assert flow_dict["2"] == 0.0
    assert tot_hours == 48.0
    assert tot_flow == 50.0 * 24.0 * 3600.0


def test_daily_hours_pumped_storage_accepts_webapp_headers_and_unit_lookup(monkeypatch):
    sim = simulation.__new__(simulation)
    sim.unit_params = pd.DataFrame(
        [
            {"Facility": "FacilityA", "Unit": "1", "op_order": 1, "Qcap": 30.0},
            {"Facility": "FacilityA", "Unit": "2", "op_order": 2, "Qcap": 40.0},
        ],
        index=pd.Index(["FacilityA - Unit 1", "FacilityA - Unit 2"], name="Unit_Name"),
    )
    sim.facility_params = pd.DataFrame(
        [
            {
                "Operations": "pumped storage",
                "Min_Op_Flow": 0.0,
                "Env_Flow": 0.0,
                "Bypass_Flow": 0.0,
            }
        ],
        index=pd.Index(["FacilityA"], name="Facility"),
    )
    sim.operating_scenarios_df = pd.DataFrame(
        [
            {
                "Scenario": "ScenarioA",
                "Facility": "FacilityA",
                "Unit": "2",
                "Hours": 24.0,
                "Prob Not Operating": 1.0,
                "Shape": 0.0,
                "Location": 0.0,
                "Scale": 0.0,
            },
            {
                "Scenario": "ScenarioA",
                "Facility": "FacilityA",
                "Unit": "1",
                "Hours": 24.0,
                "Prob Not Operating": 0.0,
                "Shape": 0.5,
                "Location": 0.0,
                "Scale": 6.0,
            },
        ],
        index=[20, 10],
    )

    monkeypatch.setattr(
        stryke_module.lognorm,
        "rvs",
        lambda shape, *args, **kwargs: np.full(kwargs.get("size", args[-1] if args else 1), 6.0),
    )
    monkeypatch.setattr(
        stryke_module.np.random,
        "uniform",
        lambda low, high, size: np.array([0.5]),
    )

    Q_dict = {
        "curr_Q": 100.0,
        "min_Q": {"FacilityA": 0.0},
        "env_Q": {"FacilityA": 0.0},
        "bypass_Q": {"FacilityA": 0.0},
        "sta_cap": {"FacilityA": 100.0},
    }
    tot_hours, tot_flow, hours_dict, flow_dict = sim.daily_hours(Q_dict, "ScenarioA")

    assert hours_dict["FacilityA - Unit 1"] == 6.0
    assert hours_dict["FacilityA - Unit 2"] == 0.0
    assert flow_dict["FacilityA - Unit 1"] == 30.0 * 6.0 * 3600.0
    assert flow_dict["FacilityA - Unit 2"] == 0.0
    assert tot_hours == 6.0
    assert tot_flow == 30.0 * 6.0 * 3600.0


def test_daily_hours_pumped_storage_invalid_active_distribution_raises():
    sim = simulation.__new__(simulation)
    sim.unit_params = pd.DataFrame(
        [
            {"Facility": "FacilityA", "Unit": "1", "op_order": 1, "Qcap": 30.0},
        ],
        index=pd.Index(["FacilityA - Unit 1"], name="Unit_Name"),
    )
    sim.facility_params = pd.DataFrame(
        [
            {
                "Operations": "pumped storage",
                "Min_Op_Flow": 0.0,
                "Env_Flow": 0.0,
                "Bypass_Flow": 0.0,
            }
        ],
        index=pd.Index(["FacilityA"], name="Facility"),
    )
    sim.operating_scenarios_df = pd.DataFrame(
        [
            {
                "Scenario": "ScenarioA",
                "Facility": "FacilityA",
                "Unit": "1",
                "Hours": 24.0,
                "Prob Not Operating": 0.0,
                "Shape": 0.0,
                "Location": 0.0,
                "Scale": 6.0,
            },
        ]
    )

    Q_dict = {
        "curr_Q": 100.0,
        "min_Q": {"FacilityA": 0.0},
        "env_Q": {"FacilityA": 0.0},
        "bypass_Q": {"FacilityA": 0.0},
        "sta_cap": {"FacilityA": 100.0},
    }

    with pytest.raises(ValueError, match="lognormal shape must be > 0"):
        sim.daily_hours(Q_dict, "ScenarioA")
