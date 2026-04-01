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


def test_daily_hours_pumped_storage_fixed_hours_mode():
    """Test fixed-hours mode: Hours column used directly when distribution params are zero/missing."""
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
    # Fixed hours mode: shape=0, scale=0, location=0 -> use Hours directly
    sim.operating_scenarios_df = pd.DataFrame(
        [
            {
                "Scenario": "ScenarioA",
                "Facility": "FacilityA",
                "Unit": "1",
                "Hours": 12.0,
                "Prob Not Operating": 0.0,
                "Shape": 0.0,
                "Location": 0.0,
                "Scale": 0.0,
            },
            {
                "Scenario": "ScenarioA",
                "Facility": "FacilityA",
                "Unit": "2",
                "Hours": 8.0,
                "Prob Not Operating": 0.0,
                "Shape": 0.0,
                "Location": 0.0,
                "Scale": 0.0,
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
    tot_hours, tot_flow, hours_dict, flow_dict = sim.daily_hours(Q_dict, "ScenarioA")

    # Fixed hours should be used exactly as specified
    assert hours_dict["FacilityA - Unit 1"] == 12.0
    assert hours_dict["FacilityA - Unit 2"] == 8.0
    assert flow_dict["FacilityA - Unit 1"] == 30.0 * 12.0 * 3600.0
    assert flow_dict["FacilityA - Unit 2"] == 40.0 * 8.0 * 3600.0
    assert tot_hours == 20.0


def test_daily_hours_pumped_storage_stochastic_mode():
    """Test stochastic mode: lognormal distribution used when params are valid."""
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
    # Stochastic mode: valid distribution parameters
    sim.operating_scenarios_df = pd.DataFrame(
        [
            {
                "Scenario": "ScenarioA",
                "Facility": "FacilityA",
                "Unit": "1",
                "Hours": 999.0,  # Should be ignored in stochastic mode
                "Prob Not Operating": 0.0,
                "Shape": 0.5,
                "Location": 0.0,
                "Scale": 6.0,
            },
        ]
    )

    # Mock lognorm to return predictable value
    import Stryke.stryke as stryke_module
    original_rvs = stryke_module.lognorm.rvs
    
    def mock_rvs(*args, **kwargs):
        return np.array([10.0])
    
    stryke_module.lognorm.rvs = mock_rvs
    try:
        Q_dict = {
            "curr_Q": 100.0,
            "min_Q": {"FacilityA": 0.0},
            "env_Q": {"FacilityA": 0.0},
            "bypass_Q": {"FacilityA": 0.0},
            "sta_cap": {"FacilityA": 100.0},
        }
        tot_hours, tot_flow, hours_dict, flow_dict = sim.daily_hours(Q_dict, "ScenarioA")

        # Should use sampled hours (10.0), not Hours column (999.0)
        assert hours_dict["FacilityA - Unit 1"] == 10.0
        assert flow_dict["FacilityA - Unit 1"] == 30.0 * 10.0 * 3600.0
    finally:
        stryke_module.lognorm.rvs = original_rvs


def test_daily_hours_pumped_storage_fixed_mode_invalid_hours_raises():
    """Test that fixed mode raises error if Hours is invalid."""
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
    # Invalid: shape/scale are 0 (fixed mode) but Hours is NaN
    sim.operating_scenarios_df = pd.DataFrame(
        [
            {
                "Scenario": "ScenarioA",
                "Facility": "FacilityA",
                "Unit": "1",
                "Hours": np.nan,
                "Prob Not Operating": 0.0,
                "Shape": 0.0,
                "Location": 0.0,
                "Scale": 0.0,
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

    with pytest.raises(ValueError, match="neither valid distribution parameters nor valid Hours"):
        sim.daily_hours(Q_dict, "ScenarioA")

