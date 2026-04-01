"""Test that empty rows in operating scenarios are handled correctly."""
import numpy as np
import pandas as pd
import pytest

from Stryke.stryke import simulation


def test_run_of_river_missing_operating_scenario_raises():
    """Test that run-of-river fails loud when unit has no operating scenario."""
    sim = simulation.__new__(simulation)
    sim.unit_params = pd.DataFrame(
        [
            {"Facility": "Upper", "Unit": 1, "op_order": 1, "Qcap": 50.0},
        ]
    )
    sim.facility_params = pd.DataFrame(
        [
            {
                "Operations": "run-of-river",
                "Min_Op_Flow": 0.0,
                "Env_Flow": 0.0,
                "Bypass_Flow": 0.0,
            }
        ],
        index=pd.Index(["Upper"], name="Facility"),
    )
    # Operating scenarios has no matching unit (wrong unit number)
    sim.operating_scenarios_df = pd.DataFrame(
        [
            {"Scenario": "LowFlow", "Facility": "Upper", "Unit": 999, "Hours": 24.0},
        ]
    )

    Q_dict = {
        "curr_Q": 100.0,
        "min_Q": {"Upper": 0.0},
        "env_Q": {"Upper": 0.0},
        "bypass_Q": {"Upper": 0.0},
        "sta_cap": {"Upper": 100.0},
    }

    with pytest.raises(ValueError, match="No operating scenario found"):
        sim.daily_hours(Q_dict, "LowFlow")


def test_empty_hours_dict_raises():
    """Test that completely empty hours_dict raises error at validation."""
    sim = simulation.__new__(simulation)
    # This scenario would previously return empty hours_dict silently
    sim.unit_params = pd.DataFrame(
        [
            {"Facility": "Upper", "Unit": 1, "op_order": 1, "Qcap": 50.0},
        ]
    )
    sim.facility_params = pd.DataFrame(
        [
            {
                "Operations": "run-of-river",
                "Min_Op_Flow": 0.0,
                "Env_Flow": 0.0,
                "Bypass_Flow": 0.0,
            }
        ],
        index=pd.Index(["Upper"], name="Facility"),
    )
    # No operating scenarios at all for this facility/scenario
    sim.operating_scenarios_df = pd.DataFrame(
        [
            {"Scenario": "OtherScenario", "Facility": "Other", "Unit": 1, "Hours": 24.0},
        ]
    )

    Q_dict = {
        "curr_Q": 100.0,
        "min_Q": {"Upper": 0.0},
        "env_Q": {"Upper": 0.0},
        "bypass_Q": {"Upper": 0.0},
        "sta_cap": {"Upper": 100.0},
    }

    # Should fail when trying to process facility "Upper" with no ops data
    with pytest.raises(ValueError):
        sim.daily_hours(Q_dict, "LowFlow")
