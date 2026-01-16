import pandas as pd
import pytest

from Stryke.stryke import simulation


def _make_sim(unit_params):
    sim = simulation.__new__(simulation)
    sim.unit_params = unit_params
    sim.pop = pd.DataFrame(
        [{"vertical_habitat": "Pelagic", "beta_0": -4.0, "beta_1": 3.0}]
    )
    return sim


def _francis_params():
    return {
        "intake_vel": 1.0,
        "rack_spacing": 1.0,
        "H": 10.0,
        "RPM": 100.0,
        "D": 5.0,
        "Q": 50.0,
        "Qper": 1.0,
        "ada": 0.9,
        "N": 4.0,
        "iota": 1.0,
        "D1": 3.0,
        "D2": 4.0,
        "B": 0.5,
        "_lambda": 0.2,
    }


def test_barotrauma_requires_fb_depth():
    unit_params = pd.DataFrame(
        [{"Facility": "FacilityA"}],
        index=pd.Index(["UnitA"], name="Unit"),
    )
    sim = _make_sim(unit_params)
    u_param_dict = {"UnitA": _francis_params()}
    with pytest.raises(ValueError, match="fb_depth"):
        sim.node_surv_rate(
            length=1.0,
            u_crit=2.0,
            status=1,
            surv_fun="Francis",
            route="UnitA",
            surv_dict={},
            u_param_dict=u_param_dict,
            barotrauma=True,
        )


def test_barotrauma_requires_tailrace_depth():
    unit_params = pd.DataFrame(
        [{"Facility": "FacilityA", "fb_depth": 10.0}],
        index=pd.Index(["UnitA"], name="Unit"),
    )
    sim = _make_sim(unit_params)
    u_param_dict = {"UnitA": _francis_params()}
    with pytest.raises(ValueError, match="elevation_head|submergence_depth"):
        sim.node_surv_rate(
            length=1.0,
            u_crit=2.0,
            status=1,
            surv_fun="Francis",
            route="UnitA",
            surv_dict={},
            u_param_dict=u_param_dict,
            barotrauma=True,
        )
