import pandas as pd

from Stryke.stryke import simulation


def _make_sim():
    sim = simulation.__new__(simulation)
    sim.unit_params = pd.DataFrame()
    return sim


def test_blocked_by_rack_impingement_only():
    sim = _make_sim()
    u_param_dict = {
        "unitA": {
            "intake_vel": 2.0,
            "rack_spacing": 0.1,
        }
    }
    prob = sim.node_surv_rate(
        length=2.0,
        u_crit=1.0,
        status=1,
        surv_fun="Francis",
        route="unitA",
        surv_dict={},
        u_param_dict=u_param_dict,
        barotrauma=True,
    )
    assert prob == 0.0


def test_blocked_by_rack_high_swim_speed_survives():
    sim = _make_sim()
    u_param_dict = {
        "unitA": {
            "intake_vel": 2.0,
            "rack_spacing": 0.1,
        }
    }
    prob = sim.node_surv_rate(
        length=2.0,
        u_crit=3.0,
        status=1,
        surv_fun="Francis",
        route="unitA",
        surv_dict={},
        u_param_dict=u_param_dict,
        barotrauma=True,
    )
    assert prob == 1.0
