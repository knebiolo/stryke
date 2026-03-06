import pandas as pd

from Stryke.stryke import simulation


def _make_unit_params():
    return pd.DataFrame(
        [{"fb_depth": 10.0, "submergence_depth": 5.0}],
        index=pd.Index(["UnitA"], name="Unit"),
    )


def test_node_surv_rate_tracks_impingement_as_counters():
    sim = simulation.__new__(simulation)
    sim.unit_params = _make_unit_params()
    sim.pop = pd.DataFrame([{"vertical_habitat": "Pelagic", "beta_0": -4.0, "beta_1": 3.0}])
    sim.Kaplan = lambda length, params: 1.0

    u_param_dict = {
        "UnitA": {
            "intake_vel": 2.0,
            "rack_spacing": 0.1,
            "H": 10.0,
            "RPM": 100.0,
            "D": 5.0,
            "ada": 0.9,
            "N": 4.0,
            "Qopt": 50.0,
            "Qper": 0.5,
            "_lambda": 0.2,
        }
    }

    # width = length * width_ratio = 2.0 * 0.2 = 0.4 > rack_spacing => blocked
    # u_crit < intake_vel => imp_surv_prob = 0 => deterministic impingement mortality cause
    for _ in range(3):
        sim.node_surv_rate(
            length=2.0,
            u_crit=0.5,
            status=1,
            surv_fun="Kaplan",
            route="UnitA",
            surv_dict={},
            u_param_dict=u_param_dict,
            barotrauma=False,
            width_ratio=0.2,
        )

    assert sim._mortality_components["impingement"] == 3
    assert sim._mortality_components["blade_strike"] == 0
    assert sim._mortality_components["barotrauma"] == 0
    assert sim._mortality_components["latent"] == 0


def test_node_surv_rate_survival_does_not_increment_counters():
    sim = simulation.__new__(simulation)
    sim.unit_params = _make_unit_params()
    sim.pop = pd.DataFrame([{"vertical_habitat": "Pelagic", "beta_0": -4.0, "beta_1": 3.0}])
    sim.Kaplan = lambda length, params: 1.0

    u_param_dict = {
        "UnitA": {
            "intake_vel": 0.1,
            "rack_spacing": 1.0,
            "H": 10.0,
            "RPM": 100.0,
            "D": 5.0,
            "ada": 0.9,
            "N": 4.0,
            "Qopt": 50.0,
            "Qper": 0.5,
            "_lambda": 0.2,
        }
    }

    for _ in range(5):
        sim.node_surv_rate(
            length=1.0,
            u_crit=2.0,
            status=1,
            surv_fun="Kaplan",
            route="UnitA",
            surv_dict={},
            u_param_dict=u_param_dict,
            barotrauma=False,
            width_ratio=0.1,
        )

    assert sim._mortality_components["impingement"] == 0
    assert sim._mortality_components["blade_strike"] == 0
    assert sim._mortality_components["barotrauma"] == 0
    assert sim._mortality_components["latent"] == 0

