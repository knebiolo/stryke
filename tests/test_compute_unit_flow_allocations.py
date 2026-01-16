import pandas as pd

from Stryke.stryke import simulation


def _make_sim(unit_params):
    sim = simulation.__new__(simulation)
    sim.unit_params = unit_params
    return sim


def test_no_production_below_min_flow():
    unit_params = pd.DataFrame(
        [
            {"Facility": "FacilityA", "op_order": 1, "Qopt": 40.0},
            {"Facility": "FacilityA", "op_order": 2, "Qopt": 40.0},
        ],
        index=pd.Index(["UnitA", "UnitB"], name="Unit"),
    )
    sim = _make_sim(unit_params)
    unit_nodes = ["UnitA", "UnitB"]
    min_Q_dict = {"FacilityA": 30.0}
    env_Q_dict = {"FacilityA": 0.0}
    bypass_Q_dict = {"FacilityA": 0.0}
    sta_cap_dict = {"FacilityA": 80.0}
    unit_fac_dict = {"UnitA": "FacilityA", "UnitB": "FacilityA"}
    penstock_id_dict = {"UnitA": "P1", "UnitB": "P1"}
    penstock_cap_dict = {"P1": 80.0}
    cap_dict = {"UnitA": 40.0, "UnitB": 40.0}

    allocations, prod_by_fac, _, _ = sim.compute_unit_flow_allocations(
        curr_Q=20.0,
        unit_nodes=unit_nodes,
        min_Q_dict=min_Q_dict,
        env_Q_dict=env_Q_dict,
        bypass_Q_dict=bypass_Q_dict,
        sta_cap_dict=sta_cap_dict,
        unit_fac_dict=unit_fac_dict,
        penstock_id_dict=penstock_id_dict,
        penstock_cap_dict=penstock_cap_dict,
        cap_dict=cap_dict,
    )

    assert prod_by_fac["FacilityA"] == 0.0
    assert allocations.get("UnitA", 0.0) == 0.0
    assert allocations.get("UnitB", 0.0) == 0.0


def test_skip_units_below_min_start():
    unit_params = pd.DataFrame(
        [
            {"Facility": "FacilityA", "op_order": 1, "Qopt": 10.0},
            {"Facility": "FacilityA", "op_order": 2, "Qopt": 40.0},
        ],
        index=pd.Index(["UnitA", "UnitB"], name="Unit"),
    )
    sim = _make_sim(unit_params)
    unit_nodes = ["UnitA", "UnitB"]
    min_Q_dict = {"FacilityA": 30.0}
    env_Q_dict = {"FacilityA": 0.0}
    bypass_Q_dict = {"FacilityA": 0.0}
    sta_cap_dict = {"FacilityA": 70.0}
    unit_fac_dict = {"UnitA": "FacilityA", "UnitB": "FacilityA"}
    penstock_id_dict = {"UnitA": "P1", "UnitB": "P1"}
    penstock_cap_dict = {"P1": 70.0}
    cap_dict = {"UnitA": 20.0, "UnitB": 50.0}

    allocations, _, _, _ = sim.compute_unit_flow_allocations(
        curr_Q=100.0,
        unit_nodes=unit_nodes,
        min_Q_dict=min_Q_dict,
        env_Q_dict=env_Q_dict,
        bypass_Q_dict=bypass_Q_dict,
        sta_cap_dict=sta_cap_dict,
        unit_fac_dict=unit_fac_dict,
        penstock_id_dict=penstock_id_dict,
        penstock_cap_dict=penstock_cap_dict,
        cap_dict=cap_dict,
    )

    assert allocations["UnitA"] == 0.0
    assert allocations["UnitB"] == 50.0


def test_penstock_cap_limits_allocations():
    unit_params = pd.DataFrame(
        [
            {"Facility": "FacilityA", "op_order": 1, "Qopt": 50.0},
            {"Facility": "FacilityA", "op_order": 2, "Qopt": 50.0},
        ],
        index=pd.Index(["UnitA", "UnitB"], name="Unit"),
    )
    sim = _make_sim(unit_params)
    unit_nodes = ["UnitA", "UnitB"]
    min_Q_dict = {"FacilityA": 0.0}
    env_Q_dict = {"FacilityA": 0.0}
    bypass_Q_dict = {"FacilityA": 0.0}
    sta_cap_dict = {"FacilityA": 120.0}
    unit_fac_dict = {"UnitA": "FacilityA", "UnitB": "FacilityA"}
    penstock_id_dict = {"UnitA": "P1", "UnitB": "P1"}
    penstock_cap_dict = {"P1": 60.0}
    cap_dict = {"UnitA": 60.0, "UnitB": 60.0}

    allocations, _, _, _ = sim.compute_unit_flow_allocations(
        curr_Q=120.0,
        unit_nodes=unit_nodes,
        min_Q_dict=min_Q_dict,
        env_Q_dict=env_Q_dict,
        bypass_Q_dict=bypass_Q_dict,
        sta_cap_dict=sta_cap_dict,
        unit_fac_dict=unit_fac_dict,
        penstock_id_dict=penstock_id_dict,
        penstock_cap_dict=penstock_cap_dict,
        cap_dict=cap_dict,
    )

    assert allocations["UnitA"] == 50.0
    assert allocations["UnitB"] == 10.0
    assert allocations["UnitA"] + allocations["UnitB"] == 60.0
