import networkx as nx
import pandas as pd

from Stryke.stryke import simulation


def _make_sim_with_units():
    sim = simulation.__new__(simulation)
    sim.unit_params = pd.DataFrame(
        [
            {
                "Facility": "FacilityA",
                "op_order": 1,
                "Qopt": 60.0,
            }
        ],
        index=pd.Index(["UnitA"], name="Unit"),
    )
    return sim


def test_unit_env_bypass_spill_routing_flows():
    sim = _make_sim_with_units()
    G = nx.DiGraph()
    G.add_edge("river_node_0", "UnitA")
    G.add_edge("river_node_0", "env_flow")
    G.add_edge("river_node_0", "bypass_flow")
    G.add_edge("river_node_0", "spillway")

    Q_dict = {
        "curr_Q": 100.0,
        "min_Q": {"FacilityA": 0.0},
        "sta_cap": {"FacilityA": 60.0},
        "env_Q": {"FacilityA": 20.0},
        "bypass_Q": {"FacilityA": 10.0},
    }
    op_order = {"UnitA": 1}
    cap_dict = {"UnitA": 60.0}
    unit_fac_dict = {"UnitA": "FacilityA"}
    route_flow_recorder = {}

    sim.movement(
        "river_node_0",
        1,
        0.0,
        G,
        {},
        Q_dict,
        op_order,
        cap_dict,
        unit_fac_dict,
        route_flow_recorder=route_flow_recorder,
    )

    assert route_flow_recorder["UnitA"] == 60.0
    assert route_flow_recorder["env_flow"] == 20.0
    assert route_flow_recorder["bypass_flow"] == 10.0
    assert route_flow_recorder["spillway"] == 10.0
