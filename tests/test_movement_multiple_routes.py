import networkx as nx
import pandas as pd

from Stryke.stryke import simulation


def _make_sim_with_spillway():
    sim = simulation.__new__(simulation)
    sim.facility_params = pd.DataFrame(
        [{"Spillway": "spillway"}],
        index=pd.Index(["FacilityA"], name="Facility"),
    )
    return sim


def test_multiple_env_bypass_nodes_split_flow():
    sim = _make_sim_with_spillway()
    G = nx.DiGraph()
    G.add_edge("river_node_0", "env_flow_a")
    G.add_edge("river_node_0", "env_flow_b")
    G.add_edge("river_node_0", "bypass_flow_a")
    G.add_edge("river_node_0", "bypass_flow_b")
    G.add_edge("river_node_0", "spillway")

    Q_dict = {
        "curr_Q": 100.0,
        "min_Q": {"FacilityA": 0.0},
        "sta_cap": {"FacilityA": 0.0},
        "env_Q": {"FacilityA": 20.0},
        "bypass_Q": {"FacilityA": 10.0},
    }
    route_flow_recorder = {}
    sim.movement(
        "river_node_0",
        1,
        0.0,
        G,
        {},
        Q_dict,
        {},
        {},
        {},
        route_flow_recorder=route_flow_recorder,
    )

    assert route_flow_recorder["env_flow_a"] == 10.0
    assert route_flow_recorder["env_flow_b"] == 10.0
    assert route_flow_recorder["bypass_flow_a"] == 5.0
    assert route_flow_recorder["bypass_flow_b"] == 5.0
    assert route_flow_recorder["spillway"] == 70.0
    assert sum(route_flow_recorder.values()) == 100.0
