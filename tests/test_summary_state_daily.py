import numpy as np
import pandas as pd

from Stryke.stryke import simulation


def _build_hdf_for_state_daily_summary(hdf_path):
    pop = pd.DataFrame([{"Species": "TestFish"}])
    flow_scens = pd.DataFrame([{"Scenario": "ScenarioA"}])
    unit_params = pd.DataFrame(
        [
            {
                "H": 10.0,
                "Qopt": 100.0,
                "Qcap": 150.0,
                "RPM": 100.0,
                "D": 5.0,
            }
        ],
        index=pd.Index(["UnitA"], name="Unit"),
    )
    # Keep padded labels to validate normalization logic in summary fallback.
    daily = pd.DataFrame(
        [
            {
                "species": "TestFish                                          ",
                "scenario": "ScenarioA                                         ",
                "season": "spring",
                "iteration": 0,
                "day": pd.Timestamp("2026-01-01"),
                "flow": 100.0,
                "pop_size": 10.0,
                "num_entrained": 2.0,
                "num_survived": 1.0,
            },
            {
                "species": "TestFish                                          ",
                "scenario": "ScenarioA                                         ",
                "season": "spring",
                "iteration": 0,
                "day": pd.Timestamp("2026-01-02"),
                "flow": 110.0,
                "pop_size": 12.0,
                "num_entrained": 3.0,
                "num_survived": 2.0,
            },
        ]
    )
    state_daily = pd.DataFrame(
        [
            # day 1
            {
                "scenario": "ScenarioA",
                "species": "TestFish",
                "iteration": 0,
                "day": pd.Timestamp("2026-01-01"),
                "move": 0,
                "state": "river_node_0",
                "successes": 10.0,
                "count": 10.0,
            },
            {
                "scenario": "ScenarioA",
                "species": "TestFish",
                "iteration": 0,
                "day": pd.Timestamp("2026-01-01"),
                "move": 1,
                "state": "UnitA",
                "successes": 7.0,
                "count": 10.0,
            },
            # day 2
            {
                "scenario": "ScenarioA",
                "species": "TestFish",
                "iteration": 0,
                "day": pd.Timestamp("2026-01-02"),
                "move": 0,
                "state": "river_node_0",
                "successes": 12.0,
                "count": 12.0,
            },
            {
                "scenario": "ScenarioA",
                "species": "TestFish",
                "iteration": 0,
                "day": pd.Timestamp("2026-01-02"),
                "move": 1,
                "state": "UnitA",
                "successes": 9.0,
                "count": 12.0,
            },
        ]
    )

    with pd.HDFStore(hdf_path, mode="w") as store:
        store["Population"] = pop
        store["Flow Scenarios"] = flow_scens
        store["Unit_Parameters"] = unit_params
        store["Daily"] = daily
        store["State_Daily"] = state_daily


def test_summary_uses_state_daily_when_sim_table_missing(local_tmp_path):
    output_name = "state_daily_fallback"
    hdf_path = local_tmp_path / f"{output_name}.h5"
    _build_hdf_for_state_daily_summary(hdf_path)

    sim = simulation.__new__(simulation)
    sim.proj_dir = str(local_tmp_path)
    sim.output_name = output_name
    sim.moves = np.array([0, 1])
    sim.summary()

    assert isinstance(sim.beta_df, pd.DataFrame)
    assert not sim.beta_df.empty
    assert "whole" in sim.beta_df["state"].values
    assert "UnitA" in sim.beta_df["state"].values
    assert isinstance(sim.cum_sum, pd.DataFrame)
    assert not sim.cum_sum.empty


def test_summary_state_daily_normalizes_padded_daily_labels(local_tmp_path):
    output_name = "state_daily_padded_labels"
    hdf_path = local_tmp_path / f"{output_name}.h5"
    _build_hdf_for_state_daily_summary(hdf_path)

    sim = simulation.__new__(simulation)
    sim.proj_dir = str(local_tmp_path)
    sim.output_name = output_name
    sim.moves = np.array([0, 1])
    sim.summary()

    # Ensure the yearly summary includes our species/scenario despite padded labels in Daily table.
    assert "TestFish" in set(sim.cum_sum["species"].astype(str).str.strip().tolist())
    assert "ScenarioA" in set(sim.cum_sum["scenario"].astype(str).str.strip().tolist())
