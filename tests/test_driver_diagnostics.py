import numpy as np
import pandas as pd

from Stryke.stryke import simulation


def test_driver_diagnostics_qopt_metrics(tmp_path):
    proj_dir = tmp_path
    output_name = "diag_test"
    hdf_path = proj_dir / f"{output_name}.h5"

    pop = pd.DataFrame([{"Species": "TestFish"}])
    flow_scens = pd.DataFrame([{"Scenario": "TestScenario"}])
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
    daily = pd.DataFrame(
        [
            {
                "species": "TestFish",
                "scenario": "TestScenario",
                "season": "spring",
                "iteration": 0,
                "day": pd.Timestamp("2026-01-01"),
                "flow": 100.0,
                "pop_size": 10.0,
                "num_entrained": 2.0,
                "num_survived": 1.0,
            }
        ]
    )
    route_flows = pd.DataFrame(
        [
            {
                "scenario": "TestScenario",
                "iteration": 0,
                "day": pd.Timestamp("2026-01-01"),
                "route": "UnitA",
                "discharge_cfs": 80.0,
            },
            {
                "scenario": "TestScenario",
                "iteration": 0,
                "day": pd.Timestamp("2026-01-02"),
                "route": "UnitA",
                "discharge_cfs": 120.0,
            },
        ]
    )
    unit_hours = pd.DataFrame(
        [
            {
                "scenario": "TestScenario",
                "iteration": 0,
                "day": pd.Timestamp("2026-01-01"),
                "route": "UnitA",
                "hours": 10.0,
                "discharge_cfs": 80.0,
            },
            {
                "scenario": "TestScenario",
                "iteration": 0,
                "day": pd.Timestamp("2026-01-02"),
                "route": "UnitA",
                "hours": 10.0,
                "discharge_cfs": 120.0,
            },
        ]
    )

    with pd.HDFStore(hdf_path, mode="w") as store:
        store["Population"] = pop
        store["Flow Scenarios"] = flow_scens
        store["Unit_Parameters"] = unit_params
        store["Daily"] = daily
        store["Route_Flows"] = route_flows
        store["Unit_Hours"] = unit_hours

    sim = simulation.__new__(simulation)
    sim.proj_dir = str(proj_dir)
    sim.output_name = output_name
    sim.moves = np.array([0])
    sim.summary()

    diag = sim.driver_diagnostics
    assert diag is not None
    row = diag.loc[diag["route"] == "UnitA"].iloc[0]
    assert row["total_hours"] == 20.0
    assert row["mean_abs_pct_off_qopt"] == 20.0
    assert row["pct_hours_outside_qopt_band"] == 100.0
