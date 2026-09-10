import numpy as np
import pandas as pd
import pytest

from Stryke.stryke import simulation


class DummyHDFStore(dict):
    """In-memory stand-in for pd.HDFStore so worksheet_import doesn't touch disk."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def flush(self):
        pass

    def close(self):
        pass


def _base_webapp_data(local_tmp_path, hydrograph_file):
    return {
        "proj_dir": str(local_tmp_path),
        "units_system": "imperial",
        "simulation_mode": "multiple_powerhouses_simulated_entrainment_routing",
        "model_setup": "single_unit_survival_only",
        "project_name": "test",
        "project_notes": "",
        "units": "imperial",
        "graph_summary": {
            "Nodes": [
                {"Location": "river_node_0", "Surv_Fun": "a-priori", "Survival": 1.0}
            ],
            "Edges": [],
        },
        "flow_scenarios": [
            {
                "Scenario": "S1",
                "Scenario Number": 1,
                "Season": "spring",
                "Months": "1",
                "Flow": "hydrograph",
                "Gage": None,
                "FlowYear": None,
                "Prorate": 1,
            }
        ],
        "hydrograph_file": str(hydrograph_file),
    }


def test_webapp_import_rejects_negative_flow_datetimeutc_branch(local_tmp_path):
    hydrograph_file = local_tmp_path / "hydro.csv"
    hydrograph_file.write_text(
        "datetimeUTC,DAvgFlow_prorate\n"
        "2016-03-01,100.0\n"
        "2016-03-02,-50.0\n"
    )

    sim = simulation.__new__(simulation)
    data = _base_webapp_data(local_tmp_path, hydrograph_file)

    with pytest.raises(ValueError, match="negative DAvgFlow_prorate"):
        sim.webapp_import(data, output_name="out")


def test_webapp_import_rejects_negative_flow_date_discharge_branch(local_tmp_path):
    hydrograph_file = local_tmp_path / "hydro.csv"
    hydrograph_file.write_text(
        "Date,Discharge\n"
        "2016-03-01,100.0\n"
        "2016-03-02,-25.0\n"
    )

    sim = simulation.__new__(simulation)
    data = _base_webapp_data(local_tmp_path, hydrograph_file)

    with pytest.raises(ValueError, match="negative DAvgFlow_prorate"):
        sim.webapp_import(data, output_name="out")


def test_worksheet_import_converts_ucrit_to_ft_per_s(monkeypatch, local_tmp_path):
    sheets = {
        "Nodes": pd.DataFrame({"Location": ["A"], "Surv_Fun": ["a priori"]}),
        "Edges": pd.DataFrame({"_from": [], "_to": []}),
        "Unit Params": pd.DataFrame({
            "Unit": [1],
            "Facility": ["FacilityA"],
            "intake_vel": [1.0],
            "H": [1.0], "D": [1.0], "Qopt": [1.0], "Qcap": [10.0],
            "B": [1.0], "D1": [1.0], "D2": [1.0],
            "fb_depth": [1.0], "ps_D": [1.0], "ps_length": [1.0],
            "submergence_depth": [1.0],
        }),
        "Facilities": pd.DataFrame({"Facility": ["FacilityA"], "Rack Spacing": [1.0]}),
        "Flow Scenarios": pd.DataFrame({"Scenario": ["S1"], "Gage": ["g"]}),
        "Hydrology": pd.DataFrame({"Date": ["2020-01-01"], "Discharge": [10.0]}),
        "Operating Scenarios": pd.DataFrame({"Scenario": ["S1"], "Unit": [1], "Hours": [24.0]}),
        "Population": pd.DataFrame({"U_crit": [0.5]}),
    }

    def fake_read_excel(path, sheet_name=None, **kwargs):
        if sheet_name == "Background and Metadata":
            df = pd.DataFrame(np.empty((14, 2), dtype=object))
            df.iat[13, 1] = "metric"
            return df
        return sheets[sheet_name].copy()

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)
    monkeypatch.setattr(pd, "HDFStore", DummyHDFStore)

    sim = simulation.__new__(simulation)
    sim.worksheet_import(str(local_tmp_path), "fake.xlsx", "out")

    # 0.5 m/s -> ft/s, same factor applied to intake_vel elsewhere in this method
    assert sim.pop["U_crit"].iloc[0] == pytest.approx(0.5 * 3.28084)
    assert sim.unit_params["intake_vel"].iloc[0] == pytest.approx(1.0 * 3.28084)


def test_node_surv_rate_sane_when_ucrit_properly_converted():
    sim = simulation.__new__(simulation)
    u_param_dict = {
        "unitA": {"intake_vel": 3.28084, "rack_spacing": 0.1},  # 1 m/s converted to ft/s
    }

    # Fish U_crit = 1.2 m/s, converted to ft/s exceeds intake_vel -> escapes impingement
    u_crit_ft = 1.2 * 3.28084
    prob = sim.node_surv_rate(
        length=2.0,
        u_crit=u_crit_ft,
        status=1,
        surv_fun="Francis",
        route="unitA",
        surv_dict={},
        u_param_dict=u_param_dict,
        barotrauma=True,
    )
    assert prob == 1.0

    # Same fish, but U_crit left unconverted in raw m/s -- reproduces the unit-mismatch bug:
    # 1.2 (m/s) < 3.28084 (ft/s) even though the fish is physically faster than the intake.
    prob_bug = sim.node_surv_rate(
        length=2.0,
        u_crit=1.2,
        status=1,
        surv_fun="Francis",
        route="unitA",
        surv_dict={},
        u_param_dict=u_param_dict,
        barotrauma=True,
    )
    assert prob_bug == 0.0
