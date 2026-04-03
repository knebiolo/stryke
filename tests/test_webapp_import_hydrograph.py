import pytest

from Stryke.stryke import simulation


def _base_data(local_tmp_path, flow_value):
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
                {
                    "Location": "river_node_0",
                    "Surv_Fun": "a-priori",
                    "Survival": 1.0,
                }
            ],
            "Edges": [],
        },
        "flow_scenarios": [
            {
                "Scenario": "S1",
                "Scenario Number": 1,
                "Season": "spring",
                "Months": "1",
                "Flow": flow_value,
                "Gage": None,
                "FlowYear": None,
                "Prorate": 1,
            }
        ],
    }


def test_webapp_import_static_flow_allows_missing_hydrograph(local_tmp_path):
    sim = simulation.__new__(simulation)
    data = _base_data(local_tmp_path, flow_value=125.0)
    data["hydrograph_file"] = None

    sim.webapp_import(data, output_name="out")

    assert sim.input_hydrograph_df is None
    assert (local_tmp_path / "out.h5").exists()


def test_webapp_import_hydrograph_flow_requires_hydrograph_file(local_tmp_path):
    sim = simulation.__new__(simulation)
    data = _base_data(local_tmp_path, flow_value="hydrograph")

    with pytest.raises(ValueError, match="require hydrograph input"):
        sim.webapp_import(data, output_name="out")
