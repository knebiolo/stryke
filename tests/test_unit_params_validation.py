import pandas as pd
import pytest

from Stryke.stryke import simulation


def _make_sim(unit_params):
    sim = simulation.__new__(simulation)
    sim.unit_params = unit_params
    return sim


def test_validate_unit_params_missing_qopt():
    df = pd.DataFrame(
        [{"Facility": "FacilityA", "Unit": 1, "Qcap": 100.0}]
    ).set_index(pd.Index(["FacilityA - Unit 1"], name="Unit_Name"))
    sim = _make_sim(df)
    with pytest.raises(ValueError, match="Qopt"):
        sim._validate_unit_params_required_fields()


def test_validate_unit_params_missing_qcap():
    df = pd.DataFrame(
        [{"Facility": "FacilityA", "Unit": 1, "Qopt": 80.0}]
    ).set_index(pd.Index(["FacilityA - Unit 1"], name="Unit_Name"))
    sim = _make_sim(df)
    with pytest.raises(ValueError, match="Qcap"):
        sim._validate_unit_params_required_fields()
