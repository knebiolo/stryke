"""Regression test for the fixed-discharge hydrograph builder.

create_hydrograph(discharge_type='fixed', ...) used to build and concat the
running DataFrame inside the per-month loop, re-adding every previously
processed month on each pass (3 months -> first month tripled, second
doubled, last correct). Guard against that regressing.
"""
import pandas as pd

from Stryke.stryke import simulation


def _sim():
    return simulation.__new__(simulation)


def test_fixed_discharge_single_month_has_no_duplicates():
    sim = _sim()
    flow_scenarios_df = pd.DataFrame([{"Scenario": "spring"}])
    df = sim.create_hydrograph("fixed", "spring", [4], flow_scenarios_df, fixed_discharge=50.0)
    assert len(df) == 30
    assert df["datetimeUTC"].duplicated().sum() == 0


def test_fixed_discharge_multi_month_has_no_duplicates():
    sim = _sim()
    flow_scenarios_df = pd.DataFrame([{"Scenario": "fall"}])
    df = sim.create_hydrograph("fixed", "fall", [9, 10, 11], flow_scenarios_df, fixed_discharge=388.46)

    # 30 (Sep) + 31 (Oct) + 30 (Nov) = 91 unique days, not 182.
    assert len(df) == 91
    assert df["datetimeUTC"].duplicated().sum() == 0
    counts = df["month"].value_counts().to_dict()
    assert counts == {9: 30, 10: 31, 11: 30}
    assert (df["DAvgFlow_prorate"] == 388.46).all()
