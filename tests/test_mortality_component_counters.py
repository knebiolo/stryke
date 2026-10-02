import numpy as np
import pandas as pd

from Stryke.stryke import (
    CAUSE_BAROTRAUMA,
    CAUSE_BLADE_STRIKE,
    CAUSE_IMPINGEMENT,
    CAUSE_NONE,
    CAUSE_OTHER,
    CAUSE_SCREENED,
    _accumulate_length_bins,
    _attribute_mortality_cause,
    simulation,
    summarize_length_bins,
)


def _make_sim():
    sim = simulation.__new__(simulation)
    sim.unit_params = pd.DataFrame(
        [{"fb_depth": 10.0, "submergence_depth": 5.0}],
        index=pd.Index(["UnitA"], name="Unit"),
    )
    sim.pop = pd.DataFrame([{"vertical_habitat": "Pelagic", "beta_0": -4.0, "beta_1": 3.0}])
    sim.Kaplan = lambda length, params: 0.8
    return sim


def _unit(intake_vel, rack_spacing):
    return {"UnitA": {"intake_vel": intake_vel, "rack_spacing": rack_spacing}}


def test_node_surv_components_impinged_fish():
    sim = _make_sim()
    # width 2.0 * 0.2 = 0.4 > rack 0.1 and u_crit < intake_vel => impinged
    prob, imp, strike, baro, screened = sim.node_surv_components(
        2.0, 0.5, 1, "Kaplan", "UnitA", {}, _unit(2.0, 0.1), barotrauma=False, width_ratio=0.2
    )
    assert prob == 0.0 and imp == 0.0 and strike == 1.0 and baro == 1.0 and screened == 0.0
    assert sim.node_surv_rate(
        2.0, 0.5, 1, "Kaplan", "UnitA", {}, _unit(2.0, 0.1), barotrauma=False, width_ratio=0.2
    ) == prob


def test_node_surv_components_strike_and_a_priori_and_dead():
    sim = _make_sim()
    prob, imp, strike, baro, screened = sim.node_surv_components(
        1.0, 2.0, 1, "Kaplan", "UnitA", {}, _unit(0.1, 1.0), barotrauma=False, width_ratio=0.1
    )
    assert np.isclose(prob, 0.8) and imp == 1.0 and strike == 0.8 and baro == 1.0 and screened == 0.0

    prob, imp, strike, baro, screened = sim.node_surv_components(
        1.0, 2.0, 1, "a priori", "spill", {"spill": 0.9}, {}, width_ratio=0.1
    )
    assert np.isclose(prob, 0.9) and np.isnan(imp) and np.isnan(strike) and np.isnan(baro)
    assert np.isnan(screened)

    prob, imp, *_ = sim.node_surv_components(
        1.0, 2.0, 0, "Kaplan", "UnitA", {}, _unit(0.1, 1.0), width_ratio=0.1
    )
    assert prob == 0.0 and np.isnan(imp)


def test_node_surv_components_screened_fish_is_not_struck():
    sim = _make_sim()
    # width 0.4 > rack 0.1 but u_crit 3.0 >= intake_vel 2.0 => screened, survives
    prob, imp, strike, baro, screened = sim.node_surv_components(
        2.0, 3.0, 1, "Kaplan", "UnitA", {}, _unit(2.0, 0.1), barotrauma=False, width_ratio=0.2
    )
    assert prob == 1.0 and imp == 1.0 and strike == 1.0 and screened == 1.0
    # equal swim speed and intake velocity also escapes (u_crit < intake_vel is impingement)
    *_, screened = sim.node_surv_components(
        2.0, 2.0, 1, "Kaplan", "UnitA", {}, _unit(2.0, 0.1), barotrauma=False, width_ratio=0.2
    )
    assert screened == 1.0


def test_attribute_cause_screened_survivors():
    dice = np.array([0.5, 0.5, 0.5])
    rates = np.float32([1.0, 1.0, 0.0])
    status = np.array([1, 1, 1])
    imp = np.array([1.0, 1.0, 0.0])
    strike = np.array([1.0, 1.0, 1.0])
    screened = np.array([1.0, 0.0, 0.0])
    cause = _attribute_mortality_cause(dice, rates, status, imp, strike, screened)
    assert list(cause) == [CAUSE_SCREENED, CAUSE_NONE, CAUSE_IMPINGEMENT]


def test_attribute_cause_uses_same_draw():
    strike = np.array([1.0, 0.8, 0.8, 0.8, 0.8, np.nan, 0.8])
    baro = np.array([1.0, 1.0, 0.5, 0.5, 1.0, np.nan, 0.5])
    imp = np.array([0.0, 1.0, 1.0, 1.0, 1.0, np.nan, 1.0])
    rates = np.float32(imp * strike * baro)
    rates[5] = np.float32(0.9)  # a priori node
    dice = np.array([0.5, 0.9, 0.9, 0.5, 0.7, 0.95, 0.99])
    status = np.array([1, 1, 1, 1, 1, 1, 0])

    cause = _attribute_mortality_cause(dice, rates, status, imp, strike)

    assert list(cause) == [
        CAUSE_IMPINGEMENT,   # imp = 0
        CAUSE_BLADE_STRIKE,  # dice above strike survival
        CAUSE_BLADE_STRIKE,
        CAUSE_BAROTRAUMA,    # strike*baro < dice <= strike
        CAUSE_NONE,          # survived
        CAUSE_OTHER,         # died at a priori node
        CAUSE_NONE,          # already dead
    ]
    died = (status == 1) & (dice > rates)
    assert np.array_equal(died, cause != CAUSE_NONE)


def test_attribute_cause_float32_rounding_is_blade_strike():
    strike = np.array([0.1 + 1e-12])
    rates = np.float32(strike)
    dice = np.array([float(rates[0]) + 1e-12])
    cause = _attribute_mortality_cause(dice, rates, np.array([1]), np.array([1.0]), strike)
    assert cause[0] == CAUSE_BLADE_STRIKE


def test_length_bins_accumulate_and_summarize():
    length_cm = np.array([2.5, 2.9, 10.1, 40.0, 5.0])
    entrained = np.array([True, True, True, True, False])
    survived = np.array([True, False, False, False, False])
    cause = np.array([0, CAUSE_BLADE_STRIKE, CAUSE_BAROTRAUMA, CAUSE_IMPINGEMENT, 0], dtype=np.int8)

    acc = _accumulate_length_bins(None, length_cm, entrained, survived, cause, 5.0)
    acc = _accumulate_length_bins(acc, np.array([52.0]), np.array([True]), np.array([True]),
                                  np.array([0], dtype=np.int8), 5.0)
    assert acc[0].tolist() == [2, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1]
    assert acc[1].tolist() == [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]
    assert acc[2, 8] == 1 and acc[3, 0] == 1 and acc[4, 2] == 1
    assert (acc[1] == acc[2:6].sum(axis=0)).all()
    assert acc[6].sum() == 0

    acc = _accumulate_length_bins(acc, np.array([30.0, 61.0]), np.array([False, False]),
                                  np.array([True, True]), np.array([0, 0], dtype=np.int8), 5.0,
                                  escaped=np.array([True, True]))
    assert acc.shape[1] == 13 and acc[6, 6] == 1 and acc[6, 12] == 1 and acc[0].sum() == 5

    raw = pd.DataFrame([
        {"scenario": "Spring", "species": "Sucker", "iteration": 0, "length_bin_lower_cm": 0.0,
         "length_bin_upper_cm": 5.0, "num_entrained": 2, "num_mortality": 1,
         "mortality_impingement": 0, "mortality_blade_strike": 1, "mortality_barotrauma": 0,
         "mortality_other": 0},
        {"scenario": "Spring", "species": "Sucker", "iteration": 2, "length_bin_lower_cm": 0.0,
         "length_bin_upper_cm": 5.0, "num_entrained": 4, "num_mortality": 3,
         "mortality_impingement": 0, "mortality_blade_strike": 2, "mortality_barotrauma": 1,
         "mortality_other": 0},
    ])
    summ = summarize_length_bins(raw, {("Spring", "Sucker"): 4})
    row = summ.iloc[0]
    assert row["iterations"] == 4
    assert row["mean_num_entrained"] == 1.5
    assert row["mean_num_mortality"] == 1.0
    assert row["mortality_rate"] == 4 / 6
