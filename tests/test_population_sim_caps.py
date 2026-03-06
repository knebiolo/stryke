import numpy as np
import pandas as pd
import pytest

import Stryke.stryke as stryke_mod
from Stryke.stryke import simulation


def _species_df(max_ent_rate):
    return pd.DataFrame(
        [
            {
                "shape": 0.5,
                "location": 0.0,
                "scale": 1.0,
                "dist": "Log Normal",
                "max_ent_rate": float(max_ent_rate),
            }
        ]
    )


def test_population_sim_caps_to_one_order_of_magnitude(monkeypatch):
    sim = simulation.__new__(simulation)
    spc_df = _species_df(max_ent_rate=17.0)

    monkeypatch.setattr(
        stryke_mod.lognorm,
        "rvs",
        lambda *args, **kwargs: np.array([1.0e12]),
    )

    n = sim.population_sim(output_units="metric", spc_df=spc_df, curr_Q=100.0)

    daily_rate_mft3 = (60 * 60 * 24 * 100.0) / 1_000_000.0
    expected = float(np.round(daily_rate_mft3 * (17.0 * 10.0), 0))
    assert n == expected


def test_population_sim_fails_fast_on_runaway_population(monkeypatch):
    sim = simulation.__new__(simulation)
    spc_df = _species_df(max_ent_rate=1000.0)

    monkeypatch.setattr(
        stryke_mod.lognorm,
        "rvs",
        lambda *args, **kwargs: np.array([1.0e12]),
    )
    monkeypatch.setenv("STRYKE_MAX_DAILY_FISH", "1000000")

    with pytest.raises(ValueError, match="STRYKE_MAX_DAILY_FISH"):
        sim.population_sim(output_units="metric", spc_df=spc_df, curr_Q=100000.0)

