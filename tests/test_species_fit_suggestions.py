import numpy as np
import pandas as pd
import pytest

from Stryke.stryke import epri
from tools.generate_species_fit_suggestions import (
    fit_distributions,
    parse_preset_filters,
    run_fits_and_select_best,
)


def test_parse_preset_filters_uses_meteorological_seasons():
    assert parse_preset_filters("Sander, Great Lakes, Met Fall & Winter") == (
        "Sander",
        (1, 2, 9, 10, 11, 12),
        4,
    )
    assert parse_preset_filters(
        "Oncorhynchus, Great Lakes, Met Spring, Summer, & Fall"
    ) == ("Oncorhynchus", tuple(range(3, 12)), 4)
    assert parse_preset_filters("Coregonus, Great Lakes, Annual") == (
        "Coregonus",
        tuple(range(1, 13)),
        4,
    )


def test_parse_preset_filters_rejects_unrecognized_names():
    with pytest.raises(ValueError, match="Unsupported species-default name"):
        parse_preset_filters("Sander, All Regions, Met Spring")

    with pytest.raises(ValueError, match="Unknown season"):
        parse_preset_filters("Sander, Great Lakes, Met Monsoon")


def test_seasonal_query_matches_only_genus_region_month_and_present_rows():
    name = "Sander, Great Lakes, Met Spring & Summer"
    genus, months, huc02 = parse_preset_filters(name)
    fish = epri(Genus=genus, HUC02=[huc02], Month=list(months))

    source = pd.read_csv(
        "Data/epri1997.csv",
        encoding="utf-8-sig",
    )
    source.columns = source.columns.str.replace("\ufeff", "", regex=False).str.strip()
    expected = source[
        (source["Genus"] == genus)
        & (source["HUC02"] == huc02)
        & source["Month"].isin(months)
        & (source["Present"] == 1)
    ]

    assert set(map(tuple, fish.epri[["ID", "Species"]].to_numpy())) == set(
        map(tuple, expected[["ID", "Species"]].to_numpy())
    )
    assert fish.presence == pytest.approx(len(expected) / len(source[
        (source["Genus"] == genus)
        & (source["HUC02"] == huc02)
        & source["Month"].isin(months)
    ]), abs=0.0001)


def test_fit_distributions_returns_finite_positive_parameters():
    observations = np.array(
        [0.003, 0.006, 0.008, 0.012, 0.019, 0.03, 0.041, 0.075, 0.11, 0.18]
    )

    best_name, params, metrics, reason, criterion = fit_distributions(observations)

    assert best_name in {"Pareto", "Log Normal", "Weibull"}
    assert params[0] > 0
    assert params[1] == 0
    assert params[2] > 0
    assert set(metrics) == {"Pareto", "Log Normal", "Weibull"}
    assert all(np.isfinite(result["aicc"]) for result in metrics.values())
    assert reason
    assert criterion == "AICc"


def test_fit_distributions_uses_aic_when_aicc_is_undefined():
    observations = np.array([0.01, 0.025, 0.06])

    best_name, params, metrics, reason, criterion = fit_distributions(observations)

    assert best_name in {"Pareto", "Log Normal", "Weibull"}
    assert criterion == "AIC"
    assert params[0] > 0
    assert params[1] == 0
    assert params[2] > 0
    assert "AIC lowest" in reason or "AIC tie" in reason
    assert all(np.isfinite(result["aic"]) for result in metrics.values())


def test_fit_distributions_rejects_zero_rate():
    with pytest.raises(ValueError, match="strictly positive"):
        fit_distributions(np.array([0.0, 0.01, 0.02]))


def test_refit_uses_filtered_rows_and_reports_occurrence_and_cap():
    result = run_fits_and_select_best(
        "Sander, Great Lakes, Met Spring & Summer"
    )

    assert result["huc02"] == 4
    assert result["months"] == (3, 4, 5, 6, 7, 8)
    assert result["n_present"] == 82
    assert result["n_positive"] == 82
    assert result["n_zero"] == 0
    assert result["occur_prob"] == pytest.approx(0.6949)
    assert result["max_ent_rate"] == pytest.approx(5.22)
    assert result["shape"] > 0
    assert result["location"] == pytest.approx(0)
    assert result["scale"] > 0

    import matplotlib.pyplot as plt

    plt.close(result["plot_fig"])
