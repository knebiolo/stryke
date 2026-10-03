"""The five species-level ecology columns dropped in 2023 (commit 55e9008) are restored in Data/epri1997.csv."""
from pathlib import Path

import pandas as pd

EPRI_CSV = Path(__file__).resolve().parents[1] / "Data" / "epri1997.csv"
ECOLOGY = ["FeedingGuild", "Habitat", "WaterType", "Host", "Migrant"]


def _load():
    return pd.read_csv(EPRI_CSV, encoding="utf-8-sig", low_memory=False)


def test_ecology_columns_and_site_identifiers_are_present():
    cols = _load().columns
    for name in ECOLOGY + ["NIDID", "HUC02", "HUC04", "HUC06", "HUC08"]:
        assert name in cols, f"{name} missing from epri1997.csv"


def test_each_species_has_one_value_per_ecology_attribute():
    df = _load()
    conflicts = (df.groupby("Common")[ECOLOGY].nunique() > 1).any(axis=1)
    assert not conflicts.any(), f"species with conflicting ecology values: {list(conflicts[conflicts].index)}"


def test_ecology_values_cover_nearly_every_row():
    df = _load()
    assert df["FeedingGuild"].notna().mean() > 0.9
    assert df["Habitat"].notna().mean() > 0.9
