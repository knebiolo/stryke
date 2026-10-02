import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EPRI_CSV = REPO_ROOT / "Data" / "epri1997.csv"


def test_epri_csv_has_real_row_breaks_and_valid_shape():
    raw = EPRI_CSV.read_bytes()
    assert b"\\r\\n" not in raw
    assert b"\\n" not in raw
    assert raw.count(b"\n") > 0

    with EPRI_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))

    assert len(rows) > 2
    header = rows[0]
    for row in rows[1:]:
        assert len(row) == len(header), f"Row has {len(row)} columns, expected {len(header)}"
