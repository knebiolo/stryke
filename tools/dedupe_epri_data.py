"""One-time cleanup: remove duplicate rows from Data/epri1997.csv.

Root cause: the original Access data assembly used a one-to-many join that
fanned out rows for any (ID, Species) with multiple candidate geographic
matches. For every duplicated group, only latitude/drainageArea/maxDischarge
differ -- every substantive column (Present, Total, FishPerMft3, taxonomy) is
identical across the dupes, and those three columns are not consumed anywhere
else in the codebase (confirmed via repo-wide grep). Keeping the first row
per (ID, Species) is therefore a safe fix with no downstream side effects
beyond correcting the sample weighting used by distribution fitting.

Run once:
    python tools/dedupe_epri_data.py
"""
import os
import pandas as pd

DATA_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "epri1997.csv")
)


def main():
    df = pd.read_csv(DATA_PATH, encoding="unicode_escape")
    n_before = len(df)
    deduped = df.drop_duplicates(subset=["ID", "Species"], keep="first")
    n_after = len(deduped)
    deduped.to_csv(DATA_PATH, index=False, encoding="unicode_escape")
    print(f"epri1997.csv: {n_before} -> {n_after} rows ({n_before - n_after} duplicate rows removed)")


if __name__ == "__main__":
    main()
