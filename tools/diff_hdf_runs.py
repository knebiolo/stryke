import argparse
import os
import tempfile
import zipfile

import numpy as np
import pandas as pd


def _resolve_hdf_path(path):
    path = os.path.abspath(path)
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path, "r") as zf:
            h5_names = [n for n in zf.namelist() if n.lower().endswith((".h5", ".hdf"))]
            if not h5_names:
                raise FileNotFoundError(f"No HDF found in zip: {path}")
            h5_name = h5_names[0]
            data = zf.read(h5_name)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(h5_name)[1])
        tmp.write(data)
        tmp.flush()
        tmp.close()
        return tmp.name, True
    return path, False


def _load_table(store, key):
    if key in store.keys():
        df = store[key]
        if isinstance(df, pd.DataFrame):
            return df.copy()
    return None


def _numeric_cols(df):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _prepare_index(df, index_col=None):
    out = df.copy()
    if index_col and index_col in out.columns:
        out = out.set_index(index_col, drop=True)
    out.index = out.index.astype(str)
    return out


def _build_driver_diagnostics(unit_params, beta_units, route_flows):
    if unit_params is None or unit_params.empty:
        return None

    diag = unit_params.copy()
    diag["route"] = diag.index.astype(str)

    preferred_cols = [
        "H", "RPM", "D", "N", "Qopt", "Qcap",
        "D1", "D2", "B", "ada", "intake_vel",
        "ps_D", "ps_length", "fb_depth", "submergence_depth"
    ]
    keep_cols = ["route"] + [c for c in preferred_cols if c in diag.columns]
    diag = diag[keep_cols]

    if beta_units is not None and not beta_units.empty:
        beta_units = beta_units.copy()
        rename_map = {}
        if "Passage Route" in beta_units.columns:
            rename_map["Passage Route"] = "route"
        if "state" in beta_units.columns:
            rename_map["state"] = "route"
        if "Mean" in beta_units.columns:
            rename_map["Mean"] = "survival_mean"
        if "survival rate" in beta_units.columns:
            rename_map["survival rate"] = "survival_mean"
        if "Variance" in beta_units.columns:
            rename_map["Variance"] = "survival_variance"
        if "variance" in beta_units.columns:
            rename_map["variance"] = "survival_variance"
        if "Lower 95% CI" in beta_units.columns:
            rename_map["Lower 95% CI"] = "survival_lcl"
        if "ll" in beta_units.columns:
            rename_map["ll"] = "survival_lcl"
        if "Upper 95% CI" in beta_units.columns:
            rename_map["Upper 95% CI"] = "survival_ucl"
        if "ul" in beta_units.columns:
            rename_map["ul"] = "survival_ucl"
        if rename_map:
            beta_units.rename(columns=rename_map, inplace=True)

        if "route" in beta_units.columns:
            beta_keep = [
                c for c in [
                    "route",
                    "survival_mean",
                    "survival_variance",
                    "survival_lcl",
                    "survival_ucl"
                ] if c in beta_units.columns
            ]
            if beta_keep:
                beta_units = beta_units[beta_keep]
                numeric_cols = [c for c in beta_keep if c != "route"]
                if numeric_cols:
                    beta_units = beta_units.groupby("route", as_index=False)[numeric_cols].mean()
                diag = diag.merge(beta_units, on="route", how="left")

    if route_flows is not None and not route_flows.empty:
        rf = route_flows.copy()
        if "route" in rf.columns and "discharge_cfs" in rf.columns:
            rf["route"] = rf["route"].astype(str)
            route_mean = rf.groupby("route")["discharge_cfs"].mean()
            diag = diag.merge(
                route_mean.rename("mean_discharge_cfs"),
                left_on="route",
                right_index=True,
                how="left",
            )
            unit_routes = set(diag["route"].dropna().astype(str).tolist())
            unit_mean = route_mean[route_mean.index.isin(unit_routes)]
            unit_total = float(unit_mean.sum())
            share_series = None
            if unit_total > 0:
                share_series = unit_mean / unit_total
            else:
                total_all = float(route_mean.sum())
                if total_all > 0:
                    share_series = route_mean / total_all
            if share_series is not None:
                diag = diag.merge(
                    share_series.rename("flow_share"),
                    left_on="route",
                    right_index=True,
                    how="left",
                )

    if "flow_share" in diag.columns and "survival_mean" in diag.columns:
        diag["mortality_weight"] = diag["flow_share"] * (1 - diag["survival_mean"])

    return diag


def _diff_frames(left, right, index_col=None, keep_cols=None):
    left = _prepare_index(left, index_col=index_col)
    right = _prepare_index(right, index_col=index_col)
    common_cols = [c for c in left.columns if c in right.columns]
    if keep_cols:
        common_cols = [c for c in common_cols if c in keep_cols]
    num_cols = [c for c in common_cols if c in _numeric_cols(left) and c in _numeric_cols(right)]
    if not num_cols:
        return pd.DataFrame()
    delta = right[num_cols] - left[num_cols]
    delta = delta.replace([np.inf, -np.inf], np.nan)
    delta = delta.fillna(0)
    delta["max_abs"] = delta.abs().max(axis=1)
    delta = delta[delta["max_abs"] > 0]
    return delta.sort_values("max_abs", ascending=False)


def _print_section(title, df, max_rows=10):
    print("\n" + title)
    if df is None or df.empty:
        print("  (no differences)")
        return
    print(df.head(max_rows).to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Compare two Stryke HDF results (HDF or zip) and report key diffs."
    )
    parser.add_argument("left", help="Path to left HDF or zip")
    parser.add_argument("right", help="Path to right HDF or zip")
    parser.add_argument("--top", type=int, default=10, help="Max rows to display per section")
    args = parser.parse_args()

    left_tmp = None
    right_tmp = None
    try:
        left_path, left_tmp = _resolve_hdf_path(args.left)
        right_path, right_tmp = _resolve_hdf_path(args.right)

        with pd.HDFStore(left_path, mode="r") as left_store, pd.HDFStore(right_path, mode="r") as right_store:
            left_units = _load_table(left_store, "/Unit_Parameters")
            right_units = _load_table(right_store, "/Unit_Parameters")

            left_diag = _load_table(left_store, "/Driver_Diagnostics")
            right_diag = _load_table(right_store, "/Driver_Diagnostics")
            if left_diag is None:
                left_diag = _build_driver_diagnostics(
                    left_units,
                    _load_table(left_store, "/Beta_Distributions_Units"),
                    _load_table(left_store, "/Route_Flows"),
                )
            if right_diag is None:
                right_diag = _build_driver_diagnostics(
                    right_units,
                    _load_table(right_store, "/Beta_Distributions_Units"),
                    _load_table(right_store, "/Route_Flows"),
                )

            left_year = _load_table(left_store, "/Yearly_Summary")
            right_year = _load_table(right_store, "/Yearly_Summary")

        print(f"Left:  {left_path}")
        print(f"Right: {right_path}")

        if left_units is not None and right_units is not None:
            unit_keep = [
                "H", "RPM", "D", "N", "Qopt", "Qcap",
                "D1", "D2", "B", "ada", "intake_vel",
                "ps_D", "ps_length", "fb_depth", "submergence_depth",
                "Penstock_Qcap"
            ]
            unit_diff = _diff_frames(left_units, right_units, index_col=None, keep_cols=unit_keep)
            _print_section("Unit_Parameters diffs (right - left)", unit_diff, max_rows=args.top)
        else:
            print("\nUnit_Parameters diffs (right - left)")
            print("  (missing Unit_Parameters in one or both files)")

        if left_diag is not None and right_diag is not None:
            diag_keep = [
                "flow_share", "mean_discharge_cfs", "survival_mean",
                "survival_lcl", "survival_ucl", "mortality_weight"
            ]
            diag_diff = _diff_frames(left_diag, right_diag, index_col="route", keep_cols=diag_keep)
            _print_section("Driver_Diagnostics diffs (right - left)", diag_diff, max_rows=args.top)
        else:
            print("\nDriver_Diagnostics diffs (right - left)")
            print("  (missing Driver_Diagnostics in one or both files)")

        if left_year is not None and right_year is not None:
            year_left = left_year.copy()
            year_right = right_year.copy()
            if "species" in year_left.columns and "scenario" in year_left.columns:
                year_left = year_left.set_index(["species", "scenario"])
                year_right = year_right.set_index(["species", "scenario"])
            year_keep = [
                "prob_entrainment",
                "mean_yearly_entrainment",
                "mean_yearly_mortality",
                "1_in_10_day_entrainment",
                "1_in_100_day_entrainment",
                "1_in_1000_day_entrainment",
                "1_in_10_day_mortality",
                "1_in_100_day_mortality",
                "1_in_1000_day_mortality",
            ]
            year_diff = _diff_frames(year_left, year_right, index_col=None, keep_cols=year_keep)
            _print_section("Yearly_Summary diffs (right - left)", year_diff, max_rows=args.top)
        else:
            print("\nYearly_Summary diffs (right - left)")
            print("  (missing Yearly_Summary in one or both files)")

    finally:
        for tmp_path in [left_tmp, right_tmp]:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


if __name__ == "__main__":
    main()
