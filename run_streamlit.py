import re
from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from typing import Optional

tab1, tab1b, tab2, tab3 = st.tabs(
    ["MAPE vs test size", "MAPE vs test size (multi-method)", "Run model", "Best MAPE summary"]
)

# -------------------------
# Shared helpers (you already have most of these)
# -------------------------
ROOT = Path(".")
RESULTS_DIR = ROOT / "results"


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)


def find_mape_col(df: pd.DataFrame):
    for c in df.columns:
        if "mape" in str(c).lower():
            return c
    return None


def extract_test_size_from_name(name: str):
    m = re.search(r"_([0-9]+)$", Path(name).stem)
    return int(m.group(1)) if m else None


def extract_route_from_name_any(name: str):
    # matches: <method>_trials_chile_<ROUTE>_...
    m = re.search(r"_trials_chile_([A-Z]+)_", Path(name).stem)
    return m.group(1) if m else None


def extract_method_from_name(name: str):
    # first token: lstm / sarima / tft / transformer
    return Path(name).stem.split("_", 1)[0].lower()


def list_routes_any(results_dir: Path) -> list[str]:
    routes = set()
    for p in results_dir.glob("*"):
        if p.is_file():
            r = extract_route_from_name_any(p.name)
            if r:
                routes.add(r)
    return sorted(routes)


def list_methods(results_dir: Path) -> list[str]:
    methods = set()
    for p in results_dir.glob("*"):
        if p.is_file():
            m = extract_method_from_name(p.name)
            if m:
                methods.add(m)
    return sorted(methods)


def build_best_mape_multimethod(
        results_dir: Path,
        filename_filter: str = "",
        route: Optional[str] = None,
        method: Optional[str] = None,
) -> pd.DataFrame:
    rows = []
    for p in sorted(results_dir.glob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in [".csv", ".xlsx", ".xls"]:
            continue
        if filename_filter and filename_filter.lower() not in p.name.lower():
            continue

        mth = extract_method_from_name(p.name)
        r = extract_route_from_name_any(p.name)
        if route and r != route:
            continue
        if method and mth != method:
            continue

        test_size = extract_test_size_from_name(p.name)
        if test_size is None or r is None or mth is None:
            continue

        try:
            df = read_table(p)
        except Exception:
            continue
        if df is None or df.empty:
            continue

        mape_col = find_mape_col(df)
        if mape_col is None:
            continue

        mape = pd.to_numeric(df[mape_col], errors="coerce")
        if mape.dropna().empty:
            continue

        best_idx = mape.idxmin()

        rows.append(
            {
                "file": p.name,
                "method": mth,
                "route": r,
                "test_size": int(test_size),
                "best_mape": float(mape.loc[best_idx]),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # if multiple files per (method, route, test_size), keep best
    out = out.sort_values(["method", "route", "test_size", "best_mape"])
    out = out.groupby(["method", "route", "test_size"], as_index=False).first()
    return out


# -------------------------
# NEW TAB: Multi-method version of Tab1
# -------------------------
with tab1b:
    st.subheader("MAPE evolution vs test size (multi-method)")

    filename_filter = st.text_input("Filter files (optional)", value="_trials_", key="tab1b_filter")

    methods = list_methods(RESULTS_DIR)
    routes = list_routes_any(RESULTS_DIR)

    c1, c2 = st.columns(2)
    with c1:
        method_choice = st.selectbox("Method", options=["All"] + methods, key="tab1b_method")
    with c2:
        route_choice = st.selectbox("Route", options=["All"] + routes, key="tab1b_route")

    df = build_best_mape_multimethod(
        RESULTS_DIR,
        filename_filter=filename_filter,
        route=None if route_choice == "All" else route_choice,
        method=None if method_choice == "All" else method_choice,
    )

    if df.empty:
        st.warning("No matching files found.")
    else:
    # -------- Plot logic (same idea as Tab1, but method-aware) --------
        fig, ax = plt.subplots()

        if method_choice == "All" and route_choice == "All":
            for (mth, r), g in df.groupby(["method", "route"]):
                g = g.sort_values("test_size")
                ax.plot(g["test_size"], g["best_mape"], marker="o", label=f"{mth}-{r}")
            ax.set_title("Best MAPE vs test size (method-route)")

        elif method_choice != "All" and route_choice == "All":
            dfx = df[df["method"] == method_choice]
            for r, g in dfx.groupby("route"):
                g = g.sort_values("test_size")
                ax.plot(g["test_size"], g["best_mape"], marker="o", label=r)
            ax.set_title(f"Best MAPE vs test size ({method_choice}, by route)")

        elif method_choice == "All" and route_choice != "All":
            dfx = df[df["route"] == route_choice]
            for mth, g in dfx.groupby("method"):
                g = g.sort_values("test_size")
                ax.plot(g["test_size"], g["best_mape"], marker="o", label=mth)
            ax.set_title(f"Best MAPE vs test size ({route_choice}, by method)")

        else:
            g = df.sort_values("test_size")
            ax.plot(g["test_size"], g["best_mape"], marker="o")
            ax.set_title(f"Best MAPE vs test size ({method_choice}, {route_choice})")

        ax.set_xlabel("test_size")
        ax.set_ylabel("best MAPE (min per file)")
        ax.legend()
        st.pyplot(fig)

        st.dataframe(df.sort_values(["method", "route", "test_size"]), use_container_width=True)


# -------- Plot logic (same idea as Tab1, but method-aware) --------
    fig, ax = plt.subplots()

    if method_choice == "All" and route_choice == "All":
        # one line per (method, route)
        for (mth, r), g in df.groupby(["method", "route"]):
            g = g.sort_values("test_size")
            ax.plot(g["test_size"], g["best_mape"], marker="o", label=f"{mth}-{r}")
        ax.set_title("Best MAPE vs test size (method-route)")

    elif method_choice != "All" and route_choice == "All":
        # one line per route (within method)
        dfx = df[df["method"] == method_choice]
        for r, g in dfx.groupby("route"):
            g = g.sort_values("test_size")
            ax.plot(g["test_size"], g["best_mape"], marker="o", label=r)
        ax.set_title(f"Best MAPE vs test size ({method_choice}, by route)")

    elif method_choice == "All" and route_choice != "All":
        # one line per method (within route)
        dfx = df[df["route"] == route_choice]
        for mth, g in dfx.groupby("method"):
            g = g.sort_values("test_size")
            ax.plot(g["test_size"], g["best_mape"], marker="o", label=mth)
        ax.set_title(f"Best MAPE vs test size ({route_choice}, by method)")

    else:
        # single line
        g = df.sort_values("test_size")
        ax.plot(g["test_size"], g["best_mape"], marker="o")
        ax.set_title(f"Best MAPE vs test size ({method_choice}, {route_choice})")

    ax.set_xlabel("test_size")
    ax.set_ylabel("best MAPE (min per file)")
    ax.legend()
    st.pyplot(fig)

    st.dataframe(df.sort_values(["method", "route", "test_size"]), use_container_width=True)
