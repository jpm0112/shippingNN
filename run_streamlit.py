import re
from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


tab1, tab2 = st.tabs(["MAPE vs test size", "Tab 2"])

with tab1:
    # =========================
    # PASTE ALL YOUR MAPE/ROUTE CODE HERE

    ROOT = Path(".")
    RESULTS_DIR = ROOT / "results"


    def extract_test_size_from_name(name: str):
        m = re.search(r"_([0-9]+)$", Path(name).stem)
        return int(m.group(1)) if m else None


    def extract_route_from_name(name: str):
        stem = Path(name).stem
        # expects: tft_trials_chile_<ROUTE>_YYYYMMDD_..._<testsize>
        m = re.search(r"tft_trials_chile_([A-Z]+)_", stem)
        if m:
            return m.group(1)
        # fallback: take the 4th token if your naming is consistent
        parts = stem.split("_")
        return parts[3] if len(parts) > 3 else None


    def read_table(path: Path) -> pd.DataFrame:
        if path.suffix.lower() in [".xlsx", ".xls"]:
            return pd.read_excel(path)
        return pd.read_csv(path)


    def find_mape_col(df: pd.DataFrame):
        for c in df.columns:
            if "mape" in str(c).lower():
                return c
        return None


    def build_best_mape(results_dir: Path, filename_filter: str = "", route: str | None = None) -> pd.DataFrame:
        rows = []
        for p in sorted(results_dir.glob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() not in [".csv", ".xlsx", ".xls"]:
                continue
            if filename_filter and filename_filter.lower() not in p.name.lower():
                continue

            r = extract_route_from_name(p.name)
            if route and r != route:
                continue

            test_size = extract_test_size_from_name(p.name)
            if test_size is None:
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

            rows.append({
                "file": p.name,
                "route": r,
                "test_size": test_size,
                "best_mape": float(mape.loc[best_idx]),
            })

        out = pd.DataFrame(rows)
        if out.empty:
            return out

        # if multiple files per (route,test_size), keep best mape
        out = out.sort_values(["route", "test_size", "best_mape"])
        out = out.groupby(["route", "test_size"], as_index=False).first()
        return out


    def list_routes(results_dir: Path) -> list[str]:
        routes = set()
        for p in results_dir.glob("*"):
            if p.is_file():
                r = extract_route_from_name(p.name)
                if r:
                    routes.add(r)
        return sorted(routes)


    # ===== UI =====
    st.subheader("MAPE evolution vs test size")

    filename_filter = st.text_input("Filter files (optional)", value="tft_trials")
    routes = list_routes(RESULTS_DIR)
    route_choice = st.selectbox("Route", options=["All"] + routes)

    df = build_best_mape(
        RESULTS_DIR,
        filename_filter=filename_filter,
        route=None if route_choice == "All" else route_choice,
    )

    if df.empty:
        st.warning("No matching files found.")
    else:
        if route_choice == "All":
            # plot one line per route
            fig, ax = plt.subplots()
            for r, g in df.groupby("route"):
                g = g.sort_values("test_size")
                ax.plot(g["test_size"], g["best_mape"], marker="o", label=r)
            ax.set_xlabel("test_size")
            ax.set_ylabel("best MAPE (min per file)")
            ax.set_title("Best MAPE vs test size (by route)")
            ax.legend()
            st.pyplot(fig)
        else:
            g = df.sort_values("test_size")
            fig, ax = plt.subplots()
            ax.plot(g["test_size"], g["best_mape"], marker="o")
            ax.set_xlabel("test_size")
            ax.set_ylabel("best MAPE (min per file)")
            ax.set_title(f"Best MAPE vs test size ({route_choice})")
            st.pyplot(fig)

        st.dataframe(df.sort_values(["route", "test_size"]), use_container_width=True)

    # =========================
    st.subheader("MAPE evolution vs test size")
    # ... everything from filename_filter -> route selector -> plot -> table ...
with tab2:
    st.subheader("Run Darts TFT with best params from a results file")

    import numpy as np
    from functions import error_metrics, run_darts_tft
    from lightning.pytorch import seed_everything
    import matplotlib.pyplot as plt
    from pathlib import Path
    import torch

    # -------------------------
    # Load dataset once
    # -------------------------
    @st.cache_data
    def load_weekly_data(csv_path: str):
        df = pd.read_csv(csv_path)
        df["FECHA"] = pd.to_datetime(df["FECHA"])
        df = df.sort_values("FECHA").reset_index(drop=True)
        return df

    data_path = st.text_input("Data CSV path", value="weekly_chile_data.csv")
    df_full = load_weekly_data(data_path)

    # -------------------------
    # Pick results file + best row
    # -------------------------
    files = sorted([p for p in RESULTS_DIR.glob("*") if p.suffix.lower() in [".csv", ".xlsx", ".xls"]])
    if not files:
        st.warning(f"No result files found in: {RESULTS_DIR}")
        st.stop()

    route2 = st.selectbox(
        "Route (optional)",
        options=["All"] + sorted({extract_route_from_name(p.name) for p in files if extract_route_from_name(p.name)}),
        key="tab2_route",
    )
    files2 = [p for p in files if (route2 == "All" or extract_route_from_name(p.name) == route2)]
    file_choice = st.selectbox("Results file", [p.name for p in files2], key="tab2_file")
    file_path = RESULTS_DIR / file_choice

    trials = read_table(file_path)
    mape_col = find_mape_col(trials)
    if mape_col is None:
        st.error("No column containing 'mape' found in this file.")
        st.stop()

    mape = pd.to_numeric(trials[mape_col], errors="coerce")
    best_idx = mape.idxmin()
    best = trials.loc[best_idx]

    st.caption("Best row (min MAPE)")
    st.write({mape_col: float(mape.loc[best_idx])})
    st.dataframe(best.to_frame("value"), use_container_width=True)

    # -------------------------
    # Column mapping (CSV -> function args)
    # Adjust these keys to match your CSV headers if needed
    # -------------------------
    def pick(colnames):
        for c in colnames:
            if c in trials.columns:
                return c
        return None

    col_test_size = pick(["test_size", "horizon", "H"])
    col_window = pick(["window_size", "n_lags", "input_chunk_length"])
    col_hidden = pick(["hidden_size", "d_model", "hidden_dim"])
    col_lstm_layers = pick(["lstm_layers", "num_lstm_layers"])
    col_heads = pick(["num_attention_heads", "n_head", "n_heads"])
    col_dropout = pick(["dropout"])
    col_batch = pick(["batch_size"])
    col_lr = pick(["lr", "learning_rate"])
    col_epochs = pick(["n_epochs", "epoch_number", "epochs"])
    col_clip = pick(["grad_clip", "gradient_clip_val"])
    col_patience = pick(["patience"])
    col_min_delta = pick(["min_delta"])
    col_seed = pick(["seed"])

    # Manual target (your CSV might not store it)
    numeric_cols = [c for c in df_full.columns if c not in ["FECHA", "series"]]
    target_col = st.selectbox("target_col", options=numeric_cols, index=numeric_cols.index("NE") if "NE" in numeric_cols else 0)

    # Pull params from best row, with fallbacks
    def to_int(x, default):
        try:
            return int(float(x))
        except Exception:
            return default

    def to_float(x, default):
        try:
            return float(x)
        except Exception:
            return default

    test_size = to_int(best[col_test_size], 12) if col_test_size else 12
    window_size = to_int(best[col_window], 26) if col_window else 26
    hidden_size = to_int(best[col_hidden], 64) if col_hidden else 64
    lstm_layers = to_int(best[col_lstm_layers], 1) if col_lstm_layers else 1
    num_attention_heads = to_int(best[col_heads], 1) if col_heads else 1
    dropout = to_float(best[col_dropout], 0.1) if col_dropout else 0.1
    batch_size = to_int(best[col_batch], 32) if col_batch else 32
    lr = to_float(best[col_lr], 1e-3) if col_lr else 1e-3
    n_epochs = to_int(best[col_epochs], 2000) if col_epochs else 2000
    grad_clip = to_float(best[col_clip], 3.0) if col_clip else 3.0
    patience = to_int(best[col_patience], 100) if col_patience else 100
    min_delta = to_float(best[col_min_delta], 1e-5) if col_min_delta else 1e-5
    seed = to_int(best[col_seed], 1048596) if col_seed else 1048596

    st.markdown("### Params (editable)")
    c1, c2, c3 = st.columns(3)
    with c1:
        test_size = st.number_input("test_size", min_value=1, value=int(test_size), step=1)
        window_size = st.number_input("window_size", min_value=1, value=int(window_size), step=1)
        hidden_size = st.number_input("hidden_size", min_value=1, value=int(hidden_size), step=1)
        lstm_layers = st.number_input("lstm_layers", min_value=1, value=int(lstm_layers), step=1)
    with c2:
        num_attention_heads = st.number_input("num_attention_heads", min_value=1, value=int(num_attention_heads), step=1)
        batch_size = st.number_input("batch_size", min_value=1, value=int(batch_size), step=1)
        n_epochs = st.number_input("n_epochs", min_value=1, value=int(n_epochs), step=50)
        seed = st.number_input("seed", min_value=0, value=int(seed), step=1)
    with c3:
        dropout = st.number_input("dropout", min_value=0.0, max_value=0.99, value=float(dropout), step=0.01)
        lr = st.number_input("lr", min_value=0.0, value=float(lr), format="%.10f")
        grad_clip = st.number_input("grad_clip", min_value=0.0, value=float(grad_clip), step=0.5)
        patience = st.number_input("patience", min_value=1, value=int(patience), step=10)
        min_delta = st.number_input("min_delta", min_value=0.0, value=float(min_delta), format="%.10f")

    # -------------------------
    # deleted_sample control
    # -------------------------
    st.markdown("### deleted_sample")
    default_deleted = int(test_size)
    deleted_sample = st.number_input(
        "Delete last N rows from data before training (0 = keep all)",
        min_value=0,
        max_value=int(len(df_full) - 1),
        value=default_deleted,
        step=1,
    )

    run_btn = st.button("Run Darts TFT", type="primary")

    if run_btn:
        seed_everything(int(seed))

        tmp = df_full.copy().sort_values("FECHA")
        if deleted_sample > 0:
            tmp = tmp.iloc[:-int(deleted_sample)]

        with st.spinner("Running..."):
            true_vals, pred_vals, _ = run_darts_tft(
                tmp,
                target_col,
                int(test_size),
                int(window_size),
                int(hidden_size),
                int(lstm_layers),
                int(num_attention_heads),
                float(dropout),
                int(batch_size),
                int(n_epochs),
                float(lr),
                float(grad_clip),
                patience=int(patience),
                min_delta=float(min_delta),
                seed=int(seed),
            )

            mae, mape_v, mse, rmse, r2 = error_metrics(true_vals, pred_vals)

        st.success("Done.")
        st.write({"MAE": mae, "MAPE": mape_v, "MSE": mse, "RMSE": rmse, "R2": r2})

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(true_vals, marker="o", label="Real")
        ax.plot(pred_vals, marker="o", label="Prediction")
        ax.set_title("Prediction (original scale)")
        ax.set_xlabel("Weeks")
        ax.set_ylabel(target_col)
        ax.legend()
        ax.grid(True, linestyle="--", linewidth=0.5)
        st.pyplot(fig)

        save_plot = st.checkbox("Save plot to plots/z_darts_prediction.png", value=True)
        if save_plot:
            (ROOT / "plots").mkdir(exist_ok=True)
            out_path = ROOT / "plots" / "z_darts_prediction.png"
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            st.caption(f"Saved: {out_path}")



