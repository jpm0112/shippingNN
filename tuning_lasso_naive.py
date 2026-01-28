import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from collections import deque
from datetime import datetime
from pathlib import Path

from functions import error_metrics

# ============================================================
# Settings
# ============================================================
country = "chile"
target_cols = ["FE", "NAE", "NAW", "NE", "SE", "SAW", "SAE"]

for target_col in target_cols:
    print(f"\n=== TARGET COL: {target_col} ===")



    test_size = 4  # H
    window_size = 40  # n_lags
    n_splits = 10
    sample_sets = 3  # number of test sets (like your TFT function)

    RESULTS_DIR = Path("results")
    PLOTS_DIR = Path("plots")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # match your TFT-like header (you can tweak if needed)
    CSV_COLS = [
        "trial_index",
        "test_size",
        "window_size",
        "hidden_size",
        "lstm_layers",
        "num_attention_heads",
        "dropout",
        "batch_size",
        "lr",
        "epochs",
        "grad_clip",
        "mae",
        "mape",
        "mse",
        "rmse",
        "r2",
        "sd",
        "runtime_s",
        "started_at",
        "epochs_ran",
    ]

    # ============================================================
    # Load data
    # ============================================================
    df = pd.read_csv("weekly_chile_data.csv")
    df["FECHA"] = pd.to_datetime(df["FECHA"])
    df = df.sort_values("FECHA").reset_index(drop=True)


    # ============================================================
    # Feature builder (lags of target + lags of exog)
    # ============================================================
    def build_lagged_df(df_in, target_col, n_lags):
        df_in = df_in.sort_values("FECHA").reset_index(drop=True)

        lagged_parts = []
        for lag in range(1, n_lags + 1):
            new_cols = {f"{target_col}_lag{lag}": df_in[target_col].shift(lag)}
            for col in df_in.columns:
                if col not in ["FECHA", target_col] and "_lag" not in col:
                    new_cols[f"{col}_lag{lag}"] = df_in[col].shift(lag)
            lagged_parts.append(pd.DataFrame(new_cols))

        df_feat = pd.concat([df_in, pd.concat(lagged_parts, axis=1)], axis=1)
        df_feat = df_feat.dropna().reset_index(drop=True)
        feature_cols = [c for c in df_feat.columns if "_lag" in c]
        return df_feat, feature_cols


    # ============================================================
    # Lasso model factory
    # ============================================================
    def make_model(train_len, n_splits):
        tscv = TimeSeriesSplit(n_splits=min(n_splits, max(2, train_len // 20)))
        return Pipeline([
            ("scaler", StandardScaler()),
            ("lasso", LassoCV(cv=tscv, max_iter=20000))
        ])


    # ============================================================
    # 1) DIRECT: trains H separate models on (origin -> y_{t+h})
    # ============================================================
    def predict_lasso_direct(tmp_df, target_col, test_size, n_lags, n_splits):
        df_feat, feature_cols = build_lagged_df(tmp_df, target_col, n_lags)

        train_df = df_feat.iloc[:-test_size].reset_index(drop=True)
        test_df = df_feat.iloc[-test_size:].reset_index(drop=True)

        if len(train_df) <= test_size:
            raise ValueError("Not enough training rows after lagging for DIRECT.")

        X_train_all = train_df[feature_cols].values
        y_train_all = train_df[target_col].values

        max_origin_train = len(train_df) - test_size
        X_origin = X_train_all[:max_origin_train]
        Y_future = np.column_stack([y_train_all[h:max_origin_train + h] for h in range(1, test_size + 1)])

        X_cutoff_origin = train_df[feature_cols].iloc[-1].values.reshape(1, -1)

        preds = np.zeros(test_size)
        for h in range(test_size):
            m = make_model(train_len=len(X_origin), n_splits=n_splits)
            m.fit(X_origin, Y_future[:, h])
            preds[h] = m.predict(X_cutoff_origin)[0]

        y_true = test_df[target_col].values
        return y_true, preds


    # ============================================================
    # 2) RECURSIVE: one 1-step model, roll FE lags, freeze exog lags
    # ============================================================
    def predict_lasso_recursive(tmp_df, target_col, test_size, n_lags, n_splits):
        df_feat, feature_cols = build_lagged_df(tmp_df, target_col, n_lags)

        train_df = df_feat.iloc[:-test_size].reset_index(drop=True)
        test_df = df_feat.iloc[-test_size:].reset_index(drop=True)

        if len(train_df) <= test_size:
            raise ValueError("Not enough training rows after lagging for RECURSIVE.")

        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values

        model = make_model(train_len=len(train_df), n_splits=n_splits)
        model.fit(X_train, y_train)

        x_frozen = train_df[feature_cols].iloc[-1].copy()

        initial_lags = [x_frozen[f"{target_col}_lag{lag}"] for lag in range(1, n_lags + 1)]
        fe_lags = deque(initial_lags, maxlen=n_lags)

        preds = []
        for step in range(test_size):
            x_row = x_frozen.copy()
            for lag in range(1, n_lags + 1):
                col = f"{target_col}_lag{lag}"
                if col in x_row.index:
                    x_row[col] = fe_lags[lag - 1]
            yhat = model.predict(x_row.values.reshape(1, -1))[0]
            preds.append(yhat)
            fe_lags.appendleft(yhat)

        y_true = test_df[target_col].values
        return y_true, np.array(preds)


    # ============================================================
    # 3) NAIVE: last observed before test window
    # ============================================================
    def predict_naive_last(tmp_df, target_col, test_size, n_lags):
        df_feat, _ = build_lagged_df(tmp_df, target_col, n_lags)

        train_df = df_feat.iloc[:-test_size].reset_index(drop=True)
        test_df = df_feat.iloc[-test_size:].reset_index(drop=True)

        last_val = train_df[target_col].iloc[-1]
        preds = np.repeat(last_val, test_size)
        y_true = test_df[target_col].values
        print("Naive last predictions:", preds)
        return y_true, preds


    # ============================================================
    # Wrapper that matches your TFT "delete test_size*(i+1)" scheme
    # ============================================================
    def eval_with_for(df, method_fn, target_col, test_size, n_lags, n_splits=None, seed=1048596, sample_sets=sample_sets):
        mae_values, mape_values, mse_values, rmse_values, r2_values = [], [], [], [], []

        df_sorted = df.copy().sort_values("FECHA").reset_index(drop=True)

        for i in range(sample_sets):
            tmp = df_sorted.copy()
            #set the seed:
            np.random.seed(seed + i)
            random_number = i
            # random_number = np.random.randint(0, 20)
            deleted_sample = test_size * (i + random_number)  # EXACTLY like your TFT function
            if deleted_sample > 0:
                tmp = tmp.iloc[:-deleted_sample]

            if n_splits is None:
                y_true, y_pred = method_fn(tmp, target_col, test_size, n_lags)
            else:
                y_true, y_pred = method_fn(tmp, target_col, test_size, n_lags, n_splits)

            mae, mape, mse, rmse, r2 = error_metrics(y_true, y_pred)

            mae_values.append(mae)
            mape_values.append(mape)
            mse_values.append(mse)
            rmse_values.append(rmse)
            r2_values.append(r2)

        print("MAPEs per test set:", mape_values)

        # epochs dummy for non-neural models
        mean_epochs = np.nan
        return (
            float(np.mean(mae_values)),
            float(np.mean(mape_values)),
            float(np.mean(mse_values)),
            float(np.mean(rmse_values)),
            float(np.mean(r2_values)),
            mean_epochs,
            float(np.std(mape_values, ddof=0)),
        )


    # ============================================================
    # Save ONE-ROW csv (same name pattern, one per method)
    # ============================================================
    def write_one_row_csv(method_name, metrics, started_at, runtime_s):
        mae, mape, mse, rmse, r2, mean_epochs, sd = metrics

        row = {c: np.nan for c in CSV_COLS}
        row.update({
            "trial_index": 0,
            "test_size": int(test_size),
            "window_size": int(window_size),
            "mae": mae,
            "mape": mape,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "sd": sd,
            "runtime_s": float(runtime_s),
            "started_at": started_at.isoformat(timespec="seconds"),
            "epochs_ran": mean_epochs,
        })

        csv_path = RESULTS_DIR / f"test{method_name}_trials_{country}_{target_col}_{timestamp}_{test_size}.csv"
        pd.DataFrame([row], columns=CSV_COLS).to_csv(csv_path, index=False)
        print("Saved:", csv_path)


    # ============================================================
    # Run all three methods
    # ============================================================
    # Lasso direct
    start = datetime.now()
    metrics_direct = eval_with_for(
        df=df,
        method_fn=predict_lasso_direct,
        target_col=target_col,
        test_size=test_size,
        n_lags=window_size,
        n_splits=n_splits,
        sample_sets=sample_sets
    )
    runtime = (datetime.now() - start).total_seconds()
    print("[LASSO_DIRECT]", metrics_direct)
    write_one_row_csv("lasso_direct", metrics_direct, start, runtime)

    # # Lasso recursive
    # start = datetime.now()
    # metrics_rec = eval_with_for(
    #     df=df,
    #     method_fn=predict_lasso_recursive,
    #     target_col=target_col,
    #     test_size=test_size,
    #     n_lags=window_size,
    #     n_splits=n_splits,
    #     sample_sets=sample_sets
    # )
    # runtime = (datetime.now() - start).total_seconds()
    # print("[LASSO_RECURSIVE]", metrics_rec)
    # write_one_row_csv("lasso_recursive", metrics_rec, start, runtime)

    # Naive last
    start = datetime.now()
    metrics_naive = eval_with_for(
        df=df,
        method_fn=predict_naive_last,
        target_col=target_col,
        test_size=test_size,
        n_lags=window_size,
        n_splits=None,
        sample_sets=sample_sets
    )
    runtime = (datetime.now() - start).total_seconds()
    print("[NAIVE_LAST]", metrics_naive)
    write_one_row_csv("naive_last", metrics_naive, start, runtime)

    print("\nDone.")
