import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from collections import deque

from functions import error_metrics

# ============================================================
# Settings
# ============================================================
target_col = "SE"
H = 12               # forecast horizon (weeks)
n_lags = 52          # number of lag weeks used as features
n_splits = 10        # CV splits for LassoCV

# ============================================================
# Load data
# ============================================================
df = pd.read_csv("weekly_chile_data.csv")
df["FECHA"] = pd.to_datetime(df["FECHA"])
df = df.sort_values("FECHA").reset_index(drop=True)

# ============================================================
# Build lagged features (target + exogenous history)
# ============================================================
lagged_cols = []
for lag in range(1, n_lags + 1):
    new_cols = {f"{target_col}_lag{lag}": df[target_col].shift(lag)}
    for col in df.columns:
        if col not in ["FECHA", target_col] and "_lag" not in col:
            new_cols[f"{col}_lag{lag}"] = df[col].shift(lag)
    lagged_cols.append(pd.DataFrame(new_cols))

df_feat = pd.concat([df, pd.concat(lagged_cols, axis=1)], axis=1).dropna().reset_index(drop=True)
feature_cols = [c for c in df_feat.columns if "_lag" in c]

# ============================================================
# Common: define cutoff (last H weeks are the evaluation window)
# ============================================================
train_df = df_feat.iloc[:-H].reset_index(drop=True)
test_df  = df_feat.iloc[-H:].reset_index(drop=True)

tscv = TimeSeriesSplit(n_splits=min(n_splits, max(2, len(train_df) // 20)))

def make_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", LassoCV(cv=tscv, max_iter=20000))
    ])

# ============================================================
# 1) DIRECT multi-horizon (NO training on the last H weeks)
#    Trains H separate models: y(t+h) from features at time t.
# ============================================================
X_train_all = train_df[feature_cols].values
y_train_all = train_df[target_col].values

# only use origins whose labels stay inside TRAIN
max_origin_train = len(train_df) - H
if max_origin_train <= 0:
    raise ValueError("Not enough training data: need len(train_df) > H.")

X_origin = X_train_all[:max_origin_train]
Y_future = np.column_stack([y_train_all[h:max_origin_train + h] for h in range(1, H + 1)])

# forecast from cutoff origin = last feature row of train_df (time T)
X_cutoff_origin = train_df[feature_cols].iloc[-1].values.reshape(1, -1)
Y_true = test_df[target_col].values

preds_direct = np.zeros(H)
alphas_direct = np.zeros(H)

for h in range(H):
    m = make_model()
    m.fit(X_origin, Y_future[:, h])
    preds_direct[h] = m.predict(X_cutoff_origin)[0]
    alphas_direct[h] = m.named_steps["lasso"].alpha_

mae, mape, mse, rmse, r2 = error_metrics(Y_true, preds_direct)
print("DIRECT alphas:", alphas_direct)
print(f"[DIRECT] MAE={mae:.4f} | MAPE={mape:.4f} | MSE={mse:.4f} | RMSE={rmse:.4f} | R2={r2:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(Y_true, marker="o", label="Real")
plt.plot(preds_direct, marker="o", label="Direct prediction")
plt.title(f"Direct multi-horizon Lasso (H={H}) from cutoff (no test leakage)")
plt.xlabel("Horizon (weeks ahead)")
plt.ylabel(target_col)
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig(f"plots/lasso_direct_multihorizon_{target_col}_H{H}.png", dpi=200)

# ============================================================
# 2) RECURSIVE multi-horizon with FROZEN exogenous history
#    One 1-step model. During horizon:
#      - FE lags updated with predictions
#      - exogenous lagged features frozen at cutoff (no future exog)
# ============================================================
X_train = train_df[feature_cols].values
y_train = train_df[target_col].values

model = make_model()
model.fit(X_train, y_train)
print("RECURSIVE best alpha:", model.named_steps["lasso"].alpha_)


# freeze all exog-lag features at cutoff:
x_frozen = train_df[feature_cols].iloc[-1].copy()
# FE lag buffer from last observed (pre-cutoff) FE values
initial_lags = []
for lag in range(1, n_lags + 1):
    initial_lags.append(x_frozen[f"{target_col}_lag{lag}"])
fe_lags = deque(initial_lags, maxlen=n_lags)



preds_rec = []
for step in range(H):
    x_row = x_frozen.copy()

    # overwrite ONLY FE lags with rolling buffer (prevents target leakage)
    for lag in range(1, n_lags + 1):
        col = f"{target_col}_lag{lag}"
        if col in x_row.index:
            x_row[col] = fe_lags[lag - 1]

    yhat = model.predict(x_row.values.reshape(1, -1))[0]
    preds_rec.append(yhat)
    fe_lags.appendleft(yhat)

preds_rec = np.array(preds_rec)

mae, mape, mse, rmse, r2 = error_metrics(Y_true, preds_rec)
print(f"[RECURSIVE] MAE={mae:.4f} | MAPE={mape:.4f} | MSE={mse:.4f} | RMSE={rmse:.4f} | R2={r2:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(Y_true, marker="o", label="Real")
plt.plot(preds_rec, marker="o", label="Recursive prediction")
plt.title(f"Recursive multi-horizon Lasso (H={H}), frozen exog history")
plt.xlabel("Horizon (weeks ahead)")
plt.ylabel(target_col)
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig(f"plots/lasso_recursive_multihorizon_{target_col}_H{H}.png", dpi=200)

