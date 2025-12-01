import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from functions import error_metrics

# Load and prepare data
df = pd.read_csv("chile_data.csv")
df["FECHA"] = pd.to_datetime(df["FECHA"] + "-5", format="%Y-%W-%w")
df = df.sort_values("FECHA")

# Parameters
target_col = "FE"
test_size =24
n_lags = 48  # number of past values used as features

# Build lagged features for target and other variables
lagged_cols = []
for lag in range(1, n_lags + 1):
    new_cols = {f"{target_col}_lag{lag}": df[target_col].shift(lag)}
    for col in df.columns:
        if col not in ["FECHA", target_col] and "_lag" not in col:
            new_cols[f"{col}_lag{lag}"] = df[col].shift(lag)
    lagged_cols.append(pd.DataFrame(new_cols))

# Combine all lagged features
lagged_df = pd.concat(lagged_cols, axis=1)
df = pd.concat([df, lagged_df], axis=1)
df = df.dropna()

# Define feature columns (all lagged features)
feature_cols = [col for col in df.columns if "_lag" in col]

# print(feature_cols)


# testing soemthing unrelate to the code

# Train/test split
train_df = df[:-test_size]
test_df = df[-test_size:]

# Normalize features and target
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(train_df[feature_cols])
y_train = scaler_y.fit_transform(train_df[[target_col]]).ravel()

X_test = scaler_X.transform(test_df[feature_cols])
y_test = scaler_y.transform(test_df[[target_col]]).ravel()

# Train Lasso model
tscv = TimeSeriesSplit(n_splits=10)
model = LassoCV(cv=tscv, max_iter=20000).fit(X_train, y_train)
print(model.alpha_)

# model = Lasso(alpha=0.001, max_iter=10000)
model.fit(X_train, y_train)
# ============================
#  RECURSIVE 24-STEP FORECAST
# ============================

# Start from the last available row in the *training* set
last_row = train_df.iloc[-1].copy()

# We'll store predictions here (scaled space)
forecast_scaled = []

# Loop for 24 steps ahead
for step in range(test_size):

    # Build feature vector from current last_row
    X_current = last_row[feature_cols].values.reshape(1, -1)
    X_current_scaled = scaler_X.transform(X_current)

    # Predict scaled
    pred_scaled = model.predict(X_current_scaled)[0]
    forecast_scaled.append(pred_scaled)

    # ---------------------------------------------------------
    # UPDATE lag values for next prediction (core of recursion)
    # ---------------------------------------------------------

    # Insert the new predicted FE value in the lag structure
    # Shift existing lags: lag1 <- new_pred, lag2 <- previous lag1, ..., lag48 <- lag47
    for lag in range(n_lags, 1, -1):
        last_row[f"{target_col}_lag{lag}"] = last_row[f"{target_col}_lag{lag-1}"]

    # For lag1, we use the *predicted* value (scaled, but will be inverse transformed later)
    last_row[f"{target_col}_lag1"] = scaler_y.inverse_transform([[pred_scaled]])[0,0]

    # Also update lags for other variables (if any)
    for col in df.columns:
        if col not in ["FECHA", target_col] and "_lag" not in col:
            for lag in range(n_lags, 1, -1):
                last_row[f"{col}_lag{lag}"] = last_row[f"{col}_lag{lag-1}"]
            # For exogenous vars: we *cannot* know future values → use last known value blindly
            last_row[f"{col}_lag1"] = last_row[f"{col}"]

preds_scaled = model.predict(X_test)

# Evaluation


# mae, mape, mse, rmse, r2 = error_metrics(y_test, preds_scaled)



import matplotlib.pyplot as plt
# Plot results
plt.figure(figsize=(12, 6))
plt.plot(y_test, label="Real", marker="o")
plt.plot(preds_scaled, label="Prediction", marker="o")
plt.title("Last month prediction")
plt.xlabel("Days")
plt.ylabel("Normalized target")
plt.legend()
plt.xticks(ticks=range(0, len(y_test), max(1, len(y_test) // 10)))
plt.grid(True, which="both", linestyle="--", linewidth=0.5)


plt.tight_layout()
# plt.savefig("zlast_month_prediction.png", dpi=200)




# Undo scaling:

forecast_real = scaler_y.inverse_transform(
        np.array(forecast_scaled).reshape(-1, 1)
).ravel()
y_test_real = test_df[target_col].values
mae, mape, mse, rmse, r2 = error_metrics(y_test_real, forecast_real)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(y_test_real, marker="o", label="Real")
plt.plot(forecast_real, marker="o", label="Prediction")
plt.title("Prediction (original scale)")
plt.xlabel("Weeks")
plt.ylabel(target_col)
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("plots/z_lasso_prediction.png", dpi=200)
