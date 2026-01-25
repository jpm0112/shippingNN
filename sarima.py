import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from functions import error_metrics, run_sarima_with_for
from datetime import datetime
from functions import run_sarima

# Load and prepare data
df = pd.read_csv("weekly_chile_data.csv")
df["FECHA"] = pd.to_datetime(df["FECHA"])
df = df.sort_values("FECHA").reset_index(drop=True)

# Parameters
target_col = 'SE'
# target_col = "TOTAL_TEUS"
test_size = 4
target_col = 'SAE'
p, d, q, P, D, Q, m = 0, 0, 0, 1, 1, 1, 52 #SAE
target_col = 'SAW'
p, d, q, P, D, Q, m = 0, 0, 4, 0, 1, 0, 52 # SAW
target_col = 'SE'
p, d, q, P, D, Q, m = 0, 0, 0, 1, 1, 1, 52 #SE
target_col = 'NE'
p, d, q, P, D, Q, m = 1, 1, 3, 0, 1, 1, 52 #NE
target_col = 'NAW'
p, d, q, P, D, Q, m = 1, 2, 3, 2, 0, 0, 52 #NAW
target_col = 'NAE'
p, d, q, P, D, Q, m = 3, 0, 2, 2, 0, 0, 52 #NAE
target_col = 'FE'
p, d, q, P, D, Q, m = 1, 2, 2, 0, 0, 1, 52 #FE


mae, mape, mse, rmse, r2, epochs_ran, sd = run_sarima_with_for(df,
                                                               target_col,
                                                               test_size,
                                                               p, d, q, P, D, Q, m,
                                                               seed=1048596,
                                                               n_runs=3)
print(target_col)
print(mape)
print(mae)
# ============================================================



# FOR THE GRID SEARCH   :


feature_cols = [
    col for col in df.columns
    if col not in ['FECHA', target_col] and target_col not in col
]
# Split into train and test
train_df = df[:-test_size]
test_df = df[-test_size:]

# Normalize exogenous features
exog_scaler = StandardScaler()
exog_train = exog_scaler.fit_transform(train_df[feature_cols])
exog_test = exog_scaler.transform(test_df[feature_cols])

# Normalize target variable
y_train = train_df[target_col]
y_test = test_df[target_col]

target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).ravel()


results_df = pd.DataFrame(columns=[
    "seed","test_size","a", "b", "c", "d", "e","f","g","Target",
    "MAE", "MSE", "RMSE","MAPE", "r2"
])

seed = 1048596

ass = [1, 2]      # AR terms
bs = [1]          # Differencing (1 is usually enough)
cs = [0, 1]       # MA terms (avoid 2 to prevent lag conflict)

ds = [0, 1]       # Seasonal AR terms
es = [1]          # Seasonal differencing (often 1)
fs = [0, 1]       # Seasonal MA (0 or 1 only)
gs = [7,30,60]          # Weekly seasonality


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for a in ass:
    for b in bs:
        for c in cs:
            for d in ds:
                for e in es:
                    for f in fs:
                        for g in gs:

                            if (
                                (c > 0 and f > 0 and (f * g == 2 or c == 2)) or  # MA conflict
                                (a > 0 and d > 0 and (d * g == 2 or a == 2))):
                                continue

                            model = SARIMAX(
                                endog=y_train_scaled,
                                order=(a, b, c),
                                seasonal_order=(d, e, f, g),
                                enforce_stationarity=False,
                                enforce_invertibility=False
                            )
                            results = model.fit(disp=False)
                            forecast_scaled = results.predict(start=len(train_df), end=len(df)-1, exog=exog_test)
                            y_pred = target_scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).ravel()
                            y_true = target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()
                            mae, mape, mse, rmse, r2 = error_metrics(y_true, y_pred)

                            results_df.loc[len(results_df)] = [
                                seed, test_size, a, b, c, d, e, f, g, target_col, mae, mse, rmse, mape,  r2
                            ]
                            results_df.to_csv(f"results/model_results_sarima_{timestamp}_{target_col}.csv", index=False)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(y_true, marker="o", label="Real")
plt.plot(y_pred, marker="o", label="Prediction")
plt.title("Prediction (original scale)")
plt.xlabel("Weeks")
plt.ylabel(target_col)
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("plots/z_sarima_prediction.png", dpi=200)
