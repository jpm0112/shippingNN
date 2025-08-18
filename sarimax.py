import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from functions import *
from datetime import datetime

# Load and prepare data
df = pd.read_csv("test_daily.csv")
df['FECHA'] = pd.to_datetime(df['FECHA'])
df = df.sort_values('FECHA')

# Parameters
target_col = 'MEAN_FLETE_POR_BULTO'
# target_col = "TOTAL_TEUS"
test_size = 30
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
    "MAE", "MSE", "RMSE", "r2"
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

                            mae, mse, rmse, r2 = error_metrics(y_test_scaled, forecast_scaled)

                            results_df.loc[len(results_df)] = [
                                seed, test_size, a, b, c, d, e, f, g, target_col, mae, mse, rmse, r2
                            ]
                            results_df.to_csv(f"model_results_sarima_{timestamp}.csv", index=False)

                        


# Plot results (normalized)
# plt.figure(figsize=(12, 6))
# plt.plot(y_test_scaled, label='Real (Normalized)')
# plt.plot(forecast_scaled, label='SARIMA Prediction (Normalized)')
# plt.title('SARIMA Forecast (Normalized)')
# plt.xlabel('Days')
# plt.ylabel('Normalized Target')
# plt.legend()
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.xticks(ticks=range(0, len(y_test_scaled), max(1, len(y_test_scaled)//30)))
# plt.show()
