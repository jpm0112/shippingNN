import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from functions import error_metrics

df = pd.read_csv("chile_data.csv")

# Convert date if needed (optional)
# df["FECHA"] = pd.to_datetime(df["FECHA"] + "-1", format="%Y-%W-%w")

target_col = "FE"

# Compute correlations with the target
corr = df.corr(numeric_only=True)[target_col].sort_values(ascending=False)
print(corr)

top = corr.head(15)
print(top)

top.plot(kind="bar", figsize=(10, 4), title=f"Correlation with {target_col}")
plt.tight_layout()
plt.savefig("zcorrelations.png", dpi=200)


# naive forecasting
y = df["SAE"].values
y_true = y[1:]  # from second point onward
y_naive = y[:-1]  # previous value as prediction

mae, mape, mse, rmse, r2 = error_metrics(y_true, y_naive)

# autocorrelation plot

from statsmodels.graphics.tsaplots import plot_acf

plot_acf(df["SAE"].dropna(), lags=30)
plt.savefig("z_autocorrelation.png", dpi=200)
