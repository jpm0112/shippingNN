import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from functions import error_metrics

df = pd.read_csv("weekly_uruguay_data.csv")

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
y = df["FE"].values
y_true = y[1:]  # from second point onward
y_naive = y[:-1]  # previous value as prediction

mae, mape, mse, rmse, r2 = error_metrics(y_true, y_naive)

# autocorrelation plot

from statsmodels.graphics.tsaplots import plot_acf

plot_acf(df["FE"].dropna(), lags=30)
plt.savefig("z_autocorrelation.png", dpi=200)

df_no_fe = df.drop(columns=["FE"])


df_num = df_no_fe.select_dtypes(include=["number"])

corr = df_num.corr()
print(corr)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.imshow(corr, interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.tight_layout()
plt.show()