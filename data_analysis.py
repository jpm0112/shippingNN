import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from functions import error_metrics

df = pd.read_csv("weekly_uruguay_data.csv")
df_without_targets = df.drop(columns=["FE", "NA","SA","NE","SE"])



cols_to_drop = df_without_targets.columns[df_without_targets.nunique() == 1].tolist()
cols_to_drop += ["INCOTERMS","PAIS_ORIGEN"]
cols_to_drop_temp = ["WEEK", "FECHA"]
df_without_targets = df_without_targets.drop(columns=cols_to_drop)
df_without_targets = df_without_targets.drop(columns=cols_to_drop_temp)

corr = df_without_targets.corr().abs()
upper = np.triu(corr, k=1)
to_drop = [column for column in corr.columns if any(upper[:, corr.columns.get_loc(column)] > 0.95)]

df_without_targets = df_without_targets.drop(columns=to_drop)
cols_to_drop.extend(to_drop)

df = df.drop(columns=to_drop)

threshold = 0.95
corr = df_without_targets.corr().abs()

pairs = []
for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        if corr.iloc[i, j] > threshold:
            pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

pairs

df.to_csv("weekly_uruguay_data.csv", index=False)
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