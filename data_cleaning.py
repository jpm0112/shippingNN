import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from functions import error_metrics

df = pd.read_csv("weekly_uruguay_data.csv")
df_without_targets = df.drop(columns=["FE", "NA","SA","NE","SE"])



cols_to_drop = df_without_targets.columns[df_without_targets.nunique() == 1].tolist()
# cols_to_drop += ["INCOTERMS","PAIS_ORIGEN"]
cols_to_drop_temp = ["FECHA"]
df_without_targets = df_without_targets.drop(columns=cols_to_drop)
df_without_targets = df_without_targets.drop(columns=cols_to_drop_temp)

corr = df_without_targets.corr().abs()
upper = np.triu(corr, k=1)
to_drop = [column for column in corr.columns if any(upper[:, corr.columns.get_loc(column)] > 0.95)]

df_without_targets = df_without_targets.drop(columns=to_drop)
cols_to_drop += to_drop

df = df.drop(columns=cols_to_drop)

threshold = 0.95
corr = df_without_targets.corr().abs()

pairs = []
for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        if corr.iloc[i, j] > threshold:
            pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

pairs


def replace_zeros_with_neighbors_mean(df, col):
    s = df[col].copy()
    for i in range(1, len(s) - 1):
        if s.iat[i] <= 0 :
            s.iat[i] = (s.iat[i - 1] + s.iat[i + 1]) / 2
    df[col] = s
    return df


df = replace_zeros_with_neighbors_mean(df, 'MIN_FOB_USD')
df = replace_zeros_with_neighbors_mean(df, 'MEAN_flete_per_TEU_Francia')
df = replace_zeros_with_neighbors_mean(df, 'MEAN_flete_per_TEU_Reino Unido')
df = replace_zeros_with_neighbors_mean(df, 'MEAN_flete_per_TEU_Turquia')
df = replace_zeros_with_neighbors_mean(df, 'MIN_CIF_USD')




df.to_csv("weekly_uruguay_data.csv", index=False)
# Convert date if needed (optional)
# df["FECHA"] = pd.to_datetime(df["FECHA"] + "-1", format="%Y-%W-%w")





#_________________________________________

#CHILE


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from functions import error_metrics

df = pd.read_csv("chile_data.csv")


df = df.drop(columns=["MEAN_FLETE_per_TEU_AF", "MEAN_FLETE_per_TEU_ME",'MEAN_FLETE_per_TEU_OC'])


df["FECHA"] = df["WEEK"]
df = df.drop(columns=['WEEK'])

cols_to_rename = {
    "MEAN_FLETE_per_TEU_FE": "FE",
    "MEAN_FLETE_per_TEU_NAE": "NAE",
    "MEAN_FLETE_per_TEU_NAW": "NAW",
    "MEAN_FLETE_per_TEU_NE": "NE",
    "MEAN_FLETE_per_TEU_SE": "SE",
    "MEAN_FLETE_per_TEU_SAW": "SAW",
    "MEAN_FLETE_per_TEU_SAE": "SAE",
}

df = df.rename(columns=cols_to_rename)

df_without_targets = df.drop(columns=["FE", "NAE","NAW","NE","SE","SAW","SAE"])

cols_to_drop = df_without_targets.columns[df_without_targets.nunique() == 1].tolist()
# cols_to_drop += ["INCOTERMS","PAIS_ORIGEN"]
cols_to_drop_temp = ["FECHA"]
df_without_targets = df_without_targets.drop(columns=cols_to_drop)
df_without_targets = df_without_targets.drop(columns=cols_to_drop_temp)


# to print the pairs of highly correlated features
threshold = 0.90
corr = df_without_targets.corr().abs()
pairs = []
for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        if corr.iloc[i, j] > threshold:
            pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

pairs

corr = df_without_targets.corr().abs()
upper = np.triu(corr, k=1)
to_drop = [column for column in corr.columns if any(upper[:, corr.columns.get_loc(column)] > 0.95)]

df_without_targets = df_without_targets.drop(columns=to_drop)
cols_to_drop += to_drop

df = df.drop(columns=cols_to_drop)


def replace_zeros_with_neighbors_mean(df, col):
    s = df[col].copy()
    for i in range(1, len(s) - 1):
        if s.iat[i] <= 0 and s.iat[i - 1] > 0 and s.iat[i + 1] > 0:
            s.iat[i] = (s.iat[i - 1] + s.iat[i + 1]) / 2
    df[col] = s
    return df


for col in df.columns:
    if col.startswith("MEAN_FLETE"):
        df = replace_zeros_with_neighbors_mean(df, col)

fixed_cols = df.columns[df.nunique() == 1].tolist()
print(fixed_cols)
df = df.drop(columns=fixed_cols)

df = df[df["FECHA"] >= "2017-01-01"].copy()
# df = df[df["FECHA"] < "2024-11-23"].copy()
df = df[df["FECHA"] <= "2025-06-30"].copy()

df["series"] = "chile"  # harmless for baseline
df["time_idx"] = df.groupby("series").cumcount()
df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")
df["dow"] = df["FECHA"].dt.weekday.astype(int)
df["month"] = df["FECHA"].dt.month.astype(int)

df = df.drop(columns=["series","dow"])



df.columns = df.columns.str.replace(".", "_", regex=False)

df = df.sort_values("FECHA")

# full_weeks = pd.date_range(
#     start=df["FECHA"].min(),
#     end=df["FECHA"].max(),
#     freq="W-MON"
# )
#
# df = (
#     df.set_index("FECHA")
#     .reindex(full_weeks)
# )
#
# df.index.name = "FECHA"
# df = df.reset_index()
# df = df.fillna(method='ffill').fillna(method='bfill')



df.to_csv("weekly_chile_data.csv", index=False)




plt.figure(figsize=(12, 4))
plt.plot(df["FECHA"], df["NUMERO DE ACEPTACION"])
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # ~10 ticks
plt.gcf().autofmt_xdate()
plt.title("Número de Aceptación por Semana")
plt.xlabel("Week")
plt.ylabel("Valor")
plt.tight_layout()
plt.show()





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