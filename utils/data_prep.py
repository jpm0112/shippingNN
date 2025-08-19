import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
root_dir = os.path.dirname(script_dir)

target_list = [
    "XSICFEUW Index  (R4)",
    "XSICFENE Index  (R1)",
    "XSICFESE Index  (L4)",
    "XSICNEFE Index  (R2)",
    "XSICNESE Index  (L2)",
    "XSICUENE Index  (R1)",
    "XSICUWFE Index  (L1)",
    "XSICNEUE Index  (L3)",
    "BDIY Index  (R3)",
    "Close_pct_change_XSICFEUW Index  (R4)",
    "Close_pct_change_XSICFENE Index  (R1)",
    "Close_pct_change_XSICFESE Index  (L4)",
    "Close_pct_change_XSICNEFE Index  (R2)",
    "Close_pct_change_XSICNESE Index  (L2)",
    "Close_pct_change_XSICUENE Index  (R1)",
    "Close_pct_change_XSICUWFE Index  (L1)",
    "Close_pct_change_XSICNEUE Index  (L3)",
    "Close_pct_change_BDIY Index  (R3)",
]


def prepare_data_w_lag(df, target_col_num, n_lags, n_splits):
    """
    TARGET LIST:
    0:"XSICFEUW Index  (R4)",
    1:"XSICFENE Index  (R1)",
    2:"XSICFESE Index  (L4)",
    3:"XSICNEFE Index  (R2)",
    4:"XSICNESE Index  (L2)",
    5:"XSICUENE Index  (R1)",
    6:"XSICUWFE Index  (L1)",
    7:"XSICNEUE Index  (L3)",
    8:"BDIY Index  (R3)",
    9:"Close_pct_change_XSICFEUW Index  (R4)",
    10:"Close_pct_change_XSICFENE Index  (R1)",
    11:"Close_pct_change_XSICFESE Index  (L4)",
    12:"Close_pct_change_XSICNEFE Index  (R2)",
    13:"Close_pct_change_XSICNESE Index  (L2)",
    14:"Close_pct_change_XSICUENE Index  (R1)",
    15:"Close_pct_change_XSICUWFE Index  (L1)",
    16:"Close_pct_change_XSICNEUE Index  (L3)",
    17:"Close_pct_change_BDIY Index  (R3)",

    Leave df None to use the default data
    """
    if df is None:
        df = pd.read_csv(os.path.join(root_dir, "proc", "test_final_kz.csv"))
    df["FECHA"] = pd.to_datetime(df["FECHA"] + "-5", format="%Y-%W-%w")
    df = df.sort_values("FECHA").reset_index(drop=True)

    target_col = target_list[target_col_num]
    cols_to_lag = [c for c in df.columns if c != "FECHA"]
    if n_lags > 0:
        lagged = {
            f"{c}_lag{lag}": df[c].shift(lag)
            for c in cols_to_lag
            for lag in range(1, n_lags + 1)
        }
        lagged_df = pd.DataFrame(lagged, index=df.index)
        df_full = pd.concat([df, lagged_df], axis=1).dropna().reset_index(drop=True)
    else:
        df_full = df.copy()
    feature_cols = [
        c
        for c in df_full.columns
        if c.endswith(tuple(f"_lag{lag}" for lag in range(1, n_lags + 1)))
    ]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    for i, (train_index, test_index) in enumerate(tscv.split(df_full)):
        train_df = df_full.iloc[train_index]
        test_df = df_full.iloc[test_index]

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train = scaler_X.fit_transform(train_df[feature_cols])
        y_train = scaler_y.fit_transform(train_df[[target_col]]).ravel()

        X_test = scaler_X.transform(test_df[feature_cols])
        y_test = scaler_y.transform(test_df[[target_col]]).ravel()

        yield X_train, y_train, X_test, y_test
