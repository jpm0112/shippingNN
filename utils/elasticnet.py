from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
import os
import numpy as np
from data_prep import prepare_data_w_lag
from baseline import evaluate_baseline_model
import xgboost as xgb

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
root_dir = os.path.dirname(script_dir)


def run_xgb_model(df=None, target_col_num=0, n_lags=20, n_splits=5):
    mae_list = []
    mape_list = []
    r2_list = []
    for X_train, y_train, X_test, y_test in prepare_data_w_lag(
        df, target_col_num, n_lags, n_splits
    ):
        model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        mape = mean_absolute_percentage_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        mae_list.append(mae)
        mape_list.append(mape)
        r2_list.append(r2)
        print(f"XGB - MAE: {mae:.4f}, MAPE: {mape:.4f}, R2: {r2:.4f}")
    print(
        f"Average MAE: {sum(mae_list) / len(mae_list):.4f}, "
        f"Average MAPE: {np.prod(mape_list) ** (1 / len(mape_list))}, "
        f"Average R2: {sum(r2_list) / len(r2_list):.4f}"
    )


if __name__ == "__main__":
    # Example usage
    for n in range(11, 18):
        print(f"Running index {n} for target")
        print("Running ElasticNet model...")
        run_xgb_model(target_col_num=n, n_lags=8, n_splits=5)
        print("Running baseline model for comparison...")
        evaluate_baseline_model(target_col_num=n, n_lags=8, n_splits=5)
