from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
import numpy as np
import os
from data_prep import prepare_data_w_lag

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
root_dir = os.path.dirname(script_dir)


def evaluate_baseline_model(df=None, target_col_num=0, n_lags=20, n_splits=5):
    mae_list = []
    mape_list = []
    r2_list = []
    for X_train, y_train, X_test, y_test in prepare_data_w_lag(
        df, target_col_num, n_lags, n_splits
    ):
        # Calculate mean of training target
        last_target = y_train[-1]
        baseline_preds = [last_target] * len(y_test)
        # Calculate metrics
        mae = mean_absolute_error(y_test, baseline_preds)
        mape = mean_absolute_percentage_error(y_test, baseline_preds)
        r2 = r2_score(y_test, baseline_preds)
        mae_list.append(mae)
        mape_list.append(mape)
        r2_list.append(r2)
        print(f"MAE: {mae}, MAPE: {mape}, R2: {r2}")
    print(
        f"Average MAE: {sum(mae_list) / len(mae_list)}, "
        f"Average MAPE: {np.prod(mape_list) ** (1 / len(mape_list))}, "
        f"Average R2: {sum(r2_list) / len(r2_list)}"
    )


if __name__ == "__main__":
    # Example usage
    evaluate_baseline_model(target_col_num=7, n_lags=1, n_splits=5)
