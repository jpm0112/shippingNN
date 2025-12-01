from datetime import datetime
from functions import run_tft_tftorch, run_tft, error_metrics
import torch
import pandas as pd

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
df = pd.read_csv("weekly_chile_data.csv")
tmp = df.copy().sort_values("FECHA")
tmp["series"] = "chile"  # specific for TFT

target_col = "FE"
window_size = 24
test_size = 8
d_model = 512
n_head = 8
num_layers = 3  # lstm layers
epoch_number = 692
lr = 0.00930896825864283
batch_size = 32
seed = 1048596
grad_clip = 0.0265635004162112


train_cut = tmp["time_idx"].max() - test_size

y_true, y_pred, model, train_loader, val_loader, train_ds = run_tft_tftorch(tmp, target_col, window_size, test_size, grad_clip, d_model, n_head, num_layers, epoch_number, lr,batch_size,seed, train_cut)


error_metrics(y_true, y_pred)



















