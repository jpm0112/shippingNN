from datetime import datetime
from functions import run_tft_tftorch, run_tft
import torch
import pandas as pd

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
df = pd.read_csv("weekly_chile_data.csv")
tmp = df.copy().sort_values("FECHA")


target_col = "FE"
window_size = 24
test_size = 4
d_model = 512
n_head = 4
num_layers = 9  # lstm layers
epoch_number = 10
lr = 0.00166941602502568
batch_size = 16
seed = 1048596
grad_clip = 0.1


train_cut = tmp["time_idx"].max() - test_size

y_true, y_pred, model, train_loader, val_loader, train_ds = run_tft_tftorch(tmp, target_col, window_size, test_size, grad_clip, d_model, n_head, num_layers, epoch_number, lr,batch_size,seed, train_cut)






















