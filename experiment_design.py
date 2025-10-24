import itertools
import numpy as np
import pandas as pd
# ---- user-tunable settings ----
factor_names = [f"F{i}" for i in range(1, 9)]  # you can replace with your hyperparameter names
random_seed = 42
n_center_points = 8  # e.g., 1 per dataset; set to 0 to skip
dataset_labels = [f"ds{i}" for i in range(1, 9)]  # 8 datasets
# --------------------------------

rng = np.random.default_rng(random_seed)

# Build 2^(7) full factorial for basics A..G
levels = [-1, 1]
basic_cols = list(itertools.product(levels, repeat=7))
FF = np.array(basic_cols, dtype=int)  # shape (128, 7)
A, B, C, D, E, F, G = FF.T

# Define H using a 5-letter generator to ensure Resolution >= V.
# Choose H = A*B*C*D*E (length 5 generator). With only one generator, the shortest word is 6 (ABCDEH).
H = A * B * C * D * E

# Stack columns in order to create 8-factor design
design_coded = np.column_stack([A, B, C, D, E, F, G, H])

# Build DataFrame
df = pd.DataFrame(design_coded, columns=factor_names)
#test

# Assign datasets in a balanced, randomized way
order = rng.permutation(len(df))
df = df.iloc[order].reset_index(drop=True)
# Round-robin assign datasets to keep near-balance
df["dataset"] = [dataset_labels[i % len(dataset_labels)] for i in range(len(df))]

# Add center points (all zeros) and spread them across datasets
if n_center_points > 0:
    center = pd.DataFrame(np.zeros((n_center_points, len(factor_names)), dtype=int), columns=factor_names)
    # assign center points to datasets as evenly as possible
    center["dataset"] = [dataset_labels[i % len(dataset_labels)] for i in range(n_center_points)]
    # mark center points
    center["is_center_point"] = True
    df["is_center_point"] = False
    # Append and reshuffle slightly (keep coded runs first if you wish; here we interleave)
    df = pd.concat([df, center], ignore_index=True)
else:
    df["is_center_point"] = False

# Add a run_id
df.insert(0, "run_id", np.arange(1, len(df) + 1))

# Save to CSV for you to download
csv_path = "fft_8factors_128run_resVI_with_centers.csv"
df.to_csv(csv_path, index=False)
