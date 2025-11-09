import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("homework_4.1.csv")

# 1 Simple (global) Wald estimator
mean_y_z1 = df.loc[df["Z"] == 1, "Y"].mean()
mean_y_z0 = df.loc[df["Z"] == 0, "Y"].mean()
mean_x_z1 = df.loc[df["Z"] == 1, "X"].mean()
mean_x_z0 = df.loc[df["Z"] == 0, "X"].mean()

global_effect = (mean_y_z1 - mean_y_z0) / (mean_x_z1 - mean_x_z0)

print(f"Global IV (Wald) Effect: {global_effect:.4f}")

# 2 Local (conditional on W) Wald estimator
# Define bins of W (narrow ranges)
num_bins = 10   # can adjust (e.g., 5–20) for granularity
df["W_bin"] = pd.qcut(df["W"], q=num_bins, duplicates="drop")

effects = []

for w_bin, group in df.groupby("W_bin"):
    # Check that both Z=0 and Z=1 exist in this bin
    if group["Z"].nunique() < 2:
        continue

    mean_y_z1 = group.loc[group["Z"] == 1, "Y"].mean()
    mean_y_z0 = group.loc[group["Z"] == 0, "Y"].mean()
    mean_x_z1 = group.loc[group["Z"] == 1, "X"].mean()
    mean_x_z0 = group.loc[group["Z"] == 0, "X"].mean()

    denom = mean_x_z1 - mean_x_z0
    if denom == 0 or np.isnan(denom):
        continue

    local_effect = (mean_y_z1 - mean_y_z0) / denom
    effects.append(local_effect)

# Average the local effects
if effects:
    avg_local_effect = np.mean(effects)
    print(f"Average Local IV Effect (averaged over W bins): {avg_local_effect:.4f}")
else:
    print("No valid bins found for local effect calculation.")
