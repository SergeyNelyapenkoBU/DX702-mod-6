import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("homework_3.1.csv")

# Event cutoff
cutoff = 50
df["D"] = (df["time"] >= cutoff).astype(int)
df["time_x_D"] = df["time"] * df["D"]

results = {}

for col in ["value1", "value2", "value3"]:
    y = df[col].values.reshape(-1, 1)

    # --- Model 1: discontinuity in level only ---
    X1 = df[["time", "D"]].values
    reg1 = LinearRegression().fit(X1, y)
    jump = reg1.coef_[0, 1]  # coefficient on D

    # --- Model 2: discontinuity in level + slope ---
    X2 = df[["time", "D", "time_x_D"]].values
    reg2 = LinearRegression().fit(X2, y)
    slope_change = reg2.coef_[0, 2]  # coefficient on time*D

    results[col] = {
        "jump_at_50": jump,
        "slope_change": slope_change,
        "R2_level": reg1.score(X1, y),
        "R2_slope": reg2.score(X2, y)
    }

# Show results
results_df = pd.DataFrame(results).T
print(results_df)

# Identify dataset with strongest jump
strongest_jump = results_df["jump_at_50"].abs().idxmax()
print("\nDataset with strongest discontinuity in LEVEL at time=50:", strongest_jump)

# Identify dataset with strongest slope change
strongest_slope = results_df["slope_change"].abs().idxmax()
print("Dataset with strongest discontinuity in SLOPE at time=50:", strongest_slope)
