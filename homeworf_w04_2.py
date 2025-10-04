import pandas as pd
from sklearn.linear_model import LinearRegression

def analyze_rdd(df, x_col, y_col, cutoff=80):
    # Split data before/after cutoff
    before = df[df[x_col] < cutoff]
    after = df[df[x_col] >= cutoff]

    # Fit simple linear regressions
    reg_before = LinearRegression().fit(before[[x_col]], before[y_col])
    reg_after = LinearRegression().fit(after[[x_col]], after[y_col])

    slope_before = reg_before.coef_[0]
    slope_after = reg_after.coef_[0]

    print(f"\nDataset: {x_col}")
    print(f"Slope before cutoff: {slope_before:.4f}")
    print(f"Slope after cutoff:  {slope_after:.4f}")

    if slope_after > slope_before:
        print("→ Q3: Slope is HIGHER after cutoff.")
    else:
        print("→ Q3: Slope is LOWER after cutoff.")

    if slope_before > 0:
        print("→ Q4: Y is INCREASING before cutoff.")
    else:
        print("→ Q4: Y is DECREASING before cutoff.")

# --- Load both datasets ---
a = pd.read_csv("homework_4.2.a.csv")
b = pd.read_csv("homework_4.2.b.csv")

# Run RDD analysis
analyze_rdd(a, "X", "Y")
analyze_rdd(b, "X2", "Y2")
