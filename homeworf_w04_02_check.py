import pandas as pd
from sklearn.linear_model import LinearRegression

def analyze_dataset(df, x_col, y_col, cutoff=80):
    # Split data before and after cutoff
    before = df[df[x_col] < cutoff]
    after = df[df[x_col] >= cutoff]

    # Fit linear regressions (Y on X)
    reg_before = LinearRegression().fit(before[[x_col]], before[y_col])
    reg_after = LinearRegression().fit(after[[x_col]], after[y_col])

    slope_before = reg_before.coef_[0]
    slope_after = reg_after.coef_[0]

    print(f"\nDataset: {x_col}")
    print(f"Slope before cutoff: {slope_before:.4f}")
    print(f"Slope after cutoff:  {slope_after:.4f}")

    # Question 3: Compare slopes before/after
    if slope_after > slope_before:
        q3 = "Higher"
    else:
        q3 = "Lower"

    # Question 4: Direction before cutoff
    if slope_before > 0:
        q4 = "Increasing"
    else:
        q4 = "Decreasing"

    print(f"Question 3: Y's slope is {q3} after the cutoff.")
    print(f"Question 4: Y is {q4} before the cutoff.\n")

# --- Load both datasets ---
a = pd.read_csv("homework_4.2.a.csv")
b = pd.read_csv("homework_4.2.b.csv")

# Run the analysis
analyze_dataset(a, "X", "Y")
analyze_dataset(b, "X2", "Y2")
