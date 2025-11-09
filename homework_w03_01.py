import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load data
df = pd.read_csv("homework_3.1.csv")

# Create event indicator and interaction term
cutoff = 50
df["D"] = (df["time"] >= cutoff).astype(int)
df["time_x_D"] = df["time"] * df["D"]

results = {}

for col in ["value1", "value2", "value3"]:
    y = df[col]

    # --- Model with level discontinuity only ---
    X1 = sm.add_constant(df[["time", "D"]])
    model1 = sm.OLS(y, X1).fit()

    # --- Model with both level and slope discontinuity ---
    X2 = sm.add_constant(df[["time", "D", "time_x_D"]])
    model2 = sm.OLS(y, X2).fit()

    # Save results
    results[col] = {
        "jump_coef": model2.params.get("D", np.nan),
        "jump_pval": model2.pvalues.get("D", np.nan),
        "slope_change_coef": model2.params.get("time_x_D", np.nan),
        "slope_change_pval": model2.pvalues.get("time_x_D", np.nan),
        "R2_level": model1.rsquared,
        "R2_slope": model2.rsquared
    }

    # --- Plot ---
    plt.figure(figsize=(7,4))
    plt.scatter(df["time"], y, alpha=0.6, label="data")

    # Predictions from model 2
    pred = model2.predict(X2)
    plt.plot(df["time"], pred, color="red", label="fitted (with slope change)")

    # Vertical line at the event point
    plt.axvline(x=cutoff, color="black", linestyle="--", label="event @ 50")

    plt.title(f"{col}: jump={results[col]['jump_coef']:.3f} (p={results[col]['jump_pval']:.3g}), "
              f"slopeΔ={results[col]['slope_change_coef']:.3f} (p={results[col]['slope_change_pval']:.3g})")
    plt.xlabel("time")
    plt.ylabel(col)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Final summary table of results
results_df = pd.DataFrame(results).T
print("\n=== Summary by variables ===")
print(results_df.round(4))
