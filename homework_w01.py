import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

# 1) Load data (first column is just an index -> drop it)
df = pd.read_csv("homework_1.1.csv", index_col=0)

# 2) Features and target
X = df[["X1", "X2", "X3"]]
y = df["Y"]

# 3) Fit linear regression
model = LinearRegression()
model.fit(X, y)

## Question 1

# 4) Inspect and evaluate (in-sample)
print("Intercept:", model.intercept_)
print("Coefficients:", dict(zip(X.columns, model.coef_)))
print("R^2 (in-sample):", r2_score(y, model.predict(X)))


################################################
## Question 2
X_cols = ["X1", "X2", "X3"]
y = df["Y"]
X_all = df[X_cols]

# Multiple regression (ceteris paribus effects)
mod_all = sm.OLS(y, sm.add_constant(X_all)).fit()
betas_multi = mod_all.params[X_cols]  # slopes holding others fixed

rows = []
for xi in X_cols:
    # Simple regression (as-observed association)
    Xi = df[[xi]]
    mod_simple = sm.OLS(y, sm.add_constant(Xi)).fit()
    beta_simple = mod_simple.params[xi]

    beta_multi = betas_multi[xi]
    diff = beta_simple - beta_multi
    rows.append({
        "Xi": xi,
        "slope_simple_Y_on_Xi": beta_simple,
        "slope_multi_Y_on_Xi_given_others": beta_multi,
        "difference_simple_minus_multi": diff,
        "abs_difference": abs(diff),
    })

out = pd.DataFrame(rows).sort_values("abs_difference", ascending=False)
print(out)

# Identify which Xi has the greatest difference
top = out.iloc[0]
print("\nGreatest difference is for:", top['Xi'])


## Question 3
# Fit OLS with intercept
Xc = sm.add_constant(X, has_constant="add")
model = sm.OLS(y, Xc).fit()

# 3) Get t-stats (absolute), exclude intercept
tvals = model.tvalues.drop(labels=["const"], errors="ignore").abs().sort_values(ascending=False)

print("t-statistics (absolute), sorted:")
print(tvals)

most_sig = tvals.index[0]
print(f"\nMost significant coefficient by |t|: {most_sig} (t = {tvals.iloc[0]:.3f})")

# Optional: also show p-values
pvals = model.pvalues.loc[tvals.index]
print("\nCorresponding p-values:")
print(pvals)

## Question 4


# Load CSV (first column is just an index)
df = pd.read_csv("homework_1.2.csv", index_col=0)

# Split treated and control
treated = df[df["X"] == 1].reset_index(drop=True)
control = df[df["X"] == 0].reset_index(drop=True)

# Nearest neighbor in Z (with replacement)
nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
nn.fit(control[["Z"]])
distances, indices = nn.kneighbors(treated[["Z"]], return_distance=True)

# Q4: farthest match distance
farthest = float(distances.max())

# Matched controls (use indices found above)
matched_controls = control.iloc[indices.ravel()].reset_index(drop=True)

# Q5: effect = mean(Y | X=1) - mean(Y | matched X=0)
effect = float(treated["Y"].mean() - matched_controls["Y"].mean())

print(f"Q4 - farthest match distance: {farthest:.6f}")
print(f"Q5 - effect (treated mean Y - matched control mean Y): {effect:.6f}")


## Questions 6 and 7
# Load data (first column is just an index)
df = pd.read_csv("homework_1.2.csv", index_col=0)

# Split treated and control
treated = df[df["X"] == 1].reset_index(drop=True)
control = df[df["X"] == 0].reset_index(drop=True)

# Radius-NN on Z (duplicates across groups allowed)
radius = 0.2
nn = NearestNeighbors(radius=radius, metric="euclidean")
nn.fit(control[["Z"]])

# For each treated item, find all control neighbors within radius
dist_lists, idx_lists = nn.radius_neighbors(treated[["Z"]], return_distance=True)

# Keep only treated units that have ≥1 neighbor
kept = [i for i, idxs in enumerate(idx_lists) if len(idxs) > 0]
if not kept:
    print(f"No matches found within radius {radius}.")
else:
    # ----- Q6: How many duplicates? -----
    # Count how many times control rows are reused across groups,
    # counting all but the first occurrence of each control index.
    all_ctrl_indices = np.concatenate([idx_lists[i] for i in kept])
    total_matches = len(all_ctrl_indices)
    unique_controls = len(np.unique(all_ctrl_indices))
    duplicates = total_matches - unique_controls
    print(f"Q6 - duplicates (counting all but the first per control): {duplicates}")

    # ----- Q7: Effect -----
    # For each treated unit, take mean Y over its control neighbors (group mean),
    # then average those group means; subtract from mean Y among kept treated.
    treated_mean = treated.loc[kept, "Y"].mean()
    group_ctrl_means = [control.loc[idx_lists[i], "Y"].mean() for i in kept]
    matched_ctrl_mean = float(np.mean(group_ctrl_means))
    effect = float(treated_mean - matched_ctrl_mean)

    print(f"Q7 - effect = mean(Y | X=1, kept) - mean(group-mean Y of matched controls): {effect:.6f}")
    # (Optional helpers)
    print(f"(n treated kept = {len(kept)}, total matched pairs = {total_matches}, unique controls used = {unique_controls})")
