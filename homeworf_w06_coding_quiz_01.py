import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# --- Load data ---
df = pd.read_csv("homework_6.1.csv")  # expects columns: X (0/1), Z (confounder), Y (outcome)

# Basic checks & cleanup
df = df.dropna(subset=["X", "Z", "Y"])
treated = df[df["X"] == 1].reset_index(drop=True)
control = df[df["X"] == 0].reset_index(drop=True)

if treated.empty or control.empty:
    raise ValueError("Need both treated (X=1) and untreated (X=0) observations.")

# --- Match treated -> nearest control on Z ---
nn_ctrl = NearestNeighbors(n_neighbors=1, metric="euclidean")
nn_ctrl.fit(control[["Z"]])

dist_tc, idx_ctrl = nn_ctrl.kneighbors(treated[["Z"]], return_distance=True)
ctrl_match_for_treated = control.iloc[idx_ctrl.ravel()].reset_index(drop=True)

# Effects for treated items: Y(1)_i - Y(0)_i^cf  (cf from matched control)
eff_treated = treated["Y"].to_numpy() - ctrl_match_for_treated["Y"].to_numpy()

# --- Match untreated -> nearest treated on Z ---
nn_treat = NearestNeighbors(n_neighbors=1, metric="euclidean")
nn_treat.fit(treated[["Z"]])

dist_ct, idx_treat = nn_treat.kneighbors(control[["Z"]], return_distance=True)
treat_match_for_control = treated.iloc[idx_treat.ravel()].reset_index(drop=True)

# Effects for untreated items: Y(1)_i^cf - Y(0)_i  (cf from matched treated)
eff_untreated = treat_match_for_control["Y"].to_numpy() - control["Y"].to_numpy()

# --- ATE, ATT, ATUT, Optimal treatment effect (as defined) ---
ATE  = np.mean(np.concatenate([eff_treated, eff_untreated]))
ATT  = np.mean(eff_treated)          # average over treated only
ATUT = np.mean(eff_untreated)        # average over untreated only
OPT  = np.max(eff_untreated)         # maximum effect across untreated items (single best)

# --- Report ---
print("Matching: 1-NN on Z (with replacement)")
print(f"Pairs used: treated->{len(eff_treated)}, untreated->{len(eff_untreated)}")
print(f"Avg dist (treated->control): {dist_tc.mean():.4f}, farthest: {dist_tc.max():.4f}")
print(f"Avg dist (control->treated): {dist_ct.mean():.4f}, farthest: {dist_ct.max():.4f}\n")

print(f"ATE  (all units):            {ATE:.6f}")
print(f"ATT  (treated only):         {ATT:.6f}")
print(f"ATUT (untreated only):       {ATUT:.6f}")
print(f"Optimal treatment effect:    {OPT:.6f}  (max over untreated)")
