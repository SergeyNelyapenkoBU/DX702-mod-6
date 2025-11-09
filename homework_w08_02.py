import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv

# --- Load data ---
df = pd.read_csv("homework_8.2.csv", index_col=0)
# Expected columns: X (0/1), Y, Z1, Z2

# --- Build inverse covariance (VI) from all Z1,Z2 as specified ---
Z = df[["Z1", "Z2"]].to_numpy().T   # shape (2, N)
cov = np.cov(Z)                      # 2x2 covariance
VI = inv(cov)                        # inverse covariance matrix

# --- Split treated and control pools ---
treated = df[df["X"] == 1].copy()
control = df[df["X"] == 0].copy()

Z_t = treated[["Z1","Z2"]].to_numpy()
Z_c = control[["Z1","Z2"]].to_numpy()

# --- Helper to compute pairwise Mahalanobis distances to a set (return nearest index and distance) ---
def nearest_mahalanobis(point, pool_matrix, VI):
    # point: shape (2,), pool_matrix: shape (M,2)
    dists = [mahalanobis(point, pool_matrix[j], VI) for j in range(pool_matrix.shape[0])]
    j_min = int(np.argmin(dists))
    return j_min, float(dists[j_min])

# --- Match each treated to nearest control (with replacement) ---
match_idx = []     # indices in 'control' DataFrame
match_dist = []    # nearest distances
for i in range(Z_t.shape[0]):
    j_min, dmin = nearest_mahalanobis(Z_t[i], Z_c, VI)
    match_idx.append(control.index[j_min])
    match_dist.append(dmin)

treated["matched_control_idx"] = match_idx
treated["mahal_dist_to_nearest_control"] = match_dist
treated = treated.merge(
    control[["Y","Z1","Z2"]].rename(columns={"Y":"Y_control","Z1":"Z1_control","Z2":"Z2_control"}),
    left_on="matched_control_idx", right_index=True, how="left"
)

# --- ATT: average over treated of Y_treated - Y_matched_control ---
ATT = (treated["Y"] - treated["Y_control"]).mean()

# --- (Optional) Symmetric ATE via two-way nearest-neighbor matching ---
# Additionally match each control to its nearest treated, then average the two direction-specific effects.
match_idx_c = []
match_dist_c = []
for j in range(Z_c.shape[0]):
    i_min, dmin = nearest_mahalanobis(Z_c[j], Z_t, VI)
    match_idx_c.append(treated.index[i_min])
    match_dist_c.append(dmin)

control["matched_treated_idx"] = match_idx_c
control["mahal_dist_to_nearest_treated"] = match_dist_c
control = control.merge(
    treated[["Y","Z1","Z2"]].rename(columns={"Y":"Y_treated","Z1":"Z1_treated","Z2":"Z2_treated"}),
    left_on="matched_treated_idx", right_index=True, how="left"
)

# Effect for controls: (matched treated) - (control)
effect_controls = (control["Y_treated"] - control["Y"]).mean()
# Combine both directions to approximate ATE
ATE_two_way = 0.5 * ((treated["Y"] - treated["Y_control"]).mean() + effect_controls)

# --- Identify treated item with least common support (largest nearest distance) ---
idx_worst = treated["mahal_dist_to_nearest_control"].idxmax()
row_worst = treated.loc[idx_worst]

worst_report = {
    "treated_index": int(idx_worst),
    "treated_Z1": float(row_worst["Z1"]),
    "treated_Z2": float(row_worst["Z2"]),
    "nearest_control_index": int(row_worst["matched_control_idx"]),
    "nearest_control_Z1": float(row_worst["Z1_control"]),
    "nearest_control_Z2": float(row_worst["Z2_control"]),
    "mahalanobis_distance": float(row_worst["mahal_dist_to_nearest_control"])
}

# --- Print results ---
print(f"ATT (treated matched to nearest controls, with replacement): {ATT:.6f}")
print(f"ATE (two-way nearest-neighbor matching approximation): {ATE_two_way:.6f}")
print("\nTreated unit with least common support (largest nearest-neighbor Mahalanobis distance):")
for k, v in worst_report.items():
    print(f"  {k}: {v}")

"""
ATT (treated matched to nearest controls, with replacement): 3.437679
ATE (two-way nearest-neighbor matching approximation): 3.408055

Treated unit with least common support (largest nearest-neighbor Mahalanobis distance):
  treated_index: 494
  treated_Z1: 2.6962240525635797
  treated_Z2: 0.5381554886023228
  nearest_control_index: 418
  nearest_control_Z1: 1.5199948607657727
  nearest_control_Z2: -1.2822079376259403
  mahalanobis_distance: 1.3830045328325054
"""

"""
Distance metric: I used Mahalanobis (per instructions) with VI = Σ⁻¹ built from the 2*2 covariance of all Z1, Z2.

Matching scheme: Nearest neighbor with replacement—each treated unit picks the closest control by Mahalanobis distance; a control can be reused.

Estimand: This one-way matching naturally estimates ATT. To provide an ATE, I also performed symmetric (two-way) matching (controls→treated) and averaged both directions.

Common support diagnostic: The treated unit whose nearest control is farthest in Mahalanobis distance is flagged as having the weakest overlap; I print its Z1, Z2, its matched control's Z1, Z2, and the distance. This helps you discuss overlap/robustness.
"""