import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# --- Load data ---
# The CSV columns are: index, X (treatment: 0/1), Y (outcome), Z (covariate)
df = pd.read_csv("homework_8.1.csv", index_col=0)

# --- Step 1: Fit logistic regression for propensity scores ---
# Model P(X=1 | Z): feature is Z (reshape to 2D), label is X
X_treat = df["X"].values.astype(int)                 # treatment indicator
Z_feat = df[["Z"]].values                            # covariate(s) as 2D array

logit = LogisticRegression(solver="lbfgs")
logit.fit(Z_feat, X_treat)

# --- Step 2: Predict propensity scores ---
# predict_proba returns [P(X=0), P(X=1)] columns
p = logit.predict_proba(Z_feat)[:, 1]                # propensity: P(X=1|Z)

# (Optional but recommended) clip extreme propensities to avoid infinite/unstable weights
eps = 1e-6
p = np.clip(p, eps, 1 - eps)

df["propensity"] = p

# --- Step 3: Compute inverse probability weights ---
# w = 1/p for treated; w = 1/(1-p) for controls
w = np.where(X_treat == 1, 1.0 / p, 1.0 / (1.0 - p))
df["ipw"] = w

# --- Step 4: IPW ATE (difference in weighted means of Y) ---
Y = df["Y"].values

treated_mask = (X_treat == 1)
control_mask = ~treated_mask

# Weighted means
y_treat_w = np.sum(w[treated_mask] * Y[treated_mask]) / np.sum(w[treated_mask])
y_ctrl_w  = np.sum(w[control_mask]  * Y[control_mask])  / np.sum(w[control_mask])

ate_ipw = y_treat_w - y_ctrl_w

# --- Outputs ---
print("Average Treatment Effect (IPW):", ate_ipw)
print("\nPropensity scores for the first three rows:")
print(df.loc[df.index[:3], ["propensity"]])

"""
Average Treatment Effect (IPW): 2.2743411898510133

Propensity scores for the first three rows:
   propensity
0    0.840114
1    0.584646
2    0.711082
"""

"""
What I did (and why)

- Model: Followed your instruction to “fit the model so that the Z values predict X,” i.e., a logistic regression for 
P(X=1|Z). This is the standard way to estimate propensity scores; other classifiers would also work, but logistic regression is transparent and stable here.

- Predictions: Used predict_proba to get propensities pi. 

- Weights: Constructed inverse probability weights: wi=1/pi  for treated and wi=1/(1-pi​) for controls.

Estimator: Computed the IPW ATE as the difference of weighted means of Y between treated and control groups. This matches your step-by-step description.

Numerical stability: Clipped propensities to avoid infinite weights if any pi  is numerically 0 or 1. This doesn't change results materially but prevents runtime issues in small/sparse samples.
"""