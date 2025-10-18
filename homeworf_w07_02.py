import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- Settings ---
csv_path = "homework_7.1.csv"
targets = [-1.0, 0.0, 1.0]   # W values at which we want "constant W" analyses
bandwidth_initial = 0.15     # initial half-width for the W window
min_rows = 40                # try to ensure at least this many rows per window
max_bandwidth = 1.0          # don't widen beyond this
same_tol = 0.2               # "about the same" tolerance for X-coef differences
mono_tol = 0.05              # minimal gap to call a strict monotone change

def fit_local(df, w0, h, min_rows=40, max_h=1.0):
    """
    Fit Y ~ X + Z using rows with |W - w0| <= h. If too few rows, expand h.
    Returns (coef_X, n, used_bandwidth)
    """
    hh = h
    while True:
        sub = df[np.abs(df["W"] - w0) <= hh]
        if len(sub) >= min_rows or hh >= max_h:
            break
        hh *= 1.5  # widen window adaptively

    if len(sub) < 3:
        raise ValueError(f"Too few rows around W={w0} even with bandwidth={hh:.3f} (n={len(sub)}).")

    Xmat = sub[["X", "Z"]].values
    yvec = sub["Y"].values
    reg = LinearRegression().fit(Xmat, yvec)
    coef_X = float(reg.coef_[0])  # coefficient on X
    return coef_X, len(sub), hh

def decide_trend(coefs, same_tol=0.2, mono_tol=0.05):
    """
    coefs is a dict: {w0: coef_X_at_w0}
    Returns one of: "increasing", "about the same", "decreasing"
    """
    c_minus, c_zero, c_plus = coefs[-1.0], coefs[0.0], coefs[1.0]
    cvals = np.array([c_minus, c_zero, c_plus])

    # If all within same_tol range, call it "about the same"
    if cvals.max() - cvals.min() <= same_tol:
        return "staying about the same (say, within 0.2 or so)"

    # Monotone increasing?
    if (c_zero - c_minus) > mono_tol and (c_plus - c_zero) > mono_tol:
        return "increasing"

    # Monotone decreasing?
    if (c_minus - c_zero) > mono_tol and (c_zero - c_plus) > mono_tol:
        return "decreasing"

    # Not strictly monotone but spread is larger than same_tol: default to "about the same"
    return "staying about the same (say, within 0.2 or so)"

def main():
    df = pd.read_csv(csv_path).dropna(subset=["X", "Y", "Z", "W"])

    coefs = {}
    counts = {}
    bands = {}

    for w0 in targets:
        coef_x, n, used_h = fit_local(df, w0, bandwidth_initial, min_rows=min_rows, max_h=max_bandwidth)
        coefs[w0] = coef_x
        counts[w0] = n
        bands[w0] = used_h

    print("Local regressions of Y ~ X + Z at W ≈ -1, 0, 1 (using |W-w0| <= bandwidth):")
    for w0 in targets:
        print(f"  W={w0:+.0f}: coef_X={coefs[w0]: .4f}  (n={counts[w0]}, bandwidth={bands[w0]:.3f})")

    verdict = decide_trend(coefs, same_tol=same_tol, mono_tol=mono_tol)
    print("\nQuestion 3 — Is Y's slope (coef on X) higher or lower after the cutoff compared with before?")
    print("→ This script answers your requested trend over W, not a cutoff; result over W:", verdict)

    # Map to the multiple-choice phrasing:
    if "increasing" in verdict:
        q3_choice = "Option A — increasing"
    elif "decreasing" in verdict:
        q3_choice = "Option C — decreasing"
    else:
        q3_choice = "Option B — staying about the same (within ~0.2)"
    print("Final choice for the prompt's options:", q3_choice)

if __name__ == "__main__":
    main()

    """
    We can't literally set W to a constant, so we use a narrow window around each target
    and adaptively widen it until we have enough rows.
    Within each window, we run Y ~ X + Z and take the coefficient on X.
    Then we compare the coefficients across W = -1, 0, 1 and classify them as Increasing, About the same (~0.2), or Decreasing.
    """
