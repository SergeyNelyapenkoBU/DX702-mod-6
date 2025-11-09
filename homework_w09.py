"""
Homework Week 9: Statistical Power and Hypothesis Testing
Boston University DX702 - Experimental Design and Causality

This script explores statistical power, hypothesis testing, and the factors that
affect the ability to detect true effects in regression analysis.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

def simulate(A=1, B=1, C=10, D=1000):
    """
    Data Generating Process (DGP) for the homework.

    Parameters:
    - A: True effect of X on Y
    - B: Standard deviation of noise in X
    - C: Standard deviation of noise in Y
    - D: Sample size

    Returns:
    - Y: Outcome variable
    - X: Treatment/predictor variable
    - W: Confounder variable
    """
    W = np.random.normal(0, 1, D)
    X = W + np.random.normal(0, B, D)
    Y = A * X - W + np.random.normal(0, C, D)
    return Y, X, W


def run_regression_with_W(Y, X, W):
    """
    Run OLS regression of Y on X, controlling for W.

    Returns:
    - coefficient on X
    - t-statistic for X
    - p-value for X
    """
    # Add constant and W to the model
    X_with_controls = sm.add_constant(pd.DataFrame({'X': X, 'W': W}))
    model = sm.OLS(Y, X_with_controls).fit()

    # Extract statistics for X (index 1, after constant)
    coef_X = model.params['X']
    t_stat_X = model.tvalues['X']
    p_value_X = model.pvalues['X']

    return coef_X, t_stat_X, p_value_X


# Question 1: Probability of detecting nonzero effect
# (Power of the test with A=1, B=1, C=10, D=1000)
print("=" * 70)
print("Question 1: Power of test (A=1, B=1, C=10, D=1000)")
print("=" * 70)

np.random.seed(42)
n_simulations = 1000
A, B, C, D = 1, 1, 10, 1000

detected_count = 0
for i in range(n_simulations):
    Y, X, W = simulate(A=A, B=B, C=C, D=D)
    coef, t_stat, p_value = run_regression_with_W(Y, X, W)

    # Check if |t-statistic| > 1.96 (significant at 5% level)
    if abs(t_stat) > 1.96:
        detected_count += 1

power = detected_count / n_simulations
print(f"Number of simulations: {n_simulations}")
print(f"Significant results (|t| > 1.96): {detected_count}")
print(f"Power (probability of detection): {power:.3f} or {power*100:.1f}%")
print()


# Question 2: Skew of the estimate
print("=" * 70)
print("Question 2: Skew of the coefficient estimates")
print("=" * 70)

np.random.seed(42)
n_simulations = 1000
A, B, C, D = 1, 1, 10, 1000

coefficients = []
for i in range(n_simulations):
    Y, X, W = simulate(A=A, B=B, C=C, D=D)
    coef, t_stat, p_value = run_regression_with_W(Y, X, W)
    coefficients.append(coef)

coefficients = np.array(coefficients)
skewness = stats.skew(coefficients)

print(f"Number of simulations: {n_simulations}")
print(f"Mean of estimates: {np.mean(coefficients):.4f}")
print(f"Std of estimates: {np.std(coefficients):.4f}")
print(f"Skewness of estimates: {skewness:.4f}")
print()


# Question 3: Find B value for 50% power (A=1, C=10, D=1000)
print("=" * 70)
print("Question 3: Find B for 50% power (A=1, C=10, D=1000)")
print("=" * 70)

np.random.seed(42)
n_simulations = 500  # Reduced for speed
A, C, D = 1, 10, 1000

# Test different values of B
B_values = [0.2, 0.6, 1.8, 5.4]

print("Testing different B values:")
for B in B_values:
    detected_count = 0
    for i in range(n_simulations):
        Y, X, W = simulate(A=A, B=B, C=C, D=D)
        coef, t_stat, p_value = run_regression_with_W(Y, X, W)
        if abs(t_stat) > 1.96:
            detected_count += 1

    power = detected_count / n_simulations
    print(f"  B = {B:4.1f}: Power = {power:.3f} ({power*100:.1f}%)")

print()


# Question 4: Find A value for 50% power (B=1, C=10, D=100)
print("=" * 70)
print("Question 4: Find A for 50% power (B=1, C=10, D=100)")
print("=" * 70)

np.random.seed(42)
n_simulations = 500  # Reduced for speed
B, C, D = 1, 10, 100

# Test different values of A
A_values = [0.5, 1.0, 2.0, 4.0]

print("Testing different A values:")
for A in A_values:
    detected_count = 0
    for i in range(n_simulations):
        Y, X, W = simulate(A=A, B=B, C=C, D=D)
        coef, t_stat, p_value = run_regression_with_W(Y, X, W)
        if abs(t_stat) > 1.96:
            detected_count += 1

    power = detected_count / n_simulations
    print(f"  A = {A:4.1f}: Power = {power:.3f} ({power*100:.1f}%)")

print()
print("=" * 70)
print("Analysis Complete")
print("=" * 70)
