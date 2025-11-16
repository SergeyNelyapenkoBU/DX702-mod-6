import numpy as np
import statsmodels.api as sm

np.random.seed(42)
num = 10000

Z = np.random.normal(0, 1, (num,))
X = Z + np.random.normal(0, 1, (num,))
Y = 1.5 * X + 2.3 * Z + np.random.normal(0, X**2, (num,))

features = sm.add_constant(np.column_stack([X, Z]))
model = sm.OLS(Y, features).fit()

se_python = model.bse[1]
print(f"1) Python standard error for X: {se_python:.6f}")

n_simulations = 100
coefficients = []

for _ in range(n_simulations):
    Z_sim = np.random.normal(0, 1, (num,))
    X_sim = Z_sim + np.random.normal(0, 1, (num,))
    Y_sim = 1.5 * X_sim + 2.3 * Z_sim + np.random.normal(0, X_sim**2, (num,))

    features_sim = sm.add_constant(np.column_stack([X_sim, Z_sim]))
    model_sim = sm.OLS(Y_sim, features_sim).fit()
    coefficients.append(model_sim.params[1])

se_simulation = np.std(coefficients, ddof=1)
print(f"2) Simulation standard error for X: {se_simulation:.6f}")

print(f"\nRatio (2/1): {se_simulation/se_python:.4f}")
