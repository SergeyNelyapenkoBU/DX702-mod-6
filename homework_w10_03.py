import numpy as np
import statsmodels.api as sm

np.random.seed(42)
num = 10000

X = np.clip(np.random.normal(3, 1, (num,)), 0.01, 100)
Z = np.clip(np.random.normal(3, 1, (num,)), 0.01, 100)
Y = np.log(X + Z) + np.random.normal(0, 1, (num,))

exp_Y = np.exp(Y)

features = sm.add_constant(np.column_stack([X, Z]))
model = sm.OLS(exp_Y, features).fit()

print("Coefficients for modeling exp(Y) as function of X and Z:")
print(f"X coefficient: {model.params[1]:.4f}")
print(f"Z coefficient: {model.params[2]:.4f}")
