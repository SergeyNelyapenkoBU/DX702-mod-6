import numpy as np

np.random.seed(0)

W = np.random.normal(0, 1, (1000,))
X = W + np.random.normal(0, 1, (1000,))
Z = np.random.normal(0, 1, (1000,))
Y = X + Z + W + np.random.normal(0, 1, (1000,))


epsilon = Y - (X + Z + W)

# Question #1
# Correlation between X and epsilon
corr = np.corrcoef(X, epsilon)[0, 1]
print(f"Question #1. Estimated corr(X, u): {corr:.4f}")

theory = 1/np.sqrt(2) 
print(f"Question #2. Theoretical corr(X, u): {theory:.4f}")
