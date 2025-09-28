import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import skew

# Load dataset
df = pd.read_csv("homework_2.2.csv", index_col=0)

n_boot = 1000
n = len(df)

boot_coefs = []

for _ in range(n_boot):
    # Resample with replacement
    sample = df.sample(n=n, replace=True)
    
    # Regression Y ~ X (with intercept)
    X = sample[["X"]]
    y = sample["Y"]
    model = LinearRegression().fit(X, y)
    
    # Store coefficient of X
    boot_coefs.append(model.coef_[0])

# Compute skewness of bootstrap coefficient distribution
coef_skewness = skew(boot_coefs)

print("Bootstrap skewness of effect:", coef_skewness)
