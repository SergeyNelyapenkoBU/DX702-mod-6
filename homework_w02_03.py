import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv("homework_2.2.csv", index_col=0)

# Parameters
n_boot = 1000  # number of bootstrap replications
n = len(df)

# Store bootstrap results
boot_coefs = []

for _ in range(n_boot):
    # Sample with replacement
    sample = df.sample(n=n, replace=True)
    
    # Example: run regression Y ~ Z
    X = sm.add_constant(sample["Z"])
    y = sample["Y"]
    model = sm.OLS(y, X).fit()
    
    # Save coefficient of Z
    boot_coefs.append(model.params["Z"])

boot_coefs = np.array(boot_coefs)

# Summarize
print("Bootstrap mean coefficient:", boot_coefs.mean())
print("Bootstrap std error:", boot_coefs.std())
print("95% CI:", np.percentile(boot_coefs, [2.5, 97.5]))
