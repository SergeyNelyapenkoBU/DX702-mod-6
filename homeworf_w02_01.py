import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("homework_2.2.csv", index_col=0)

# Parameters
n_boot = 1000
n = len(df)

boot_effects = []

for _ in range(n_boot):
    # Sample with replacement
    sample = df.sample(n=n, replace=True)
    
    # Mean outcome for treated (X=1) and control (X=0)
    mean_treated = sample.loc[sample["X"] == 1, "Y"].mean()
    mean_control = sample.loc[sample["X"] == 0, "Y"].mean()
    
    # Store naive effect
    boot_effects.append(mean_treated - mean_control)

boot_effects = np.array(boot_effects)

# Summarize
print("Naive effect (original sample):",
      df.loc[df["X"]==1,"Y"].mean() - df.loc[df["X"]==0,"Y"].mean())
print("Bootstrap mean naive effect:", boot_effects.mean())
print("Bootstrap std error:", boot_effects.std())
print("95% CI:", np.percentile(boot_effects, [2.5, 97.5]))
