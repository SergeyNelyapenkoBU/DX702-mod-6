import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

 
num = 100_000      # samples per dataset
num_runs = 50      # how many datasets to simulate

betas_difficulty = []
betas_accident = []

for _ in range(num_runs):
 
    difficulty = np.random.uniform(0, 1, (num,))
    speed = np.maximum(np.random.normal(15, 5, (num, )) - difficulty * 10, 0)
    accident = np.minimum(np.maximum(0.03 * speed + 0.4 * difficulty + np.random.normal(0, 0.3, (num,)), 0), 1)
    df = pd.DataFrame({'difficulty': difficulty, 'speed': speed, 'accident': accident})

    # regress Y on X and Z: speed ~ difficulty + accident
    reg = LinearRegression().fit(df[['difficulty', 'accident']], df['speed'])
    betas_difficulty.append(reg.coef_[0])  # coefficient on X
    betas_accident.append(reg.coef_[1])    # coefficient on Z

betas_difficulty = np.array(betas_difficulty)
betas_accident = np.array(betas_accident)

print(f"Average coef on difficulty (X): {betas_difficulty.mean():.4f}")
print(f"Std dev across runs:            {betas_difficulty.std(ddof=1):.4f}")
print(f"(For reference) avg coef on accident (Z): {betas_accident.mean():.4f}")


# Question: In practice, should we run such a regression? We are controlling for Z, but Z is a collider. That is, Y and Z both cause Z. Should we control of it or are we better off ignoring Z? Why or why not? 
# Answer: We should not control for Z, because it is a collider. Conditioning on a collider opens a spurious backdoor path between X and Y, biasing the estimate. If your goal is the total causal effect of difficulty (X) on speed (Y), you’re better off ignoring Z (and using proper design/controls that block confounding, not colliders).

