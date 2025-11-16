import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv("homework_10.1.csv", index_col=0)

time_effects = []
for t in range(12):
    data_t = df[df['time'] == t]
    X = sm.add_constant(data_t[['X']])
    model = sm.OLS(data_t['y'], X).fit()
    time_effects.append(model.params['const'])

print("Time fixed effects (months 0-11):")
for i, effect in enumerate(time_effects):
    print(f"Month {i}: {effect:.4f}")
