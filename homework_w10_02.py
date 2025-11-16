import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv("homework_10.1.csv", index_col=0)

city_effects = []
for c in range(10):
    data_c = df[df['city'] == c]
    X = sm.add_constant(data_c[['X']])
    model = sm.OLS(data_c['y'], X).fit()
    city_effects.append(model.params['const'])

print("City fixed effects (cities 0-9):")
for i, effect in enumerate(city_effects):
    print(f"City {i}: {effect:.4f}")
