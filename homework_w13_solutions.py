import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS

print("=" * 60)
print("QUESTION 1: Instrumental Variables - Effect of X on Y")
print("=" * 60)

# Load the data
df1 = pd.read_csv('homework_13.1.csv')
print("\nDataset columns:", df1.columns.tolist())

# Two-Stage Least Squares (2SLS):
# Stage 1: Regress X on Z
X1_stage1 = sm.add_constant(df1['Z'])
model1_stage1 = sm.OLS(df1['X'], X1_stage1).fit()
print("\nStage 1: X ~ Z")
print("Coefficient of Z:", model1_stage1.params['Z'])

# Get predicted X
X_predicted = model1_stage1.predict(X1_stage1)

# Stage 2: Regress Y on predicted X
X1_stage2 = sm.add_constant(X_predicted)
model1_stage2 = sm.OLS(df1['Y'], X1_stage2).fit()
print("\nStage 2: Y ~ X_predicted")
print("Coefficient (effect of X on Y):", model1_stage2.params.iloc[1])

# Alternative: Use IV2SLS directly
endog = df1['Y']
exog = sm.add_constant(df1['X'])
instrument = sm.add_constant(df1['Z'])

iv_model = IV2SLS(endog, exog, instrument).fit()
print("\nIV coefficient for X:", iv_model.params['X'])

print("\n" + "=" * 60)
print("QUESTION 2: Two-Stage Least Squares - Effect of X2 on Y2")
print("=" * 60)

df2 = pd.read_csv('homework_13.2.csv')
print("\nDataset columns:", df2.columns.tolist())

# Stage 1: X2 ~ Z2
X2_stage1 = sm.add_constant(df2['Z2'])
model2_stage1 = sm.OLS(df2['X2'], X2_stage1).fit()
print("\nStage 1: X2 ~ Z2")
print("Coefficient of Z2:", model2_stage1.params['Z2'])

# Get predicted X2
X2_predicted = model2_stage1.predict(X2_stage1)

# Stage 2: Y2 ~ X2_predicted
X2_stage2 = sm.add_constant(X2_predicted)
model2_stage2 = sm.OLS(df2['Y2'], X2_stage2).fit()
print("\nStage 2: Y2 ~ X2_predicted")
print("Coefficient (effect of X2 on Y2):", model2_stage2.params.iloc[1])

# Alternative: Use IV2SLS directly
endog2 = df2['Y2']
exog2 = sm.add_constant(df2['X2'])
instrument2 = sm.add_constant(df2['Z2'])

iv_model2 = IV2SLS(endog2, exog2, instrument2).fit()
print("\nIV coefficient for X2:", iv_model2.params['X2'])

print("\n" + "=" * 60)
print("QUESTION 3: Identifying Compliers")
print("=" * 60)

df3 = pd.read_csv('homework_13.3.csv')
print("\nDataset columns:", df3.columns.tolist())

# Regress X3 on Z3, W3, and ZW_int to find marginal effects
X_vars = sm.add_constant(df3[['Z3', 'W3', 'ZW_int']])
model3 = sm.OLS(df3['X3'], X_vars).fit()
print("\nRegression: X3 ~ Z3 + W3 + ZW_int")
print(model3.params)

# Calculate marginal effect of Z3 on X3 for each individual
# dX3/dZ3 = beta_Z3 + beta_ZW_int * W3
beta_Z3 = model3.params['Z3']
beta_ZW_int = model3.params['ZW_int']

print(f"\nMarginal effect formula:")
print(f"dX3/dZ3 = {beta_Z3:.4f} + ({beta_ZW_int:.4f}) * W3")
print(f"dX3/dZ3 ~= 1 - W3")

# Calculate for each individual
df3['marginal_effect_Z3'] = beta_Z3 + beta_ZW_int * df3['W3']

# Analyze by W3 sign
print("\n" + "=" * 60)
print("Complier Analysis by W3 Sign:")
print("=" * 60)

negative_W3_compliers = ((df3['W3'] < 0) & (df3['marginal_effect_Z3'] > 0)).sum()
negative_W3_total = (df3['W3'] < 0).sum()

positive_W3_compliers = ((df3['W3'] > 0) & (df3['marginal_effect_Z3'] > 0)).sum()
positive_W3_total = (df3['W3'] > 0).sum()

print(f"\nNegative W3:")
print(f"  Total: {negative_W3_total}")
print(f"  Compliers: {negative_W3_compliers}")
print(f"  Percentage: {negative_W3_compliers/negative_W3_total*100:.1f}%")

print(f"\nPositive W3:")
print(f"  Total: {positive_W3_total}")
print(f"  Compliers: {positive_W3_compliers}")
print(f"  Percentage: {positive_W3_compliers/positive_W3_total*100:.1f}%")

print("\n" + "=" * 60)
print("SUMMARY OF ANSWERS")
print("=" * 60)

print(f"\nQuestion 1: {iv_model.params['X']:.4f}")
print(f"Question 2: {iv_model2.params['X2']:.4f}")
print(f"Question 3: All negative W3 are compliers (100%), only {positive_W3_compliers/positive_W3_total*100:.1f}% of positive W3 are compliers")
