import pandas as pd
import statsmodels.api as sm

# Load data
df = pd.read_csv("homework_2.1.csv")

# Long format
df_long = df.melt(id_vars=["time"], value_vars=["G1", "G2", "G3"],
                  var_name="group", value_name="Y")

# Make dummies for group (drop_first=True so G1 is baseline)
dummies = pd.get_dummies(df_long["group"], drop_first=True)

# Design matrix: numeric only
X = pd.concat([df_long[["time"]], dummies], axis=1)
X = sm.add_constant(X)

# Ensure all numeric (prevents "object dtype" issue)
X = X.astype(float)
y = df_long["Y"].astype(float)

# Fit OLS
model = sm.OLS(y, X).fit()
print(model.summary())

# Extract group 1 fixed effect (the intercept)
group1_intercept = model.params["const"]
print("\nEstimated fixed effect (intercept) for Group 1:", group1_intercept)
