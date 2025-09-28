import pandas as pd
import statsmodels.formula.api as smf

# Load
a = pd.read_csv("homework_3.2.a.csv").rename(columns={"group1":"group","time1":"time","outcome1":"outcome"})
b = pd.read_csv("homework_3.2.b.csv").rename(columns={"group2":"group","time2":"time","outcome2":"outcome"})

def did_effect(df):
    model = smf.ols("outcome ~ group + time + group:time", data=df).fit(cov_type="HC1")
    return model.params["group:time"], model.pvalues["group:time"]

for name, data in [("Dataset A", a), ("Dataset B", b)]:
    coef, pval = did_effect(data)
    print(f"{name}: DiD estimate = {coef:.3f}, p-value = {pval:.3g}")
