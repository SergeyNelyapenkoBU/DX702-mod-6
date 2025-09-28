import pandas as pd
import statsmodels.formula.api as smf

# Load datasets
a = pd.read_csv("homework_3.2.a.csv").rename(columns={"group1":"group","time1":"time","outcome1":"outcome"})
b = pd.read_csv("homework_3.2.b.csv").rename(columns={"group2":"group","time2":"time","outcome2":"outcome"})

def did_stats(df, name):
    # Run DiD regression
    model = smf.ols("outcome ~ group + time + group:time", data=df).fit(cov_type="HC1")
    
    # Extract results for the interaction term
    coef = model.params["group:time"]
    se   = model.bse["group:time"]
    tval = model.tvalues["group:time"]
    pval = model.pvalues["group:time"]
    
    print(f"\n{name}")
    print(f"  Coefficient (DiD estimate): {coef:.4f}")
    print(f"  Standard Error:            {se:.4f}")
    print(f"  t-statistic:               {tval:.2f}")
    print(f"  p-value:                   {pval:.3g}")
    
    return coef, se, tval, pval

# Run for both datasets
did_stats(a, "Dataset A")
did_stats(b, "Dataset B")
