import numpy as np
import pandas as pd
import statsmodels.api as sm

def make_error(corr_const, num):
    err = []
    prev = np.random.normal(0, 1)
    for _ in range(num):
        prev = corr_const * prev + (1 - corr_const) * np.random.normal(0, 1)
        err.append(prev)
    return np.array(err)

def run_sim(corr_const, num=500, trials=500, beta=2.0, seed=0):
    rng = np.random.default_rng(seed)
    betas = []
    ses = []

    for t in range(trials):
        # Controls (optional): exogenous covariate(s) with i.i.d. noise
        Z = rng.normal(0, 1, size=num)

        # Serially correlated errors for X and Y
        eps_x = make_error(corr_const, num)
        eps_y = make_error(corr_const, num)

        # Treatment X and outcome Y
        # X depends on Z plus correlated error; include intercept in the Y model
        X = 0.5 * Z + eps_x
        Y = 1.0 + beta * X + 0.3 * Z + eps_y

        # OLS: Y ~ const + X + Z (with *non-robust* SEs on purpose)
        Xmat = sm.add_constant(np.column_stack([X, Z]))
        model = sm.OLS(Y, Xmat).fit()

        # store beta_hat on X and its (conventional) standard error
        betas.append(model.params[1])  # coef on X
        ses.append(model.bse[1])       # SE of coef on X

    betas = np.array(betas)
    ses = np.array(ses)
    return betas.std(ddof=1), ses.mean()

for c in [0.2, 0.5, 0.8]:
    sd_beta, mean_se = run_sim(c)
    print(f"corr_const={c}:  (i) SD(beta_hat)={sd_beta:.4f}   (ii) mean SE={mean_se:.4f}   ratio i/ii={sd_beta/mean_se:.2f}")
