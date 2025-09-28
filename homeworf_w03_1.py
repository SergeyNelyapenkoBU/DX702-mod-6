import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Загружаем данные
df = pd.read_csv("homework_3.1.csv")

# Создаём индикатор события и взаимодействие
cutoff = 50
df["D"] = (df["time"] >= cutoff).astype(int)
df["time_x_D"] = df["time"] * df["D"]

results = {}

for col in ["value1", "value2", "value3"]:
    y = df[col]

    # --- Модель с разрывом в уровне ---
    X1 = sm.add_constant(df[["time", "D"]])
    model1 = sm.OLS(y, X1).fit()

    # --- Модель с разрывом в уровне и наклоне ---
    X2 = sm.add_constant(df[["time", "D", "time_x_D"]])
    model2 = sm.OLS(y, X2).fit()

    # Сохраняем результаты
    results[col] = {
        "jump_coef": model2.params.get("D", np.nan),
        "jump_pval": model2.pvalues.get("D", np.nan),
        "slope_change_coef": model2.params.get("time_x_D", np.nan),
        "slope_change_pval": model2.pvalues.get("time_x_D", np.nan),
        "R2_level": model1.rsquared,
        "R2_slope": model2.rsquared
    }

    # --- График ---
    plt.figure(figsize=(7,4))
    plt.scatter(df["time"], y, alpha=0.6, label="data")

    # Предсказания модели 2
    pred = model2.predict(X2)
    plt.plot(df["time"], pred, color="red", label="fitted (with slope change)")

    # Вертикальная линия в точке события
    plt.axvline(x=cutoff, color="black", linestyle="--", label="event @ 50")

    plt.title(f"{col}: jump={results[col]['jump_coef']:.3f} (p={results[col]['jump_pval']:.3g}), "
              f"slopeΔ={results[col]['slope_change_coef']:.3f} (p={results[col]['slope_change_pval']:.3g})")
    plt.xlabel("time")
    plt.ylabel(col)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Итоговая таблица результатов
results_df = pd.DataFrame(results).T
print("\n=== Итоги по переменным ===")
print(results_df.round(4))
