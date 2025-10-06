import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# === Load dataset ===
# For example: replace with your actual filename ('homework_4.2.a.csv' or 'homework_4.2.b.csv')
df = pd.read_csv("homework_4.2.a.csv")  # columns: X, Y

# === Define cutoff and filter range around it ===
cutoff = 80
window = 10  # look at scores from 70–90
df = df[(df["X"] >= cutoff - window) & (df["X"] <= cutoff + window)]

# === Fit logistic regression model ===
model = LogisticRegression()
X = df[["X"]]
y = df["Y"]
model.fit(X, y)

# === Predict probabilities over the same range ===
x_range = np.linspace(df["X"].min(), df["X"].max(), 200).reshape(-1, 1)
y_pred = model.predict_proba(x_range)[:, 1]

# === Compute binned averages of Y for smoother plot ===
bin_width = 1
bins = np.arange(df["X"].min(), df["X"].max() + bin_width, bin_width)
df["X_bin"] = pd.cut(df["X"], bins=bins)
bin_means = df.groupby("X_bin", observed=False)[["X", "Y"]].mean().dropna()

# === Plot ===
plt.figure(figsize=(8, 5))
plt.scatter(bin_means["X"], bin_means["Y"], color="blue", label="Observed avg Y (binned)", alpha=0.7)
plt.plot(x_range, y_pred, color="red", linewidth=2, label="Predicted prob (logistic)")
plt.axvline(x=cutoff, color="black", linestyle="--", label="Cutoff = 80")

plt.title("College Admission Probability vs. Test Score (around cutoff = 80)")
plt.xlabel("Test Score (X)")
plt.ylabel("Admission Probability (Y)")
plt.legend()
plt.tight_layout()
plt.show()
