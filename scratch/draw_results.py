import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt

data = pd.read_csv("../data/accuracy.csv")
print(len(data))
data["value_error_mean"] = np.sqrt(data["value_error_mean"])
fig = plt.figure(figsize=(30, 20))
ax = fig.subplots(3, 2)
sns.lineplot(
    data=data.loc[data["fitted"], :],
    x="fraction",
    y="value_error_mean",
    hue="method",
    ax=ax[0, 0],
)
sns.lineplot(
    data=data.loc[data["fitted"], :],
    x="n_outcomes",
    y="value_error_mean",
    hue="method",
    ax=ax[0, 1],
)
sns.lineplot(
    data=data.loc[data["fitted"], :],
    x="fraction",
    y="ranking_error",
    hue="method",
    ax=ax[1, 0],
)
sns.lineplot(
    data=data.loc[data["fitted"], :],
    x="n_outcomes",
    y="ranking_error",
    hue="method",
    ax=ax[1, 1],
)
sns.lineplot(data=data, x="fraction", y="fitted", hue="method", ax=ax[2, 0])
sns.lineplot(data=data, x="n_outcomes", y="fitted", hue="method", ax=ax[2, 1])
plt.show()
print(len(data))
print(
    data.groupby(["method"])["ranking_error"].apply(lambda x: len(x.dropna()) / len(x))
)
