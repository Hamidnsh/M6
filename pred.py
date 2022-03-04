import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import lightgbm

np.random.seed(0)


df_history = pd.read_csv("full_asset_m6_history.csv")
df_history["date"] = pd.to_datetime(df_history["date"])
df_future = pd.read_csv("full_asset_m6_future.csv")
df_future["date"] = pd.to_datetime(df_future["date"])

df_history["symbol"] = df_history["symbol"].astype("category")
df_future["symbol"] = df_future["symbol"].astype("category")

X_cols = [
    "symbol",
    "dayofyear",
    "week",
    "month",
    "year",
    "quarter",
    "shift_close",
    "shift_high",
    "shift_low",
    "shift_open",
    "shift_volume",
]
y_col = "price"

# load models
save_models = []
for i in range(3):
    save_models.append(lightgbm.Booster(model_file=f"./models/model{i}.txt"))

with open("./models/weights.npy", "rb") as f:
    weights = np.load(f)

# Make predictions for future
pred = np.array([0] * len(df_future))
for i in range(len(save_models)):
    pred = pred + weights[i] * save_models[i].predict(df_future[X_cols])
df_future[y_col] = pred

# Graph each asset history and future
def graph_asset(n):
    df = df_history[df_history["symbol"] == n]
    df_f = df_future[df_future["symbol"] == n]
    plt.title(n)
    plt.plot(df["date"], df["price"])
    plt.plot(df_f["date"], df_f["price"])
    plt.savefig(f"figs/{n}.png")
    plt.clf()


for x in df_history["symbol"].unique():
    graph_asset(x)
