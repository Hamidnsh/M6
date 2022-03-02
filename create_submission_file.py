# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 16:19:16 2022

@author: snourashrafeddin
"""

import pandas as pd
import numpy as np 
import lightgbm
from lightgbm import LGBMRegressor

starting_date = "2015-01-01"
future_starting_date = pd.to_datetime("2022-03-06")
future_end_date = pd.to_datetime("2022-04-01")
forcast_horizon_days = (future_end_date - future_starting_date).days + 1

df_history = pd.read_csv('full_asset_m6_history.csv')
df_history['date'] = pd.to_datetime(df_history['date'])
df_future = pd.read_csv('full_asset_m6_future.csv')
df_future['date'] = pd.to_datetime(df_future['date'])

df_history['symbol'] = df_history['symbol'] .astype('category')
df_future['symbol'] = df_future['symbol'] .astype('category')

models = []
for i in range(len(models)):
    models.append(lightgbm.Booster(model_file=f'./models/model{i}.txt'))

with open('./models/weights.npy', 'rb') as f:
    weights = np.load(f)
