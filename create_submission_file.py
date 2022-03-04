# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 16:19:16 2022

@author: snourashrafeddin
"""

import pandas as pd
import numpy as np 
import lightgbm
from lightgbm import LGBMRegressor
np.random.seed(0)

starting_date = "2015-01-01"
future_starting_date = pd.to_datetime("2022-03-06")
future_end_date = pd.to_datetime("2022-04-01")
forcast_horizon_days = 20

df_history = pd.read_csv('full_asset_m6_history.csv')
df_history['date'] = pd.to_datetime(df_history['date'])
df_future = pd.read_csv('full_asset_m6_future.csv')
df_future['date'] = pd.to_datetime(df_future['date'])

df_history['symbol'] = df_history['symbol'] .astype('category')
df_future['symbol'] = df_future['symbol'] .astype('category')

X_cols = ['symbol', 'dayofyear', 'dayofweek', 'week', 'month', 'year', 'quarter', 'shift_close', 'shift_high', 'shift_low', 'shift_open', 'shift_volume']
y_col = 'price'


models = []
for i in range(3):
    models.append(lightgbm.Booster(model_file=f'./models/model{i}.txt'))

with open('./models/weights.npy', 'rb') as f:
    weights = np.load(f)

preds = np.array([0]*len(df_future))
for i in range(len(models)):
    preds = preds + weights[i]*models[i].predict(df_future[X_cols])
    
df_future['price'] = preds


last_day_history_df = df_history.loc[df_history['date'] == df_history['date'].max(), ['symbol', 'price']]
last_day_history_df.reset_index(drop=True, inplace=True)
last_day_history_df.rename(columns={"price":"last_price"}, inplace=True)


last_day_future_df = df_future.loc[df_future['date'] == df_future['date'].max(), ['symbol', 'price']]
last_day_future_df.reset_index(drop=True, inplace=True)
last_day_future_df.rename(columns={"price":"new_price"}, inplace=True)

df_submission = pd.merge(last_day_history_df, last_day_future_df, on='symbol')
df_submission.rename(columns={'symbol':'ID'}, inplace=True)
df_submission['percentage_return'] = df_submission['new_price'] - df_submission['last_price']
df_submission['percentage_return'] = df_submission['percentage_return'] / df_submission['last_price']
df_submission['unsigned_percentage_return'] = np.abs(df_submission['percentage_return'])
_sum = np.nansum(df_submission['unsigned_percentage_return'].values)
df_submission['Decision'] = df_submission['percentage_return'] / _sum
df_submission['Decision'] = np.round(df_submission['Decision'], 4)
df_submission['Decision'] = df_submission['Decision'].fillna(0)

df_submission['percentage_return'] = df_submission['percentage_return'].fillna(df_submission['percentage_return'].min())

bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
percentiles = []
for i in range(0, len(bins)):
    percentiles.append(df_submission['percentage_return'].quantile(bins[i]))
percentiles[-1] += 0.5
for i in range(1, len(percentiles)):
    df_submission[f'Rank{i}'] = np.where((df_submission['percentage_return'].values >= percentiles[i-1]) &\
                                         (df_submission['percentage_return'].values < percentiles[i]), 1, 0)

cols = ['ID', 'Rank1', 'Rank2', 'Rank3', 'Rank4', 'Rank5', 'Decision']

df_submission[cols].to_csv('submission.csv', index=False)




