# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 10:50:51 2022

@author: snourashrafeddin
"""
import pandas as pd
import numpy as np

starting_date = "2015-01-01"
future_starting_date = pd.to_datetime("2022-03-06")
future_end_date = pd.to_datetime("2022-04-01")
forcast_horizon_days = 20 # excluding weekends

target = 'price'

df = pd.read_csv('full_asset_m6.csv')
df['date'] = pd.to_datetime(df['date'])

def generate_features(df, date_col, key_col, roll_cols, shift_size=forcast_horizon_days):
    df['dayofyear'] = df[date_col].dt.dayofyear
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['week'] = df[date_col].dt.week
    df['month'] = df[date_col].dt.month
    df['year'] = df[date_col].dt.year
    df['quarter'] = df[date_col].dt.quarter
    for col in roll_cols:
      df['shift_'+col] = df[[key_col, col]].groupby(by=[key_col])[col].transform(lambda x: x.shift(shift_size).fillna(0)) 
    
    return df 

df = generate_features(df, date_col='date', key_col='symbol', roll_cols=['close','high','low','open','volume'])

df_history = df.loc[df.date < future_starting_date ].copy()
df_future = df.loc[df.date >= future_starting_date ].copy()
df_history.to_csv('full_asset_m6_history.csv', index=False)
df_future.to_csv('full_asset_m6_future.csv', index=False)