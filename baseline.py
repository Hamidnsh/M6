# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 11:08:09 2022

@author: snourashrafeddin
"""

import pandas as pd
import numpy as np
import optuna
import seaborn as sns
from optuna.samplers import TPESampler
from matplotlib import pyplot as plt
from lightgbm import LGBMRegressor
np.random.seed(0)
sampler = TPESampler(seed=0)

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

df_train = df_history.loc[df_history['date'] <= future_starting_date - pd.DateOffset(days=forcast_horizon_days)].copy() 
df_test = df_history.loc[df_history['date'] > future_starting_date - pd.DateOffset(days=forcast_horizon_days)].copy()

X_cols = ['symbol', 'dayofyear', 'week', 'month', 'year', 'quarter', 'shift_close', 'shift_high', 'shift_low', 'shift_open', 'shift_volume']
y_col = 'price'

def train_model(df_train, k=3, n_trails=20, es=20):
    studies = []
    
    def LGBM_objective(trial):
        param = {
            'boosting_type':trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-5, 10.0),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-5, 10.0),
            'learning_rate': trial.suggest_float('learning_rate', 1e-10, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 4, 32), 
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0),
            'min_child_samples': 10,
        }
       
        model = LGBMRegressor(**param)
        model.fit(df_train_fold[X_cols], 
                  df_train_fold[y_col], 
                  eval_set=[(df_eval_fold[X_cols], 
                             df_eval_fold[y_col])], 
                  early_stopping_rounds=es) 
        
        pred = model.predict(df_eval_fold[X_cols])
        score = np.mean(np.abs(df_eval_fold[y_col] - pred))
        return score
    
    last_index = int(len(df_train)*0.8)
    eval_size = int(len(df_train)*0.2)
    for i in range(k):
        df_train_fold = df_train.iloc[0:last_index]  
        if i == 0:
            df_eval_fold = df_train.iloc[last_index:]
        else:
            df_eval_fold = df_train.iloc[last_index:last_index+eval_size]
        last_index -= eval_size
          
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(LGBM_objective, n_trials=n_trails)
        studies.append(study)
   
    
    return studies

studies = train_model(df_train)


models = []
vals = []
for study in studies:
    model = LGBMRegressor(**study.best_trial.params)
    model.fit(df_train[X_cols], df_train[y_col])
    models.append(model)
    vals.append(study.best_trial.value)
    
model_weights = 1/np.array(vals)
model_weights = model_weights/np.sum(model_weights)
model_weights

pred = np.array([0] * len(df_test))
for i in range(len(models)):
    pred = pred + model_weights[i]*models[i].predict(df_test[X_cols])

    
truth = df_test[y_col].values

bias_val = np.divide(np.sum(truth), np.sum(pred)) - 1
print("bias: " + str(bias_val))


errors = np.sum(np.abs(pred - truth))
wmape = errors / np.sum(truth)
print("wmape: " + str(wmape))

Accuracy = (1 - wmape)*100
print("Accuracy: " + str(Accuracy))

models = []
for study in studies:
    model = LGBMRegressor(**study.best_trial.params)
    model.fit(df_history[X_cols], df_history[y_col])
    models.append(model)

for i in range(len(models)):
    models[i].booster_.save_model(f'./models/model{i}.txt')
    

with open('./models/weights.npy', 'wb') as f:
    np.save(f, model_weights)
    
# load models 
# save_models = []
# for i in range(len(models)):
#     save_models.append(lightgbm.Booster(model_file=f'./models/model{i}.txt'))

# with open('./models/weights.npy', 'rb') as f:
#     weights = np.load(f)
