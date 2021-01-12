import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data_processing import monthly, next_month, two_months_out, three_months_out
from model_selection import models, months


# predictor currently set for current month: iterating through models and sliding window validation 

predictions = {}

for model in models:

    predictions[model] = {}

    for month in months:

        predictions[model][month] = {}
        
        models[model]['model'].fit(months[month]['X_train'], months[month]['y_train'])
        predictions[model][month]['Predictions'] = models[model]['model'].predict(months[month]['X_test'])
        predictions[model][month]['Mean Absolute Error'] = mean_absolute_error(months[month]['y_test'], predictions[model][month]['Predictions'])
        predictions[model][month]['Mean Squared Error'] = mean_squared_error(months[month]['y_test'], predictions[model][month]['Predictions'])
        predictions[model][month]['R2 Score'] = r2_score(months[month]['y_test'], predictions[model][month]['Predictions'])


# scores for predictor by X windows selected

for model in predictions:
    print(model, '-', 'Sliding Window Scores:')
    print('-'*30)

    for month in predictions[model]:
        print('Window: Current Month Minus', month)
        print('-'*10)
        print('MAE:', predictions[model][month]['Mean Absolute Error'])
        print('MSE:', predictions[model][month]['Mean Squared Error'])
        print('R2:', predictions[model][month]['R2 Score'])
        print('\n')


# baseline estimator that model must outperform: previous month's sales

for month in months:
    most_recent_sales_data = '2 Months Ago'
    print('Baseline Estimator')
    print('-'*10)
    print('Window: Current Month Minus', month)
    print('-'*10)
    print('MAE:', mean_absolute_error(months[month]['y_test'], months[month]['X_test'][most_recent_sales_data]))
    print('MSE:', mean_squared_error(months[month]['y_test'], months[month]['X_test'][most_recent_sales_data]))
    print('R2:', r2_score(months[month]['y_test'], months[month]['X_test'][most_recent_sales_data]))
    print('\n')