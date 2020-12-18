import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor


# reading in data, cleaning column names, initial filtering

df = pd.read_csv('Sales Forecaster/historical_sales.csv')

rename_cols = {'Item Description: Product Name':'Product', 'Item Description: Country':'Country',
'Item Description: Product Family':'Family', 'Item Description: Size':'Size', 'Shipping State/Province':'Shipping State',
'Item Description: MSRP / Bottle':'MSRP', 'Item Description: Percent Alcohol':'% Alc'}
df = df.rename(columns=rename_cols)

df = df[(df['Cases Sold'] > 0) & (df['MSRP'] > 0) & (df['Sample'] == 'N') & (df['Warehouse'] != 'DSW')]

df = df.dropna()


# adding in time-series columns, filtering out current month in progress

df['Delivery Date'] = pd.to_datetime(df['Delivery Date']).dt.to_period('M')
df['Year'] = df['Delivery Date'].dt.year
df['Month'] = df['Delivery Date'].dt.month

df = df[df['Delivery Date'] < pd.Timestamp.today().to_period('M')]


# selecting main columns to allow for quick iterations

cols = ['Delivery Date', 'Year', 'Month', 'Brand', 'Family', 'Size']


# tranforming data into monthly sales, grouping by above columns

monthly = (pd.pivot_table(df,
            index=cols,
            values=['Cases Sold'],
            aggfunc=np.sum)
            .sort_values(['Delivery Date', 'Family', 'Size']).reset_index())


# adding in previous time period data columns

monthly['Last Month'] = monthly.groupby(['Family', 'Brand', 'Size'])['Cases Sold'].shift()
monthly['2 Months Ago'] = monthly.groupby(['Family', 'Brand', 'Size'])['Cases Sold'].shift(2)
monthly['3 Months Ago'] = monthly.groupby(['Family', 'Brand', 'Size'])['Cases Sold'].shift(3)
monthly['4 Months Ago'] = monthly.groupby(['Family', 'Brand', 'Size'])['Cases Sold'].shift(4)
monthly['5 Months Ago'] = monthly.groupby(['Family', 'Brand', 'Size'])['Cases Sold'].shift(5)
monthly['6 Months Ago'] = monthly.groupby(['Family', 'Brand', 'Size'])['Cases Sold'].shift(6)
monthly['Last Year'] = monthly.groupby(['Month', 'Family', 'Brand', 'Size'])['Cases Sold'].shift()


# building sliding window validation to evaluate last X time periods fitted against previous historical data

def sliding_windows(df, date_col, target, dtypes, num_windows):
    windows = {}

    for i in range(num_windows, 0, -1):

        target_month = (pd.Timestamp.today() - pd.DateOffset(months=(i))).to_period('M')
        window = df[df[date_col] == target_month].select_dtypes(include=dtypes).dropna()
        prior = df[df[date_col] < target_month].select_dtypes(include=dtypes).dropna()

        X_train, y_train = prior.drop(target, axis=1), prior[target]
        X_test, y_test = window.drop(target, axis=1), window[target]

        windows[i] = {'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test}

    return windows


months = sliding_windows(monthly, 'Delivery Date', 'Cases Sold', ['int64', 'float64'], 6)


# list of ML models to test

knn = KNeighborsRegressor()
lr = LinearRegression()
rfr = RandomForestRegressor()
mlp = MLPRegressor()
xgb = XGBRegressor()

models = {'K-Nearest Neighbors':knn, 'Linear Regression':lr, 'Random Forest Regressor':rfr, 'Neural Network':mlp, 'XG Boost':xgb}


# predictor currently set for current month: iterating through models and sliding window validation 

predictions = {}

for model in models:

    predictions[model] = {}

    for month in months:

        predictions[model][month] = {}
        
        models[model].fit(months[month]['X_train'], months[month]['y_train'])
        predictions[model][month]['Predictions'] = models[model].predict(months[month]['X_test'])
        predictions[model][month]['Mean Absolute Error'] = mean_absolute_error(months[month]['y_test'], predictions[model][month]['Predictions'])
        predictions[model][month]['Mean Squared Error'] = mean_squared_error(months[month]['y_test'], predictions[model][month]['Predictions'])
        predictions[model][month]['R2 Score'] = r2_score(months[month]['y_test'], predictions[model][month]['Predictions'])


# scores for predictor by X windows selected

for model in predictions:
    print(model)
    print('-'*30)

    for month in predictions[model]:
        print('Window: Current Month Minus', month)
        print('-'*10)
        print('MAE:', predictions[model][month]['Mean Absolute Error'])
        print('MSE:', predictions[model][month]['Mean Squared Error'])
        print('R2:', predictions[model][month]['R2 Score'])
        print('\n')


# baseline score that model must outperform: previous month's sales

for month in months:
    print('Baseline Estimator')
    print('-'*10)
    print('Window: Current Month Minus', month)
    print('-'*10)
    print('MAE:', mean_absolute_error(months[month]['y_test'], months[month]['X_test']['Last Month']))
    print('MSE:', mean_squared_error(months[month]['y_test'], months[month]['X_test']['Last Month']))
    print('R2:', r2_score(months[month]['y_test'], months[month]['X_test']['Last Month']))
    print('\n')