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


# building sliding window validation to evaluate last 3 time periods fitted against previous historical data

def x_months_ago(df, date_col, x_months_ago):
    lower_bound = (pd.Timestamp.today() - pd.DateOffset(months=(x_months_ago + 1))).to_period('M')
    upper_bound = (pd.Timestamp.today() - pd.DateOffset(months=(x_months_ago - 1))).to_period('M')
    month = df[(df[date_col] > lower_bound) & (df[date_col] < upper_bound)]
    return month.dropna()

train = monthly[monthly['Delivery Date'] < (pd.Timestamp.today() - pd.DateOffset(months=3)).to_period('M')]

train = train.select_dtypes(include=['int64', 'float64']).dropna()

X_train = train.drop('Cases Sold', axis=1)
y_train = train['Cases Sold']

sep_2020 = x_months_ago(monthly, 'Delivery Date', 3)
oct_2020 = x_months_ago(monthly, 'Delivery Date', 2)
nov_2020 = x_months_ago(monthly, 'Delivery Date', 1)

sep_2020_X = sep_2020.select_dtypes(include=['int64', 'float64']).drop('Cases Sold', axis=1)
sep_2020_y = sep_2020['Cases Sold']

oct_2020_X = oct_2020.select_dtypes(include=['int64', 'float64']).drop('Cases Sold', axis=1)
oct_2020_y = oct_2020['Cases Sold']

nov_2020_X = nov_2020.select_dtypes(include=['int64', 'float64']).drop('Cases Sold', axis=1)
nov_2020_y = nov_2020['Cases Sold']

months = {'3 Months Ago':[sep_2020_X, sep_2020_y], '2 Months Ago':[oct_2020_X, oct_2020_y], '1 Month Ago':[nov_2020_X, nov_2020_y]}


# list of ML models to test

knn = KNeighborsRegressor()
lr = LinearRegression()
rfr = RandomForestRegressor()
mlp = MLPRegressor()
xgb = XGBRegressor()

models = {'K-Nearest Neighbors':knn, 'Linear Regression':lr, 'Random Forest Regressor':rfr, 'Neural Network':mlp, 'XG Boost':xgb}


# finalizing model data, excluding object columns

predictions = {}

for model in models:
    predictions[model] = {}
    models[model].fit(X_train, y_train)

    for month in months:
        predictions[model][month] = {}
        predictions[model][month]['Predictions'] = models[model].predict(months[month][0])
        predictions[model][month]['Mean Absolute Error'] = mean_absolute_error(months[month][1], predictions[model][month]['Predictions'])
        predictions[model][month]['Mean Squared Error'] = mean_squared_error(months[month][1], predictions[model][month]['Predictions'])
        predictions[model][month]['R2 Score'] = r2_score(months[month][1], predictions[model][month]['Predictions'])