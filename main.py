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

df = pd.read_csv('historical_sales.csv')

rename_cols = {'Item Description: Product Name':'Product', 'Item Description: Country':'Country',
'Item Description: Product Family':'Family', 'Item Description: Size':'Size', 'Shipping State/Province':'Shipping State',
'Item Description: MSRP / Bottle':'MSRP', 'Item Description: Percent Alcohol':'% Alc'}
df = df.rename(columns=rename_cols)

df = df[(df['Cases Sold'] > 0) & (df['MSRP'] > 0) & (df['Sample'] == 'N') & (df['Warehouse'] != 'DSW')]

df = df.dropna()


# adding in time-series columns, filtering out current month in progress

df['Delivery Date'] = pd.to_datetime(df['Delivery Date'])
df['Year'] = df['Delivery Date'].dt.year
df['Month'] = df['Delivery Date'].dt.month

df = df[df['Delivery Date'] < dt.datetime(year=pd.Timestamp.today().year, month=pd.Timestamp.today().month, day=1)]


# selecting main columns to allow for quick iterations

cols = ['Year', 'Month', 'Brand', 'Family', 'Size']


# tranforming data into monthly sales, grouping by above columns

monthly = (pd.pivot_table(df,
            index=cols,
            values=['Cases Sold'],
            aggfunc=np.sum)
            .sort_values(['Year', 'Month', 'Family', 'Size']).reset_index())


# adding in previous time period data columns

monthly['Last Month'] = monthly.groupby(['Family', 'Brand', 'Size'])['Cases Sold'].shift()
monthly['2 Months Ago'] = monthly.groupby(['Family', 'Brand', 'Size'])['Cases Sold'].shift(2)
monthly['3 Months Ago'] = monthly.groupby(['Family', 'Brand', 'Size'])['Cases Sold'].shift(3)
monthly['4 Months Ago'] = monthly.groupby(['Family', 'Brand', 'Size'])['Cases Sold'].shift(4)
monthly['5 Months Ago'] = monthly.groupby(['Family', 'Brand', 'Size'])['Cases Sold'].shift(5)
monthly['6 Months Ago'] = monthly.groupby(['Family', 'Brand', 'Size'])['Cases Sold'].shift(6)
monthly['Last Year'] = monthly.groupby(['Month', 'Family', 'Brand', 'Size'])['Cases Sold'].shift()


# list of ML models to test

knn = KNeighborsRegressor()
lr = LinearRegression()
rfr = RandomForestRegressor()
mlp = MLPRegressor()
xgb = XGBRegressor()

models = {'K-Nearest Neighbors':knn, 'Linear Regression':lr, 'Random Forest Regressor':rfr, 'Neural Network':mlp, 'XG Boost':xgb}


# finalizing model data, excluding object columns

model_data = monthly.select_dtypes(exclude=['object']).dropna()

X = model_data.drop('Cases Sold', axis=1)
y = model_data['Cases Sold']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=3)

predictions = pd.DataFrame()


# ML model evaluation

for model in models:
    models[model].fit(X_train, y_train)
    predictions[model] = models[model].predict(X_test)
    mae = mean_absolute_error(y_test, predictions[model])
    mse = mean_squared_error(y_test, predictions[model])
    r2 = r2_score(y_test, predictions[model])
    print(model, '\n', 'MAE:', mae, 'MSE:', mse, 'r2:', r2)


# coef grid

features = pd.DataFrame()
features['feature'] = X.columns
features['lr coef'] = lr.coef_
features['rfr importance'] = rfr.feature_importances_
features['xgb importance'] = xgb.feature_importances_


# graphing predictions vs actual, focusing on core sales volume range

predictions['actual'] = y_test.reset_index(drop=True)

sns.pairplot(predictions[predictions['actual'] < 500], y_vars=['actual'], diag_kind=None)

plt.show()