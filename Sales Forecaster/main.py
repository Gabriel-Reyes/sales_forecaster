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
monthly['Last Month Delta YoY'] = monthly['Last Month'] - monthly.groupby(['Family', 'Brand', 'Size'])['Cases Sold'].shift(13)
monthly['2 Months Ago'] = monthly.groupby(['Family', 'Brand', 'Size'])['Cases Sold'].shift(2)
monthly['3 Months Ago'] = monthly.groupby(['Family', 'Brand', 'Size'])['Cases Sold'].shift(3)
monthly['4 Months Ago'] = monthly.groupby(['Family', 'Brand', 'Size'])['Cases Sold'].shift(4)
monthly['5 Months Ago'] = monthly.groupby(['Family', 'Brand', 'Size'])['Cases Sold'].shift(5)
monthly['6 Months Ago'] = monthly.groupby(['Family', 'Brand', 'Size'])['Cases Sold'].shift(6)
monthly['7 Months Ago'] = monthly.groupby(['Family', 'Brand', 'Size'])['Cases Sold'].shift(7)
monthly['8 Months Ago'] = monthly.groupby(['Family', 'Brand', 'Size'])['Cases Sold'].shift(8)
monthly['9 Months Ago'] = monthly.groupby(['Family', 'Brand', 'Size'])['Cases Sold'].shift(9)
monthly['10 Months Ago'] = monthly.groupby(['Family', 'Brand', 'Size'])['Cases Sold'].shift(10)
monthly['11 Months Ago'] = monthly.groupby(['Family', 'Brand', 'Size'])['Cases Sold'].shift(11)
monthly['Last Year'] = monthly.groupby(['Month', 'Family', 'Brand', 'Size'])['Cases Sold'].shift()


# creating dataframes for predicting X months out

next_month = monthly.drop(columns=['Last Month', 'Last Month Delta YoY', 'Last Year'])
next_month = next_month.rename(columns={'11 Months Ago':'Last Year'})

two_months_out = next_month.drop(columns=['2 Months Ago', 'Last Year'])
two_months_out = two_months_out.rename(columns={'10 Months Ago':'Last Year'})

three_months_out = two_months_out.drop(columns=['3 Months Ago', 'Last Year'])
three_months_out = three_months_out.rename(columns={'9 Months Ago':'Last Year'})


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


months = sliding_windows(next_month, 'Delivery Date', 'Cases Sold', ['int64', 'float64'], 6)


# list of ML models to test with parameters for tuning

knn = KNeighborsRegressor()
lr = LinearRegression()
rfr = RandomForestRegressor()
mlp = MLPRegressor()
xgb = XGBRegressor()

model_list = {'K-Nearest Neighbors':{'model':knn, 'parameters':{'n_neighbors': [65, 75, 85],
                                                            'weights': ['uniform'],
                                                            'algorithm': ['auto'],
                                                            'leaf_size': [45],
                                                            'p': [2],
                                                            'n_jobs': [-1]}},

         'Linear Regression':{'model':lr, 'parameters':{'fit_intercept': [True],
                                                         'normalize': [False],
                                                         'n_jobs': [None]}},

        'Random Forest Regressor':{'model':rfr, 'parameters':{'n_estimators': [200],
                                                                'criterion': ['mse'],
                                                                'max_depth': [None],
                                                                'min_samples_split': [2],
                                                                'min_samples_leaf': [2],
                                                                'min_weight_fraction_leaf': [0.0],
                                                                'max_features': ['auto'],
                                                                'max_leaf_nodes': [None],
                                                                'min_impurity_decrease': [0.0],
                                                                'min_impurity_split': [None],
                                                                'bootstrap': [True],
                                                                'oob_score': [False],
                                                                'n_jobs': [-1],
                                                                'random_state': [None],
                                                                'verbose': [0],
                                                                'warm_start': [False],
                                                                'ccp_alpha': [0.0],
                                                                'max_samples': [None]}},
                                                                
        'Neural Network':{'model':mlp, 'parameters':{'hidden_layer_sizes': [(100,)], # best: (50,50,50)
                                                        'activation': ['relu'],
                                                        'solver': ['adam'],
                                                        'alpha': [0.0001],
                                                        'batch_size': ['auto'],
                                                        'learning_rate': ['constant'],
                                                        'learning_rate_init': [0.001],
                                                        'power_t': [0.5],
                                                        'max_iter': [200],
                                                        'shuffle': [True],
                                                        'random_state': [None],
                                                        'tol': [1e-4],
                                                        'warm_start': [False],
                                                        'momentum': [0.9],
                                                        'nesterovs_momentum': [True],
                                                        'early_stopping': [False],
                                                        'validation_fraction': [0.1],
                                                        'beta_1': [0.9],
                                                        'beta_2': [0.999],
                                                        'epsilon': [1e-08],
                                                        'n_iter_no_change': [10],
                                                        'max_fun': [15000]}},
        
        'XG Boost':{'model':xgb, 'parameters':{'n_estimators': [200],
                                                'max_depth': [8],
                                                'learning_rate': [0.3],
                                                'verbosity': [1],
                                                'booster': ['gbtree'],
                                                'tree_method': ['exact'],
                                                'n_jobs': [0],
                                                'gamma': [0],
                                                'min_child_weight': [1],
                                                'max_delta_step': [0],
                                                'subsample': [1],
                                                'colsample_bytree': [1],
                                                'colsample_bylevel': [1],
                                                'colsample_bynode': [1],
                                                'reg_alpha': [0],
                                                'reg_lambda': [1],
                                                'scale_pos_weight': [1],
                                                'base_score': [0.5],
                                                'random_state': [0],
                                                'num_parallel_tree': [1],
                                                'importance_type': ['gain']}}}

selected_models = ['K-Nearest Neighbors']

models = dict((key, model_list[key]) for key in selected_models)


# hyperparameter optimization using GridSearchCV or RandomizedSearchCV

param_results = {}

for model in models:
    clf = RandomizedSearchCV(models[model]['model'], models[model]['parameters']) #component to change search function
    clf.fit(months[1]['X_train'], months[1]['y_train'])
    models[model]['model'] = clone(clf.best_estimator_)
    
    param_results[model] = pd.DataFrame(clf.cv_results_).sort_values('rank_test_score')
    print(model, '-', 'Best Parameters:')
    print('-'*40)
    print(models[model]['model'].get_params())
    print('\n')


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