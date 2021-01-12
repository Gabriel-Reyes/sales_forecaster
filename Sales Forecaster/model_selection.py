import pandas as pd
import numpy as np

from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data_processing import active, monthly, next_month, two_months_out, three_months_out


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

selected_models = ['Linear Regression']

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