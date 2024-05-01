## basic imports ##
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import math

# numpy imports
from numpy import load
from numpy import loadtxt
from numpy import nan
from numpy import isnan
from numpy import count_nonzero
from numpy import unique
from numpy import array

from sklearn.base import clone
# linear
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import LassoLars
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import SGDRegressor
# non linear
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor

# for optimization
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from pprint import pprint
import yaml 
from sklearn.model_selection import TimeSeriesSplit

# input selection
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SequentialFeatureSelector

	
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())/np.std(predictions)



def select_features_linear(model, train_input, train_target):
    """
     
    """
    ## plot prediction vs real data
    # x_line = np.arange(0, test_target.size)
    # plt.plot(x_line, test_target, prediction)
    # plt.legend(('real', 'predicted'))
    # plt.title(f'target: {train_target.name}')
    # plt.show()
    ## evaluate inputs

    
    #importance = np.abs(model.coef_)
    feature_names = np.array(train_input.keys())
    k = 10

    #Feature importance from coefficients
    # plt.bar(height=importance, x=feature_names)
    # plt.title(f'target: {train_target.name}')
    # plt.show()
    # # Selecting features based on coefficients
    # k = 6
    # idx = np.argpartition(importance, -k)
    # pruned_list = feature_names[idx[-k:]]

    #Selecting features based on importance
    # threshold = np.sort(importance)[-k] 
    # sfm = SelectFromModel(model, prefit=True)
    # pruned_list = sfm.get_feature_names_out(feature_names)

    # RFE
    # selector =  RFE(model) 
    # selector = selector.fit(train_input, train_target)
    # pruned_list = selector.get_feature_names_out(feature_names)

    ## Selecting features with Sequential Feature Selection (uses  cross-validation)
    selector = SequentialFeatureSelector( 
    	model, n_features_to_select='auto', tol=0.001, direction="backward", scoring='neg_root_mean_squared_error', cv=TimeSeriesSplit())
    selector = selector.fit(train_input, train_target)
    pruned_list = selector.get_feature_names_out(feature_names)

    # save
    dict_to_save = {"pruned_list": pruned_list.tolist()}
    file_name = f"config_{train_target.name}.yml"
    relative_path = "/hyperparams"
    save_path = os.path.dirname(__file__) +  relative_path
    completeName = os.path.join(save_path, file_name)

    with open(completeName, 'w') as yaml_file:
        yaml.dump(dict_to_save, yaml_file, default_flow_style=False)


    print(f"Features selected by SelectFromModel: {pruned_list}")
    return pruned_list


## data pruning
def select_features_forest(model, train_input, train_target):
    """
    Args:
        model (sklearn-model-obj): fitted model 
    """
    ## forests
    ## get importance of each input per target
    # importances = fitted_model.feature_importances_
    # std = np.std([tree.feature_importances_ for tree in fitted_model.estimators_], axis=0)
    # # plot importance
    # forest_importances = pd.Series(importances, index = test_input.keys())
    # fig, ax = plt.subplots()
    # forest_importances.plot.bar(yerr=std, ax=ax)
    # ax.set_title(f"Feature importances using MDI for target {test_target.name} ")
    # ax.set_ylabel("Mean decrease in impurity")
    # fig.tight_layout()
    # plt.show()

    # forests
    feature_names = np.array(train_input.keys())
    importance = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    # Selecting features based on coefficients
    k = 10
    idx = np.argpartition(importance, -k)
    pruned_list = feature_names[idx[-k:]].tolist()


    # plot importance
    forest_importances = pd.Series(importance, index = train_input.keys())
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title(f"Feature importances using MDI for target {train_target.name} ")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()
    print(f"Features selected by SelectFromModel: {pruned_list}")

    # save
    dict_to_save = {"pruned_list": pruned_list}
    file_name = f"config_{train_target.name}.yml"
    relative_path = "/hyperparams"
    save_path = os.path.dirname(__file__) +  relative_path
    completeName = os.path.join(save_path, file_name)

    with open(completeName, 'w') as yaml_file:
        yaml.dump(dict_to_save, yaml_file, default_flow_style=False)

    return pruned_list





## hyper paramters tuning 
def create_search_space(hparams=None):
    """
    get hyper paras based on model TODO 
    create a random hyper-parameter grid to sample from during fitting
    
    explanation of hyperparams:
        n_estimators = number of trees in the forest
        max_features = max number of features considered for splitting a node
        max_depth = max number of levels in each decision tree
        min_samples_split = min number of data points placed in a node before the node is split
        min_samples_leaf = min number of data points allowed in a leaf node
        bootstrap = method for sampling data points (with or without replacement)

    Returns: random_grid
    with form {name of hyperparam: [discrete values or a distribution of values]}
    """
    # Look at parameters used by our current forest
    # print('Parameters currently in use:\n')
    # pprint(rf.get_params())

    random_dict = {}

#     random_dict['alpha'] = [int(x) for x in np.arange(start = 0, stop = 20, step = 1)] 
#     random_dict['fit_intercept'] = [True,False]
#     random_dict['l1_ratio'] = [int(x) for x in np.arange(start = 0, stop = 1, step = 0.1)] 
#   #  random_dict['normalize'] = [True, False]
#     random_dict['selection'] = ['cyclic', 'random']

    random_dict["alpha"] = [int(x) for x in np.arange(start = 1, stop = 20, step = 1)] 
    random_dict['fit_intercept'] = [True,False]
    random_dict['max_iter'] = [int(x) for x in np.linspace(start = 0, stop = 1500, num = 100)]
    random_dict['positive'] = [True,False]
    random_dict['selection'] = ['cyclic', 'random']
    random_dict['warm_start'] = [True, False]
    random_dict['tol'] = [0.0001, 0.001, 0.01, 0.1]
    
    # random_dict['C'] = [int(x) for x in np.arange(start = 0, stop = 20, step = 1)] 
#     random_dict['fit_intercept'] = [True,False]
#     random_dict['shuffle'] = [False]
#     random_dict['max_iter'] = [int(x) for x in np.linspace(start = 0, stop = 1000, num = 100)]
#    # random_dict['l1_ratio'] = [int(x) for x in np.arange(start = 0, stop = 1, step = 0.1)] 
#     random_dict['early_stopping'] = [True, False]
   # random_dict['selection'] = ['cyclic', 'random']


#    # random_dict['criterion'] = ["squared_error", "absolute_error", "friedman_mse", "poisson"]
#     random_dict['n_estimators'] = [int(x) for x in np.linspace(start = 0, stop = 500, num = 50)]
#     random_dict['max_features'] = [int(x) for x in np.arange(start = 0, stop = 20, step = 1)] 
#     random_dict['max_depth'] = [int(x) for x in np.linspace(start = 30, stop = 70, num = 5)]
#    # random_dict['max_depth'] = random_dict['max_depth'].append(None)
#     random_dict['min_samples_split'] = [int(x) for x in np.arange(start = 60, stop = 100, step = 2)] 
#     random_dict['min_samples_leaf'] = [int(x) for x in np.arange(start = 0, stop = 20, step = 1)]
#     random_dict['bootstrap'] = [True, False]
    return random_dict




def rs_hparams(model, train_input, train_target):
    """
    random search on random_dict

    use of RandomizedSearchCV()
    Args: 
        model to tune 
        random_grid: filled with random values for hyperparas of model

    Returns: 
        best_hparams 
    """
    # Use the random grid to search for best hyperparameters
    #cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    cv = TimeSeriesSplit()
    scoring = 'neg_root_mean_squared_error' # TODO check this is correct rmse #  which score to improve

    # random search part -> sampling randomly from a distribution
    random_hparam_dict = create_search_space()

    # Random search of parameters, using pre-defined fold cross validation, 
    # search across 100 different combinations, and use all available cores
    # Fit the random search model
    rf_random = RandomizedSearchCV(estimator = model, param_distributions = random_hparam_dict, 
                n_iter = 100, cv = cv, verbose=2, random_state=42, n_jobs = -1, scoring=scoring)
    rf_random.fit(train_input, train_target)

    best_hparams = rf_random.best_params_
    pprint(best_hparams)


    # save
    file_name = f"{train_target.name}_params_lasso_not_scaled.yml"
    relative_path = "/hyperparams"
    save_path = os.path.dirname(__file__) +  relative_path
    completeName = os.path.join(save_path, file_name)

    with open(completeName, 'w') as yaml_file:
        yaml.dump(best_hparams, yaml_file, default_flow_style=False)

    
    return best_hparams



def gs_hparams(model, random_dict, cv,scoring, train_input, train_target):
	"""
	grid search 
	GridSearchCV 
	"""
	grid_search = GridSearchCV(estimator = model, param_grid = random_dict, 
                          cv = cv, n_jobs = -1, verbose = 2, scoring=scoring)
	grid_search.fit(train_input, train_target)
	best_hparams = grid_search.best_params_
	return best_hparams


def rs_gs_hparams(model, train_input, train_target):
    """ 
    tune hyperparameters of random forest -esque models 
        - random search on search space
        - feed result of random search (as ranges) to grid search

    Returns:
        best_hparams (dict)
    """
    # define cv obj
    # define evaluation
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    scoring = rmse() #  which score to improve

    # random search part -> sampling randomly from a distribution
    random_hparam_dict = create_search_space()
    best_hparams = rs_hparams(model, random_hparam_dict, cv, scoring, train_input, train_target) 
    pprint(best_hparams)

    # feed a smaller range of values to grid search.
    # grid search part -> evaluates all combinations we define
    # TODO automatically Create the parameter grid based on the results of random search 
    random_hparam_dict = create_search_space(best_hparams)
    best_hparams = rs_hparams(model, random_hparam_dict, cv, scoring, train_input, train_target) 
    pprint(best_hparams)

    # save best


    # return tuned_model 
    return best_hparams


def get_tuned_model(model, best_hparams):
    """
    get model that is tuned based on a best_hparams dict 

    Returns:
        tuned_model
    """

    # for param_name in best_hparams:
    #         model.param_name = best_hparams
    model = model.set_params(**best_hparams)
  #  print('Parameters currently in use:\n')
   # pprint(model.get_params())
    return model




def run_model(model_name, fitted_models_targets, eval_input, targets_dict):

	predictions = {}
	for target_name in targets_dict.keys():
		fitted_model = fitted_models_targets[target_name][model_name] 
		predictions[target_name] =  fitted_model.predict(eval_input) 		

	predictions =  pd.DataFrame.from_dict(predictions)
	print("predictions")


####
# inputs_dict, targets_dict = prep_data()
# eval_all_tars  =  iter_all_tars(inputs_dict, targets_dict)
# eval_all_tars = pd.DataFrame.from_dict(eval_all_tars)
# print(eval_all_tars)
