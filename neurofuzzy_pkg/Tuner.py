# basics
from numpy import dtype, float32
import tensorflow as tf
import numpy as np
import pandas as pd
from itertools import product

# custom
from model_pkg.Trainer import Trainer
#from neurofuzzy_pkg.fuzzyLayers import MF_gaussian_prime_a
from neurofuzzy_pkg import utils
from neurofuzzy_pkg.utils.MFs import MF_gaussian,MF_gaussian_prime_a, MF_gaussian_prime_b
from neurofuzzy_pkg.utils.MFs import MF_tri, MF_tri_prime_a, MF_tri_prime_b
import neurofuzzy_pkg.utils.MFs as MFs

from sklearn.model_selection import RepeatedKFold

from tqdm import tqdm 
import os
import yaml

class Tuner():   
    """
    - tunes hyper parameters
    ___________________________________________________________
    """

    def __init__(self, model):
        """Initializing neurofuzzyTrainer by inheriting from Trainer
        """
        self.arc = model.arc
        self.__call__(model)

    def __call__(self, model):
        """Running the training loop and saving the MFs before and after 
        
        Args:
            train_ds (PrefetchDataset): dataset for training
            test_ds (PrefetchDataset): dataset for testing
        """
        search_dict = self.create_search_space()
        search_dict = self.grid_search(model, search_dict)
        self.save_experiment(model, search_dict)

        
        
    def create_search_space(self, hparams=None):
        """
        create a hyper-parameter grid to sample from during fitting
        
        explanation of hyperparams:
            constraint_center = how much centers are allowed to move during training (over range)
                                p >= maxs.iloc[i] - (constraint_center* (maxs.iloc[i] - mins.iloc[i])): 
            constraint_width = how big widths are allowed to get during training
            learning_rate = learning rate for adapting with gradient descent
            n_epochs = number of training epochs

        Returns: 
            grid (dict): hyper-parameter grid with form {'name of hyperparam': [discrete values or a distribution of values]}
        """
        # Look at parameters used by our current forest
        # print('Parameters currently in use:\n')
        # pprint(rf.get_params())

        search_dict = {}

        search_dict['constraint_center'] = [1/x for x in np.arange(start = 0, stop = 11, step = 1)] 
        search_dict['constraint_width'] = [x for x in np.arange(start = 0, stop = 11, step = 1)] 
        search_dict['learning_rate'] = [0.01, 0.1, 1.0]
        search_dict['n_epochs'] = [int(x) for x in np.arange(start = 1, stop = 11, step = 1)] 

       # search_dict['MF'] = [True,False]
       # search_dict['n_fuzzy_labels'] = [int(x) for x in np.linspace(start = 0, stop = 1500, num = 100)]
       # search_dict['T-Norm'] = [True,False]
       # search_dict['ratio_train_test'] = ['cyclic', 'random']
       # search_dict['error_func'] = [True, False]

        return search_dict

    def grid_search(self, model, search_dict, avg_epochs=5):
        """
        simple grid search on search space 

         Args: 
            arc: to tune 
            avg_epochs (int): epochs to train and then average over 
            input (numpy.ndarray): 
            target (numpy.ndarray):

        Returns: 
            best_hparams 

        """
        all_accs = []
        model.data.load_data_for_building()
        model.data.load_data_for_training() 
        
        # set up dict
        new_dict  = {key: [] for key in search_dict.keys()}
        new_dict['accuracy'] = []

        # Iterate over all possible combinations of hyperparameters
        for params in  product(*search_dict.values()):

            # save 
            for i, key in enumerate(search_dict):
                new_dict[key].append(params[i])


            a,b,c,d = params # hc

            # train with those hyperparams
            accs = []
            for _ in range(avg_epochs):
                model.train(constraint_center=a, constraint_width=b, learning_rate=c, n_epochs=d )
                # get accuracy 
                acc = self.get_class_accuracy(model.data.inputs, model.data.targets)
                accs.append(acc)

            new_dict['accuracy'].append(np.mean(accs))

        print(new_dict)
        new_dict = pd.DataFrame(new_dict)
        return new_dict


            
    def save_experiment(self, model, search_dict):
        save_path = os.path.dirname(__file__) +  f'/../results/{model.data.df_name}'
        file_name = f"{model.data.df_name}_hp.csv"
        completeName = os.path.join(save_path, file_name)

       # df = pd.DataFrame(search_dict)
        search_dict.to_csv(completeName)

        # save best hps
        l = search_dict['accuracy']
        idx_min = np.argmin(l)
        search_dict.pop('accuracy') # del for saving only params
        best_hps = search_dict.iloc(idx_min)

        # save
        file_name = f"hyperparams.yml"
        relative_path = f'/../config/{model.data.df_name}'
        save_path = os.path.dirname(__file__) +  relative_path
        completeName = os.path.join(save_path, file_name)

        with open(completeName, 'w') as yaml_file:
            yaml.dump(best_hps, yaml_file, default_flow_style=False)

        
        
    def get_class(self, input_vec, df_name=None): 
        # propagating through network
        outputs = self.arc(input_vec)
        outputs = np.sum(outputs, axis=1) # make 1d
        idx_max = np.argmax(outputs)
        classID = self.arc.RuleConsequentLayer.class_weights[idx_max]
        return classID

    def get_class_accuracy(self, inputs, targets, df_name =None):
        acc = []
        for input_vec, target_vec in (zip(inputs, targets)):
            classID = self.get_class(input_vec) 
            acc.append(classID == target_vec)
        return np.mean(acc)

    def rs_hparams(self, arc, train_input, train_target):
        """
        random search on random_dict

        use of RandomizedSearchCV()
        Args: 
            arc to tune 
            random_grid: filled with random values for hyperparas of arc

        Returns: 
            best_hparams 
        """
        # Use the random grid to search for best hyperparameters
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

        scoring = 'neg_root_mean_squared_error' 

        random_hparam_dict = self.create_search_space()

        # Random search of parameters, using pre-defined fold cross validation, 
        # search across 100 different combinations, and use all available cores
        # Fit the random search model
        rf_random = RandomizedSearchCV(estimator = arc, param_distributions = random_hparam_dict, 
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
