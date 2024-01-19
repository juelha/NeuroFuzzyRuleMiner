# basics
import tensorflow as tf
import pandas as pd
import os.path
from numpy import nan
import numpy as np
import yaml 
from yaml.loader import UnsafeLoader
import os

# custom
from model_pkg import *
from neurofuzzy_pkg import * 


class WeightArchiver():
    """ saving and loading weights from yaml files
    """

    def __init__(self) -> None:
        pass



    def save_weights(self, weights_name, df_name):
        """saves weights to yaml file
        
        Args:
            weights_name (str): name of weights for file name
            df_name (str): name of dataframe that weights have been built on
        
        Raises:
            AssertionError: Save path for weights could not be found. 
        """
        # opt 1: yaml
        relative_path = f"/weights/{df_name}"
        save_path = os.path.dirname(__file__) + relative_path
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            print(f'Directory {df_name} created') 

        assert  os.path.exists(save_path), f'save_path {save_path} not found'

        file_name = weights_name + ".yaml"
        full_path = os.path.join(save_path, file_name)
        with open(full_path, 'w') as yaml_file:
            yaml.dump(self.weights.tolist(), yaml_file, default_flow_style=False)

        # opt 2: np.save
        file_name = weights_name
        other_name = os.path.join(save_path, file_name)
        np.save(other_name, self.weights)
    

    def load_weights(self, weights_name, df_name):
        """load weights from yaml file
        
        Args:
            weights_name (str): name of weights for file name
            df_name (str): name of dataframe that weights have been built on
            
        Returns:
            weights (numpy.ndarray): 

        Raises:
            AssertionError: Save path for weights could not be found. 
        """
        # opt 1: yaml
        file_name = weights_name + ".yaml"
        relative_path =  f"/weights/{df_name}"
        save_path = os.path.dirname(__file__) +  relative_path
        full_path = os.path.join(save_path, file_name)
        assert  os.path.exists(full_path), f'File {file_name} not found'
        with open(full_path, 'r') as config_file:
            # Converts yaml document to python object
            config =yaml.load(config_file, Loader=UnsafeLoader)
            weights = np.array(config)
            print(type(weights))
            print(weights)
        
        # opt 2: np.save
    #     file_name = "config_weights"
    #     other_name = os.path.join(save_path, file_name)
    #     loaded_weights = np.load(other_name+'.npy')
    #     print("sucessfully loaded weights")
    #    # print(loaded_weights)
    #    # return weights
        
        self.weights = weights
        self.train_params = {'weights': self.weights}
       
        self.built = True
        return weights