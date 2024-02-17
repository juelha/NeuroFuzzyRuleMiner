# basics
import tensorflow as tf
import pandas as pd

# custom
from neurofuzzy_pkg.fuzzyLayers import *
import neurofuzzy_pkg.utils.MFs as MFs


from tqdm import tqdm 

import yaml 
from yaml.loader import UnsafeLoader
import os

class MyArc():
    """Architecture of the neuro fuzzy neural network
    """

    def __init__(self):
        """Initializes MyArc()-Object

        Attributes: 
            total_params (int): number of total params in network
            trainable_params (int): number of params that can be trained 
            FuzzificationLayer (FuzzificationLayer()): fuzzifies input
            RuleAntecedentLayer (RuleAntecedentLayer()): combines fuzzified input (IF-Part)
            RuleConsequentLayer (RuleConsequentLayer()): get value of combinations to output mfs (Then-Part)
            DefuzzificationLayer (DefuzzificationLayer()): get crip output
        """

        self.Name = "MyArc"
        self.total_params = 0
        self.trainable_params = 0

        self.FuzzificationLayer = FuzzificationLayer()
        self.RuleAntecedentLayer = RuleAntecedentLayer()
        self.RuleConsequentLayer = RuleConsequentLayer()

        self.internal_layers = [
            self.FuzzificationLayer,
            self.RuleAntecedentLayer,
            self.RuleConsequentLayer,
        ]



    def __call__(self, x):
        """Forward propagating the inputs through the network

        Args: 
            x (tf.Tensor):  crisp input (1 row of the dataset)

        Returns: 
            x (tf.Tensor): final output
        """
        for layer in self.internal_layers:
            x = layer(x)
        return x        

   

        
    def get_params(self):
        """Get stats of params of network
        """
        
        self.trainable_params = 0

        for layer in self.internal_layers:
         #   print(layer)
            if hasattr(layer,'train_params'):
               # print(layer.built)
                assert layer.built == True, f'The layer {type(layer)} has not been built yet'
                for param in layer.train_params:
          #          print(param)
           #         print(layer.train_params[param].size)
                    self.trainable_params += layer.train_params[param].size
    
    def __str__(self) -> str:
        return "MyArc"

        
    def summary(self):
        # get params
        self.get_params()

        column_names = ['Total layer params', 'Trainable layer params', 'Non-trainable layer params']
        d = [self.total_params, self.trainable_params, self.total_params-self.trainable_params]

        df = pd.DataFrame(d, column_names)
        return df