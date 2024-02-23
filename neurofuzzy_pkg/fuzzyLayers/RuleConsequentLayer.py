# basics
import tensorflow as tf
import numpy as np
import yaml 
from yaml.loader import UnsafeLoader
import os

# custom
import neurofuzzy_pkg.utils.MFs as MFs


class RuleConsequentLayer():
    """
    The RuleConsequentLayer()-Class is:
    - mapping the rule strengths calculated in RuleAntecedentLayer to the MFs of the output

    THEN-Neuron for the MFs Low, and High:

    x_1 
        \ 
    N_1 -- THEN 
        /
    x_2              
    ___________________________________________________________
    """

    def __init__(self, n_classes):
        """Initializes RuleConsequentLayer()-Object
        
        Args:
            n_mfs (int): number of MFs, hardcoded as 2 -> binary classification
            mf_type (callable): type of MFs

        Attributes:
            tunable (boolean): if parameters of layers can be tuned during training
            inputs (list): inputs of layer
            outputs (list): outputs of layer 
            rulesTHEN (dict): used for rule extraction (THEN-Part)
            threshold (float): defines wether a rule will be counted to MF_low or MF_high
        """

        # for MFs
        self.dictrules = {}
        self.tars = {}
        self.built = False
        self.threshold = 0.1 # hc
        self.class_weights = None # the weights that assign a rule to a class 

        # for training
        self.tunable = False 
        self.inputs = []
        self.outputs = []
        self.n_classes = n_classes
        
        # for rule extraction
        self.rulesTHEN = {}


    def build(self, inputs, one_hot_tar):
        """Initializes trainable parameters

        Args:
            inputs (tf.Tensor): inputs

        Attr:
            weights (np): shape=(n_inputs, n_inputs_og)
        """
   
        self.class_weights = np.zeros((inputs.shape[0], self.n_classes), dtype=np.float32) # danger output classes hc 


        for ruleID, firingStrength in enumerate(inputs):  
            self.dictrules[ruleID].append(firingStrength)
            self.tars[ruleID].append(one_hot_tar)

        self.built = True

        # call self
        return self(inputs)


    def __call__(self, x):
        """Assigns rules per firing strength to a class 

        Args:
            inputs (tf.Tensor): fuzzified inputs 
        
        Returns:
            TNorms (tf.Tensor): tensor containing the normed firing strength of the rules, 
            shape=(n_rules,), dtype=tf.float32
        """

        self.inputs = x
        x = x[:, np.newaxis] * self.class_weights
        return x # returning layer as well for defuzzication  

    