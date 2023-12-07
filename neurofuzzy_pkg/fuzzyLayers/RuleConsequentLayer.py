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

    def __init__(self, mf_type=MFs.MF_gaussian, n_mfs=2):
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
        self.built = False
        self.threshold = 0.5 # hc
        self.weights = None # the weights that assign a rule to a class 

        # for training
        self.tunable = False 
        self.inputs = []
        self.outputs = []
        
        # for rule extraction
        self.rulesTHEN = {}

    def save_weights(self, dataset_name=None):
        """saves weights to yaml file
        
        Args:
            dataset_name (str): name of datasets weights have been built on
        """
        # save
        # opt 1: yaml
        file_name = f"config_weights.yaml"
        relative_path = "\weights"
        save_path = os.path.dirname(__file__) +  relative_path
        completeName = os.path.join(save_path, file_name)
        with open(completeName, 'w') as yaml_file:
            yaml.dump(self.weights.tolist(), yaml_file, default_flow_style=False)

        # opt 2: np.save
        file_name = "config_weights"
        other_name = os.path.join(save_path, file_name)
        np.save(other_name, self.weights)
    
    def load_weights(self):
        """load weights from yaml file
        
        Args:
            filename etc
        Returns:
            loaded_weights (numpy.ndarray): 
        """
        # opt 1: yaml
        file_name = f"config_weights.yaml"
        relative_path =  "\weights"
        save_path = os.path.dirname(__file__) +  relative_path
        completeName = os.path.join(save_path, file_name)
        with open(completeName, 'r') as config_file:
            # Converts yaml document to python object
            config =yaml.load(config_file, Loader=UnsafeLoader)
            weights = np.array(config)
          #  print(type(weights))
          # print(weights)
        
        # opt 2: np.save
        file_name = "config_weights"
        other_name = os.path.join(save_path, file_name)
        loaded_weights = np.load(other_name+'.npy')
        print("sucessfully loaded weights")
       # print(loaded_weights)
       # return weights
        self.train_params = {'weights': self.weights}
        self.weights = loaded_weights
       
        self.built = True
        return loaded_weights


    def build(self, inputs, one_hot_tar):
        """Initializes trainable parameters

        Args:
            inputs (tf.Tensor): inputs

        Attr:
            weights (np): shape=(n_inputs, n_inputs_og)
        """
        #inputs, inputs_og = inputs_inputs_og_zip

      #  print("inputs_inputs_og_zip", inputs_inputs_og_zip)
        
        load =0
        
        if load==0:
            # # build weights     
            self.weights = np.zeros((inputs.shape[0], 2), dtype=np.float32) # danger output classes hc 
        # print("weights", self.class_weights)
            for ruleID, firingStrength in enumerate(inputs):  
                if firingStrength > self.threshold:
                    self.weights[ruleID] = one_hot_tar


        if load:
            self.weights = self.load_weights()
      #  print("weights after", self.class_weights)

        # self.weights = np.ones((inputs.shape[0], inputs_og.shape[0] ), dtype=np.float32)

        # # build biases
        # self.biases = np.full(inputs.shape[0], 0.5, dtype=np.float32)

      #  self.train_params = {'weights': self.weights}#, 'biases': self.biases}
        self.built = True

        # call self
        return self(inputs)


    def __call__(self, inputs):
        """Assigns rules per firing strength to a class 

        Args:
            inputs (tf.Tensor): fuzzified inputs 
        
        Returns:
            TNorms (tf.Tensor): tensor containing the normed firing strength of the rules, 
            shape=(n_rules,), dtype=tf.float32
        """
        # check if built
        assert self.built, f'Layer {type(self)} is not built yet'

        self.inputs = inputs

        out  = tf.TensorArray(tf.float32, size=0, dynamic_size=True)# []

        # for rule extraction
        self.rulesTHEN = {}

        ruleID = 0      
   
        # iterate over i    nputs (here rules  
        for ruleID, x in enumerate(inputs):     


            #       0.8 * [0 1] = [0 0.8]
            output = x * self.weights[ruleID]

            self.rulesTHEN[ruleID] = []
            self.rulesTHEN[ruleID].append({'RS': x, 'target': self.weights})             

            out = out.write(out.size(), output)

            ruleID += 1

        out = out.stack()       
        self.outputs = out #     saved for training   

       # print("out", out)  
      #
      #   self.save_weights()
       # self.load_weights()
        return out # returning layer as well for defuzzication  

    