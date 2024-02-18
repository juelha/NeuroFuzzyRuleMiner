# basics
import tensorflow as tf
import numpy as np

# custom
from neurofuzzy_pkg.fuzzyLayers.FuzzificationLayer import FuzzificationLayer


class RuleAntecedentLayer():
    """
    The RuleAntecedentLayer()-Class is:
    - combine the membership values from FuzzificationLayer

    IF-Neuron:

    μ_L(x_1) 
            \ 
            Π = T(μ_L(x_1), μ_L(x_2))
            /
    μ_L(x_2)
    ___________________________________________________________
    """

    def __init__(self, n_mfs, n_features):
        """Initializes RuleAntecedentLayer()-Object

        Attributes:
            tunable (boolean): if parameters of layers can be tuned during training
            rulesIF (dict): used for rule extraction (IF-Part)
        """
        # for training
        self.built = False
        self.tunable = False
        self.train_params = None 

        # for rule extraction
        self.rulesIF = {}   
        self.n_rules = 0     
        self.n_features = n_features 
        self.n_mfs = n_mfs 


    def build(self, inputs):
        """Initializes trainable parameters

        Args:
            inputs (tf.Tensor): inputs
        """
        # n_rows, n_cols = inputs.shape

        # # build weights which will be used to weight the inputs <- importance of participant in rule?
        self.weights = np.ones((self.n_mfs**self.n_participants, self.n_participants), dtype=np.float32)

        # self.train_params = {'weights': self.weights}
        # self.built = True

        # call self
        
        return self(inputs)



    def __call__(self, x):
        """Combines the fuzzified inputs to form rules and calculates its firing strength 
        -> fuzzy intersection

        Args:
            inputs (tf.Tensor): fuzzified inputs, 
                                shape=(n_inputs, n_mfs), dtype=float32
        
        Returns:
            TNorms (tf.Tensor): tensor containing the firing strength of the rules, 
                                shape=(n_rules,), dtype=tf.float32
        """

        # check if trainable params have been built
       # assert self.built, f'Layer {type(self)} is not built yet'

       # self.inputs = x # saved for training

    
      #  print("honk", self.n_mfs)
        self.n_rules = int(self.n_mfs**self.n_features)

        # x = np.array_split(x, range(self.n_mfs, len(x), self.n_mfs))


       
        x = np.array_split(x, range(self.n_mfs, len(x), self.n_mfs)) # hc
       # x.reverse()  # so it fits with convention 
        x = np.array(np.meshgrid(*x,indexing='ij')) # the '*' unpacks x and passes to messgrid
        self.inputs = x #  need meshgrid for training


      #  x = (x[0] * x[1]).ravel()
        x = np.prod(x, axis=0).ravel()

       # x = x *np.prod(self.weights, axis=1)
  

        assert self.n_rules == x.size, f'the number of rules generated: {x.size} has to equal: {self.n_rules} -> n_mfs ** n_participants' 

        return x
