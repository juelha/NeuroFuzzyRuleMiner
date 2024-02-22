# basics
import tensorflow as tf
import numpy as np

# custom
from neurofuzzy_pkg.fuzzyLayers.FuzzificationLayer import FuzzificationLayer
from neurofuzzy_pkg.utils.math_funcs import coefficient


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

    def __init__(self, ):
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
        self.n_participants = 2 # two participants in a rule


    def build(self, inputs):
        """Initializes trainable parameters

        Args:
            inputs (tf.Tensor): inputs
        """
        # n_rows, n_cols = inputs.shape

        # # build weights which will be used to weight the inputs <- importance of participant in rule?
        # self.weights = np.ones((n_rows, n_cols), dtype=np.float32)

        # self.train_params = {'weights': self.weights}
        # self.built = True

        # call self
        return self(inputs)



    def __call__(self, inputs):
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

        self.inputs = inputs # saved for training

        # to output
        TNorms = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        # for rule extraction
        n_inputs, n_mfs = inputs.shape
        self.rulesIF = {}     
        ruleID = 0
        
        # picking first participant of a rule 
        # by looping over rows of input 
        for xID1 in range(n_inputs):
            for mfID1 in range(n_mfs):
                mu1 = inputs[xID1][mfID1]
                
                # weighting the input
                weighted_mu1 =  mu1 #* self.weights[xID1][mfID1] *

                # get second participant
                # by looping over the rest of rows
                for xID2 in range(xID1+1, n_inputs):
                    for mfID2 in range(n_mfs):  
                        mu2 = inputs[xID2][mfID2]

                        # weighting the input
                        weighted_mu2 =  mu2 # self.weights[xID2][mfID2] *

                        # calculating TNorm with min() or mul
                        # TNorm = min(weighted_mu1,weighted_mu2)
                        TNorm = weighted_mu1 * weighted_mu2
                        TNorms = TNorms.write(TNorms.size(),TNorm)
                        
                        # adding information to ruleIF dict
                        self.rulesIF[ruleID] = []
                        self.rulesIF[ruleID].append({'xID': xID1, 'mfID': mfID1})
                        self.rulesIF[ruleID].append({'xID': xID2, 'mfID': mfID2})
                        ruleID += 1

        # validate shape of output
        n = n_mfs * n_inputs
        k = self.n_participants 
        self.n_rules = int(coefficient(n, k) - n)

      
        # print("in", inputs)
        # print("n", n)
        # print("k", k)
        # print("n rules", self.n_rules)
       # print("ruleID",ruleID)

        assert self.n_rules == ruleID, f'the number of rules generated in IF-Part: {ruleID} has to equal: {self.n_rules} -> coefficient(n_cols * n_rows, k) - n_cols * n_rows' 

        return TNorms.stack()



    

