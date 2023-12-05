# basics
import tensorflow as tf
import numpy as np

# custom
import neurofuzzy_pkg.utils.MFs as MFs


class RuleConsequentLayerSug():
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
        self.mf_type = mf_type
        self.built = False

        # for training
        self.tunable = True 
        self.inputs = []
        self.outputs = []
        
        # for rule extraction
        self.rulesTHEN = {}


    def build(self, inputs, inputs_og ):
        """Initializes trainable parameters

        Args:
            inputs (tf.Tensor): inputs

        Attr:
            weights (np): shape=(n_inputs, n_inputs_og)
        """
        #inputs, inputs_og = inputs_inputs_og_zip

      #  print("inputs_inputs_og_zip", inputs_inputs_og_zip)
        
        
        # build weights     
        self.weights = np.ones((inputs.shape[0], inputs_og.shape[0] ), dtype=np.float32)

        # build biases
        self.biases = np.full(inputs.shape[0], 0.5, dtype=np.float32)

        self.train_params = {'weights': self.weights, 'biases': self.biases}
        self.built = True

        # call self
        return self(inputs, inputs_og)


    def __call__(self, inputs, inputs_og):
        """Combines the fuzzified inputs to form rules and calculates its firing strength 
        -> fuzzy intersection 

        formula
        w(p_i* x_1 + q_i * x_2 + r)

        Args:
            inputs (tf.Tensor): fuzzified inputs 
        
        Returns:
            TNorms (tf.Tensor): tensor containing the firing strength of the rules, 
            shape=(n_rules,), dtype=tf.float32
        """
        self.inputs = inputs

        out  = tf.TensorArray(tf.float32, size=0, dynamic_size=True)# []

        # for rule extraction
        self.rulesTHEN = {}

        self.rulesTHEN[0] = []
        self.rulesTHEN[1] = []

        # build "weights" parameters once 
        if not self.built: self.build(inputs)

        ruleID = 0      
        # iterate over i    nputs (here Tnor    ms) 
        for xID, x in enumerate(inputs):     

            # iterate over og inputs    
            for x_ogID, x_og in enumerate(inputs_og): 
                act =  tf.reduce_sum(tf.multiply(self.weights[xID][x_ogID], x_og ))
                act += self.biases[xID]
            
            # how to add input ??? 
            act *= x
                          

            out = out.write(out.size(),act)

            ruleID += 1

        out = out.stack()       
        self.outputs = out #     saved for training   

      #  print("out", out)  

        return out # returning layer as well for defuzzication  

    











