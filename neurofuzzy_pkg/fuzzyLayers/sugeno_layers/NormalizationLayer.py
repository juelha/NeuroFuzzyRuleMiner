# basics
import tensorflow as tf
import numpy as np

# custom
import neurofuzzy_pkg.utils.MFs as MFs


class NormalizationLayer():
    """
    The NormalizationLayer()-Class is:
    - mapping the rule strengths calculated in RuleAntecedentLayer to the MFs of the output

    THEN-Neuron for the MFs Low, and High:
                            
    Π(μ_L(x_1),μ_L(x_2)) \ 
                          \ 
    Π(μ_L(x_1),μ_M(x_2)) -- N()
                          / 
    Π(μ_L(x_1),μ_H(x_2)) /
            .
            .
            .
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
        self.built = False

        # for training
        self.tunable = True 
        self.inputs = []
        self.outputs = []
        

    def build(self, inputs):
        """Initializes trainable parameters

        Args:
            inputs (tf.Tensor): inputs
        """
                
        # build weights and biases
        self.weights = np.ones(inputs.shape, dtype=np.float32)


        self.train_params = {'weights': self.weights}
        self.built = True

        # call self
        return self(inputs)


    def __call__(self, inputs):
        """Calculating the MF degree of the IF-Part of a rule
        
        Args:
            inputs (tf.Tensor): rule strengths from RuleAntecedentLayer
        
        Returns:
            (tuple): can be unzipped into: 
                - out (tensor): membership degree of rules to output membership functions,
                                row per output-MFs, column per rule, has shape=(n_mfs, n_rules)
                - layer (RuleConsequentLayer): needed for defuzzification, 
                                               since we need the number of MFs and the centers of the MFs 
        """
        self.inputs = inputs

     #   print("in", inputs)
        out  = tf.TensorArray(tf.float32, size=0, dynamic_size=True)# []

        # for rule extraction
        self.rulesTHEN = {}

        # build "weights" parameters once 
        if not self.built: self.build(inputs)

        sum_inputs = tf.math.reduce_sum(inputs)

        # iterate over inputs (here Tnorms)
        for i,x in enumerate(inputs):

            x /= sum_inputs
            # activation saved to output to be defuzzified in next layer
            out = out.write(out.size(), x)
            

        out = out.stack()
        self.outputs = out # saved for training 

       # print("out norm", out)

        return out 

    











