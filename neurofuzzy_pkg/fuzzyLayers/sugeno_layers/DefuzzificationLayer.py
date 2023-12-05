# basics
import tensorflow as tf
import numpy as np

class DefuzzificationLayer():
    """
    The DefuzzificationLayer()-Class is:
    - mapping the membership degrees of the fuzzy sets to a crisp output 

    Defuzzification-Neuron for MF_Low:

    μ_L(μ_i(x_1,x_2)) 
                      \ 
    μ_L(μ_i(x_1,x_3)) - y_L 
    ...               /
    μ_L(μ_i(x_n,x_(n-1)))
    ___________________________________________________________
    """

    def __init__(self):
        """Initializes DefuzzificationLayer()-Object

        Attributes: 
            tunable (boolean): if parameters of layers can be tuned during training
            built (boolean): if tunable parameters have been built yet
        """
        # for training
        self.tunable = True
        self.built = False
        self.inputs = None 
        self.outputs = None
        self.weights = None
        self.train_params = None 


    def build(self, inputs):
        """Initializes trainable parameters

        Args:
            input_shape (tuple): shape of inputs
        """
        self.weights = np.ones(inputs.shape, dtype=np.float32)
        self.train_params = {'weights':self.weights}
        self.built = True


    def __call__(self, inputs):
        """Calculates the crisp output from the degrees of MFs from RuleConsequentLayer()

        Args:
            zip_inputs_layer (tuple): can be unzipped into: 
                - inputs (tf.Tensor): membership degree of rules to output membership functions, has shape=(n_rules,)
                - layer (RuleConsequentLayer()): layer that comes before, needed for defuzzification, 
                                               since we need the number of Mfs and the centers of the MFs 

        Returns:
            outputs (tf.Tensor): the crisp outputs of the model, since this is a binary classfication: shape=(2,)
        
        """
        # unzipping

        # save for training
        self.inputs = inputs

        # build parameters once 
        if not self.built: self.build(tf.shape(inputs))

        # calculate crisp outputs 
        outputs  = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        # one hot manually todo make better 
        # 
        act = tf.reduce_sum(inputs)
        if act >= 0:
            outputs =  tf.convert_to_tensor([0,act])
           # self.rulesTHEN[1].append(ruleID)

        else:
            outputs =  tf.convert_to_tensor([act,0])
         #   self.rulesTHEN[0].append(ruleID)
        #crisp_out = tf.reduce_sum(inputs)
        #  outputs = outputs.write(outputs.size(), inputs)

           


      #  print("out", outputs)

        return outputs





    

