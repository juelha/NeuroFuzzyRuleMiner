# basics
import tensorflow as tf
import numpy as np

class DefuzzificationLayerMam():
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
        self.tunable = False
        self.built = False
        self.inputs = None 
        self.outputs = None
        self.weights = None
        self.train_params = None 


    def build(self, zip_inputs_layer):
        """Initializes trainable parameters

        Args:
            inputs (tf.Tensor): inputs
        """

        # for some reason unzipping does work here but if you print its the right thing? 
     #   print("hereeeee", zip_inputs_layer[0]) 
        # unzipping
      #  inputs, layer = zip_inputs_layer
        inputs = zip_inputs_layer[0]
        

        #print(inputs)
        # build weights
        self.weights = np.ones(inputs.shape, dtype=np.float32)

        self.train_params = {'weights': self.weights}
        self.built = True

        # call self
        return self(zip_inputs_layer)


    def __call__(self, zip_inputs_layer):
        """Calculates the crisp output from the degrees of MFs from RuleConsequentLayer()

        Args:
            zip_inputs_layer (tuple): can be unzipped into: 
                - inputs (tf.Tensor): membership degree of rules to output membership functions, has shape=(n_rules,)
                - layer (RuleConsequentLayer()): layer that comes before, needed for defuzzification, 
                                               since we need the centers of the MFs 

        Returns:
            outputs (tf.Tensor): the crisp outputs of the model, since this is a binary classfication: shape=(2,)
        
        """
        

        # check if trainable params have been built
        assert self.built, f'Layer {type(self)} is not built yet'

        # unzipping
        inputs, layer = zip_inputs_layer

        self.inputs = inputs # saved for training 

        # calculate crisp outputs 
        outputs  = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        n_rows, n_cols = inputs.shape
    
        # for mfID in range(n_cols): 
        #     rules_per_mf = tf.gather(inputs, mfID, axis=1)
        #     weights_per_mf = tf.gather(self.weights, mfID, axis=1)

        #     # defuzzify
        #     crisp_out = tf.reduce_sum(tf.multiply(rules_per_mf,weights_per_mf))/ tf.reduce_sum(rules_per_mf)
            
        #     # write to TensorArray
        #     outputs = outputs.write(outputs.size(), crisp_out)


        for mfID in range(n_cols): 
            centers_per_mf = tf.gather(layer.centers, mfID, axis=1)
            rules_per_mf = tf.gather(inputs, mfID, axis=1)

            crisp_out = tf.reduce_sum(tf.multiply(rules_per_mf,centers_per_mf))/ tf.reduce_sum(rules_per_mf)
            outputs = outputs.write(outputs.size(), crisp_out)

        outputs = outputs.stack()

    #    print("out,", outputs)

        return outputs





    

