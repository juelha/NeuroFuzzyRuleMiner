# basics
import tensorflow as tf

# specifics 
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import *
from tensorflow.keras.optimizers import *

class MLP(tf.keras.Model):
    """Inherited from tf.keras.Model and responsible for:
    - describing architecture of NN  
    - forward pass of information  
    """

    def __init__(self, dim_hidden, perceptrons_out, activation=tf.sigmoid, k_r='l1_l2', a_r='l2'):
        """Initializes MLP()-Object

        Args:
            dim_hidden (tuple(int,int)): dimensions of hidden layers (hardcoded as dense layers)
                            1st arg: n_layers
                            2nd arg: n_perceptrons per layer
            perceptrons_out (int): n of perceptrons in output layer
            kr (str): kernel_regularizer, used to reduce the sum, l_1 uses abs(x) and l_2 uses square(x)
            ar (str): activity_regularizer, used to reduce the layer's output

        Attributes:
            hidden (list(Dense)): list containing hidden layers
            out (Dense): output layer
        """
        super(MLP, self).__init__()

        self.Name = "MLP"

        # initalizing hidden layers
        n_layers, n_perceptrons = dim_hidden

        self.hidden = [Dense(3)]

        # self.hidden = [Dense(
        #     n_perceptrons,
        #     activation=activation,
        #     kernel_regularizer=k_r,
        #     activity_regularizer=a_r
        #     ) for _ in range(n_layers)]

        perceptrons_out = 2
        self.out = Dense(perceptrons_out, activation=tf.sigmoid)


    def manage_input(self, inputs):
        """Encapsulates each input value in the input vector
        to allow the MLP to process the inputs

        Args:
            inputs (tf.Tensor): inputs to be managed (scalar tensor)
        
        Returns: 
            encaps_inputs (tf.Tensor): inputs for the model to process
                                       (each scalar is encapsulated in a list)

        Note:
            It is important that the input shape of the MLP 
            matches the input shape of the neuro-fuzzy-model 
            to check the rules  
        """
        encaps_inputs = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        [[x] for x in inputs]
        for input in inputs:
    
            encaps_inputs = encaps_inputs.write(encaps_inputs.size(), [input])
        return encaps_inputs.stack()


   # @tf.function
    def call(self, inputs):
        """Forward propagation of the inputs through the network

        Args:
            inputs (tf.Tensor): input vector of a batch 

        Returns:
            (tf.Tensor): final prediction of model
        """
        # make input fit to layer dense
        inputs = self.manage_input(inputs)

        for layer in self.hidden:
              inputs = layer(inputs)
        inputs = self.out(inputs)

        return inputs


    def invert(self, target):
        """ Backward pass the target through the network

        Args:
            target (tf.Tensor): the desired output

        Returns:
            (tf.Tensor): optimal input
        """
        for layer in reversed(list(self.hidden)):
            target = layer(target)
        return target

    
