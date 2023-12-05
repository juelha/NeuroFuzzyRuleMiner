# basics
import tensorflow as tf
import numpy as np

# custom
import neurofuzzy_pkg.utils.MFs as MFs
import neurofuzzy_pkg.utils.math_funcs as math_funcs


class FuzzificationLayer():
    """
    The FuzzificationLayer()-Class is:
    - fuzzifying of the crisp inputs by calculating their degree of membership 
    Fuzzification-Neurons for the MFs Low, Medium, High:
        μ_L(x_1)
        /
    x_1 - μ_M(x_1)
        \ 
        μ_H(x_1)
    ___________________________________________________________
    """

    def __init__(self, n_mfs=3, mf_type=MFs.MF_gaussian):
        """Initializing FuzzificationLayer()-Object
        Args:
            n_mfs (int): number of MFs
            mf_type (callable): type of MFs
        Attributes:
            tunable (boolean): if parameters of layers can be tuned during training
            inputs (): inputs of layer
            outputs (): outputs of layer 
        """

        # for MFs
        self.mf_type = mf_type
        self.n_mfs = n_mfs
        self.built = False

        # for training
        self.tunable = True 
        self.train_params = None 
        self.inputs = []
        self.outputs = []


    def build(self, x):
        """Initializes trainable parameters

        Args:
            inputs (tf.Tensor): inputs
        """

        inputs_mean, feature_names = x
        n_inputs = tf.shape(inputs_mean)[0]


        # build centers and widths of MFs
        self.centers = np.asarray(MFs.center_init(self.n_mfs, inputs_mean),dtype=np.float32)
        self.widths = np.asarray(MFs.widths_init(self.n_mfs, self.centers, n_inputs), dtype=np.float32)


        # build weights 
        # self.weights = np.ones((self.n_mfs , n_inputs), dtype=np.float32)
        # print("weights in fu", self.weights)


        MFs.visuMFs(self, dir="after_building", func="InputMFs", names=feature_names, means=inputs_mean)

        # save params for training 
        self.train_params = {'centers': self.centers, 'widths': self.widths}#, 'weights':self.weights}#, 'biases':self.biases}
        
        self.built = True

        # call self
        return self(inputs_mean)


    def __call__(self, inputs):
        """Calculates the degree of membership of the crisp inputs 
            -> Fuzzification
        Args:
            inputs (tf.Tensor): crisp inputs to be fuzzified
        
        Returns:
            fuzzified_x (tf.Tensor): the fuzzified input, 
                                    for each input a row and for each MF one column
                                    Example for three MFs Low, Medium, High and 3 inputs x_1, x_2, x_3: 
                                    tf.Tensor(
                                        [[μ_L(x_1) μ_M(x_1)  μ_H(x_1)]
                                        [μ_L(x_2) μ_M(x_2)  μ_H(x_2)]
                                        [μ_L(x_3) μ_M(x_3)  μ_H(x_3)])
                                    shape=(n_inputs, n_mfs), dtype=float32)
        """


        # check if trainable params have been built
        assert self.built, f'Layer {type(self)} is not built yet'

        self.inputs = inputs # saved for training 

        # to output
        fuzzified_inputs  = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        # calculating the MF values μ "mus" per input
        for xID, x in enumerate(inputs):

            # there will be n_mfs mus per input
            mus_per_x = []
            for mfID in range(self.n_mfs):

                # calling MF 
                mu = self.mf_type(x, self.centers[xID][mfID], self.widths[xID][mfID])    
                mus_per_x.append(mu)
        
            # write to TensorArray
            fuzzified_inputs = fuzzified_inputs.write(fuzzified_inputs.size(), mus_per_x)

        # return the values in the TensorArray as a stacked tensor
        fuzzified_inputs = fuzzified_inputs.stack()

        # check if resulting tensor has the correct shape
        assert fuzzified_inputs.shape == (inputs.shape[0], self.n_mfs), f'Output of FuzzificationLayer has wrong shape \n \
        should have shape {inputs.shape[0], self.n_mfs} but has shape {fuzzified_inputs.shape}'        
  
        self.outputs = fuzzified_inputs # saved for training 
        return fuzzified_inputs