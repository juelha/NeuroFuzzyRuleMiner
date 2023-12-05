# basics
import tensorflow as tf
import numpy as np
import yaml 
from yaml.loader import UnsafeLoader
import os

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


    def build(self, inputs_mean, inputs):
        """Initializes trainable parameters

        Args:
            inputs (tf.Tensor): inputs
        """

        feature_names = inputs_mean.keys().values.tolist()
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
        return self(inputs)


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
    
    
    def save_weights(self, dataset_name=None):
        """saves weights to yaml file
        
        Args:
            dataset_name (str): name of datasets weights have been built on
        """
        # save
        # opt 1: yaml
        file_name = f"config_mf.yaml"
        relative_path = "\weights"
        save_path = os.path.dirname(__file__) +  relative_path
        completeName = os.path.join(save_path, file_name)
        with open(completeName, 'w') as yaml_file:
            yaml.dump(self.train_params, yaml_file, default_flow_style=False)

        # opt 2: np.save
        # file_name = "config_weights"
        # other_name = os.path.join(save_path, file_name)
        # np.save(other_name, self.class_weights)
        print("saved successfully")
    
    def load_weights(self):
        """load weights from yaml file
        
        Args:
            filename etc
        Returns:
            loaded_weights (numpy.ndarray): 
        """
        # opt 1: yaml
        file_name = f"config_mf.yaml"
        relative_path =  "\weights"
        save_path = os.path.dirname(__file__) +  relative_path
        completeName = os.path.join(save_path, file_name)
        with open(completeName, 'r') as config_file:
            # Converts yaml document to python object
            config =yaml.load(config_file, Loader=UnsafeLoader)
            config = dict(config)
            self.centers = config["centers"]
            self.widths = config["widths"]
          #  print(type(weights))
          # print(weights)
        
        # # opt 2: np.save
        # file_name = "config_weights"
        # other_name = os.path.join(save_path, file_name)
        # loaded_weights = np.load(other_name+'.npy')
        #print("self.centers")
        #print(self.centers)
                # save params for training 
        self.train_params = {'centers': self.centers, 'widths': self.widths}
        print("sucessfully loaded weights")
        self.built = True
       # return weights
        return 0