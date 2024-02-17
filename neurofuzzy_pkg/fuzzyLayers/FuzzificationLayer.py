# basics
import tensorflow as tf
import numpy as np
import yaml 
from yaml.loader import UnsafeLoader
import os

# custom
import neurofuzzy_pkg.utils.MFs as MFs



class FuzzificationLayer():
    """Fuzzifying the crisp inputs by calculating their degree of membership 
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

    
    def preprocess_x(self, x):
        """
        makes it possible to utilize np vectorization 

        turns [1,2] into [1,1,1,2,2,2] for n_mfs = 3
        """
        return np.repeat(x, self.n_mfs)

 
    def build(self, x):
        """Initializes trainable parameters

        Args:
            x (tf.Tensor): inputs
        """
        x = self.preprocess_x(x)
        x = x.to_numpy() # drops names from max value, either do this or give names of features directly to visualizer
        # build centers and widths of MFs
        self.centers = MFs.center_init(x, self.n_mfs)
        self.widths = MFs.widths_init(x, self.n_mfs)

        self.built = True


    def __call__(self, x):
        """Calculates degree of membership of the crisp features 
            
        Args:
            x (tf.Tensor): crisp inputs to be fuzzified
        
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

        x = self.preprocess_x(x)
        self.inputs = x # saved for training 
       # print("CENTERS", self.centers)
       # print("WIdths", self.widths)
    
        fuzzy_x = self.mf_type(x, self.centers, self.widths)
            
        # check if resulting tensor has the correct shape
        # assert fuzzy_x.shape == (fuzzy_x.shape[0], self.n_mfs), f'Output of FuzzificationLayer has wrong shape \n \
        # should have shape {inputs.shape[0], self.n_mfs} but has shape {fuzzified_inputs.shape}'        
  
        self.outputs = fuzzy_x # saved for training 
        return fuzzy_x
    
    
    def save_weights(self, df_name):
        """saves weights to yaml file
        
        Args:
            dataset_name (str): name of datasets weights have been built on
        """
        # save
        # opt 1: yaml
        relative_path = f"/../../config/{df_name}/weights/"
        save_path = os.path.dirname(__file__) + relative_path
       

        assert  os.path.exists(save_path), f'save_path {save_path} not found'

        file_name = f"config_centers.yaml"
        full_path = os.path.join(save_path, file_name)
        with open(full_path, 'w') as yaml_file:
            yaml.dump(self.centers.tolist(), yaml_file)

        file_name = f"config_widths.yaml"
        full_path = os.path.join(save_path, file_name)
        with open(full_path, 'w') as yaml_file:
            yaml.dump(self.widths.tolist(), yaml_file)

        # opt 2: np.save
        # file_name = "config_weights"
        # other_name = os.path.join(save_path, file_name)
        # np.save(other_name, self.class_weights)
        print("saved successfully")
    
    def load_weights(self, df_name):
        """load weights from yaml file
        
        Args:
            filename etc
        Returns:
            loaded_weights (numpy.ndarray): 
        """
        # opt 1: yaml
        file_name = f"config_centers.yaml"
        relative_path = f"/../../config/{df_name}/weights/"
        save_path = os.path.dirname(__file__) +  relative_path
        full_path = os.path.join(save_path, file_name)
        assert os.path.exists(full_path), f'File {file_name} not found'
        with open(full_path, 'r') as config_file:
            # Converts yaml document to np array object
            config = yaml.load(config_file, Loader=UnsafeLoader)
            self.centers = np.array(config)

        file_name = f"config_widths.yaml"
        relative_path = f"/../../config/{df_name}/weights/"
        save_path = os.path.dirname(__file__) +  relative_path
        full_path = os.path.join(save_path, file_name)
        assert os.path.exists(full_path), f'File {file_name} not found'
        with open(full_path, 'r') as config_file:
            # Converts yaml document to np array object
            config = yaml.load(config_file, Loader=UnsafeLoader)
            self.widths = np.array(config)

          #  print(type(weights))
          # print(weights)
        
        # # opt 2: np.save
        # file_name = "config_weights"
        # other_name = os.path.join(save_path, file_name)
        # loaded_weights = np.load(other_name+'.npy')
        #print("self.centers")
        #print(self.centers)
                # save params for training 
        
        print("sucessfully loaded weights")
        self.built = True
       # return weights
        return 0