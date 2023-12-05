
# basics
import tensorflow as tf
import pandas as pd

# custom
from model_pkg import *
from neurofuzzy_pkg import *




def testing_mlp():
    """
    basic mlp to test    
    """
        #  ##
    dim_hidden = (2,11)
    perceptron_out = 2

    MLPArch = MLP(dim_hidden,perceptron_out)
    MLPtrainer = Trainer()

    myMLP = Model(DataPipeline(),MLPArch,MLPtrainer)

    myMLP.train()
    print(myMLP.arc.summary())
    myMLP.evalutate()

    myMLP.trainer.visualize_training()

    return myMLP
