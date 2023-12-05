# basics
import tensorflow as tf
import pandas as pd

# custom
from model_pkg import *
from neurofuzzy_pkg import * 
from neurofuzzy_pkg import utils


def testing_forward_pass():
    """Testing the forward pass of model with simple inputs

    Returns: 
        zero, if completed 
    """

    ## mamdani approach ##
    mamdaniArch = MamdaniArc()
    test_in = tf.constant([1,2,3],dtype=tf.float32)
    predict = mamdaniArch(test_in)
    print("predict",predict)

    return 0

def testing_training():
    """Testing the training of model with simple inputs

    Returns: 
        zero, if completed 
    """
    ## mamdani approach ##
    mamdaniArch = MamdaniArc()
    myTrainer = neurofuzzyTrainer() 
    

    # simple dataset
    test_seq = tf.convert_to_tensor(([[1,2,3]]),dtype=tf.float32)
    test_tar = tf.convert_to_tensor(([[4,5]]),dtype=tf.float32) 
    test_ds = tf.data.Dataset.from_tensor_slices((test_seq, test_tar))

    myMamdani = Model(test_ds, mamdaniArch, myTrainer)
    # training loop
    myMamdani.trainer.training_loop(test_ds, test_ds) 
   # myMamdani.train()

    return myMamdani

