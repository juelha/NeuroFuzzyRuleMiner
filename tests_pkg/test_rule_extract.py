# basics
import tensorflow as tf
import pandas as pd

# custom
from model_pkg import *
from neurofuzzy_pkg import * 
from tests_pkg import *
from neurofuzzy_pkg import utils

def testing_rule_extraction():
    """Testing Rule extraction of model

    Returns: 
        zero, if completed 
    """
    
    # test mamdani arc
    MamdaniModel = testing_training()

    # test 
    MLPModel = testing_mlp()

    RuleMiner(MamdaniModel, MLPModel)

    return 0



