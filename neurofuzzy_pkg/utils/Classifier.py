# basics
import tensorflow as tf
import pandas as pd

# custom
from neurofuzzy_pkg.fuzzyLayers import *
from neurofuzzy_pkg.utils.math_funcs import coefficient
import neurofuzzy_pkg.utils.MFs as MFs


from tqdm import tqdm 

import yaml 
from yaml.loader import UnsafeLoader
import os


class Classifier():
    """Outputs the class for a datasample using a trained myarc
    """

    def __init__(self, arc=None):
        self.arc = arc
        pass

    def __call__(self, input_vec, df_name=None):

        return self.get_class(self, input_vec, df_name=None)
    

    def get_class(self, input_vec, df_name=None): 
        # propagating through network
        outputs = self.arc(input_vec)
       # print("out", outputs)
        outputs = np.sum(outputs, axis=1) # make 1d
        idx_max = np.argmax(outputs)
      # print("out after", outputs)

       # max_val = max(outputs)
       # idx_max = outputs.index(max_val)
        classID = self.arc.RuleConsequentLayer.weights[idx_max]
        return classID
    

    def get_class_accuracy(self, inputs, targets, df_name =None):
        acc = []
        for input_vec, target_vec in (zip(tqdm(inputs, desc='class testing'), targets)):
            classID = self.get_class(input_vec) 
            acc.append(self.is_class_correct(classID, target_vec))
        return np.mean(acc)

    
    def is_class_correct(self, classID, target):
        #print("target", target)
       # print("classid", classID)
        return classID == target