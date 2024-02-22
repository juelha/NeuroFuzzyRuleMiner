# basics
import tensorflow as tf
import pandas as pd

# custom
from neurofuzzy_pkg.fuzzyLayers import *
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
            print("classD", classID)
            print("target_vec", target_vec) 
            acc.append(classID == target_vec)
        total_acc = np.mean(acc, axis=1)
        total_acc = np.where(total_acc != 1, 0, 1)
        print("accs", total_acc)
        self.print_results(np.mean(total_acc), len(inputs) - np.count_nonzero( total_acc))
        return np.mean(total_acc)

    
    def is_class_correct(self, classID, target):
        #print("target", target)
       # print("classid", classID)
        return classID == target
    

    def print_results(self, total_acc, n_incorrect):
        print("\n┌─────────────────────────────────────────────────────────────────┐" + ("\n") +
                f"  Classifying Results: Accuracy total: {np.round(total_acc,2)}, No. incorrect: {n_incorrect} " + ("\n") +
                "└─────────────────────────────────────────────────────────────────┘\n")