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


from neurofuzzy_pkg.utils.WeightManager import save_weights
from neurofuzzy_pkg.utils.WeightManager import load_weights

class Builder():
    """Architecture of the neuro fuzzy neural network
    """

    def __init__(self, arc=None):
        self.arc = arc
        pass

    def __call__(self, inputs, targets, feature_maxs, feature_mins, df_name, n_mfs):
        """Forward propagating the inputs through the network

        Args: 
            x (tf.Tensor): crisp input (1 row of the dataset)

        Returns: 
            done (boolean): if built
        """
        self.build_MFs(feature_maxs, feature_mins, df_name, n_mfs)
        self.build_classweights(inputs, targets, df_name)
    
        return True       
    


    def build_classweights(self, inputs, targets, df_name=None):
        """
        Args:
            inputs():
            targets():
            df_name(str): needed for saving classweights
        """
        
        n_rules = int(self.arc.RuleAntecedentLayer.n_mfs**self.arc.RuleAntecedentLayer.n_features )

        for i in range(n_rules):
            self.arc.RuleConsequentLayer.dictrules[i] = []
            self.arc.RuleConsequentLayer.tars[i] = []

        for weird_thingy, weirdtar in (zip(tqdm(inputs, desc='building'), targets)):
            x = self.arc.FuzzificationLayer(weird_thingy)
            x = self.arc.RuleAntecedentLayer(x)
            x = self.arc.RuleConsequentLayer.build(x, weirdtar)

        for ruleID in tqdm(self.arc.RuleConsequentLayer.dictrules, desc="selecting"):
            l = self.arc.RuleConsequentLayer.dictrules[ruleID]
            idx_max = np.argmax(l)
            tar = self.arc.RuleConsequentLayer.tars[ruleID][idx_max]
            self.arc.RuleConsequentLayer.class_weights[ruleID] = tar
   
        save_weights(self.arc.RuleConsequentLayer, "class_weights", df_name)
        load_weights(self.arc.RuleConsequentLayer, "class_weights", df_name)


    def build_MFs(self, feature_maxs, feature_mins, df_name, n_mfs):
        
       # self.arc.FuzzificationLayer.build(feature_ranges)

       # feature_maxs = self.arc.FuzzificationLayer.preprocess_x(feature_maxs)
        #feature_mins = self.arc.FuzzificationLayer.preprocess_x(feature_mins)

        feature_maxs = feature_maxs.to_numpy() # drops names from max value, either do this or give names of features directly to visualizer
        feature_mins = feature_mins.to_numpy()
        # build centers and widths of MFs
        self.arc.FuzzificationLayer.centers = MFs.center_init(feature_mins, feature_maxs, n_mfs)
        self.arc.FuzzificationLayer.widths = MFs.widths_init(feature_mins, feature_maxs, n_mfs)
        print("WEI", self.arc.FuzzificationLayer.widths)


        save_weights(self.arc.FuzzificationLayer, "centers", df_name)
        save_weights(self.arc.FuzzificationLayer, "widths", df_name)
        
        done = True
        return done       