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


class Builder():
    """Architecture of the neuro fuzzy neural network
    """

    def __init__(self, arc=None):
        self.arc = arc
        pass

    def __call__(self, inputs, targets, feature_ranges, df_name=None):
        """Forward propagating the inputs through the network

        Args: 
            x (tf.Tensor): crisp input (1 row of the dataset)

        Returns: 
            done (boolean): if built
        """
        self.build_MFs(feature_ranges, df_name)
        self.build_classweights(inputs, targets, df_name)
    
        return True       
    


    def build_classweights(self, inputs, targets, df_name =None):

        # calc of amount of rules
        ## HOT FIX ##
        # inputs = <MapDataset element_spec=TensorSpec(shape=(11,), dtype=tf.float64, name=None)>
        # problem: cant get to the shape in the MapDataset element 
        # print("HEREREEEEE", inputs)
        # print("HEREREEEEE", inputs.shape)
        n_inputs = inputs.shape[1] # horizontal

        # for features in inputs.take(1):
        #   n_inputs = int(features.shape[0])
        #  print(n_inputs)


        n_rules = int(self.arc.RuleAntecedentLayer.n_mfs**self.arc.RuleAntecedentLayer.n_features )

        for i in range(n_rules):
            self.arc.RuleConsequentLayer.dictrules[i] = []
            self.arc.RuleConsequentLayer.tars[i] = []

        for weird_thingy, weirdtar in (zip(tqdm(inputs, desc='building'), targets)):
            x = self.arc.FuzzificationLayer(weird_thingy)
            x = self.arc.RuleAntecedentLayer(x)
            x = self.arc.RuleConsequentLayer.build(x, weirdtar)

           

      #  for ruleID in tqdm(self.RuleConsequentLayer.dictrules, desc="selecting"):
        for ruleID in tqdm(self.arc.RuleConsequentLayer.dictrules, desc="selecting"):
            l = self.arc.RuleConsequentLayer.dictrules[ruleID]
          
          #  max_val = max(l)
           # idx_max = l.index(max_val)

            idx_max = np.argmax(l)
            
            tar = self.arc.RuleConsequentLayer.tars[ruleID][idx_max]
           # print("tar", tar)
            self.arc.RuleConsequentLayer.weights[ruleID] = tar
   
        self.arc.RuleConsequentLayer.save_weights(df_name)
        self.arc.RuleConsequentLayer.load_weights(df_name)
       # print("building done")
        done = True


    def build_MFs(self, feature_ranges, df_name):
        done = False


       # print("sdjkdfg", feature_ranges)
        self.arc.FuzzificationLayer.build(feature_ranges)

        #print("here",self.FuzzificationLayer.centers)
               
        self.arc.FuzzificationLayer.save_weights(df_name)
        self.arc.FuzzificationLayer.load_weights(df_name)
        #MFs.visuMFs(self.FuzzificationLayer, dir="after_building", func="InputMFs", max_vals=feature_ranges)
      #  print("building done")
        done = True
        return done       