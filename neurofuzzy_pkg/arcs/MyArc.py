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

class MyArc():
    """Architecture of the neuro fuzzy neural network
    """

    def __init__(self):
        """Initializes MyArc()-Object

        Attributes: 
            total_params (int): number of total params in network
            trainable_params (int): number of params that can be trained 
            FuzzificationLayer (FuzzificationLayer()): fuzzifies input
            RuleAntecedentLayer (RuleAntecedentLayer()): combines fuzzified input (IF-Part)
            RuleConsequentLayer (RuleConsequentLayer()): get value of combinations to output mfs (Then-Part)
            DefuzzificationLayer (DefuzzificationLayer()): get crip output
        """

        self.Name = "MyArc"
        self.total_params = 0
        self.trainable_params = 0

        self.FuzzificationLayer = FuzzificationLayer()
        self.RuleAntecedentLayer = RuleAntecedentLayer()
        self.NormalizationLayer = NormalizationLayer()
        self.RuleConsequentLayer = RuleConsequentLayer()

        self.internal_layers = [
            self.FuzzificationLayer,
            self.RuleAntecedentLayer,
            #self.NormalizationLayer,
            self.RuleConsequentLayer,
        ]


    def __call__(self, x):
        """Forward propagating the inputs through the network

        Args: 
            x (tf.Tensor):  crisp input (1 row of the dataset)

        Returns: 
            x (tf.Tensor): final output
        """
        for layer in self.internal_layers:
            x = layer(x)
        return x        

    def build(self, inputs, targets, feature_ranges, df_name =None):
        """Forward propagating the inputs through the network

        Args: 
            x (tf.Tensor): crisp input (1 row of the dataset)

        Returns: 
            done (boolean): if built
        """
        done = False


        # calc of amount of rules
        ## HOT FIX ##
        # inputs = <MapDataset element_spec=TensorSpec(shape=(11,), dtype=tf.float64, name=None)>
        # problem: cant get to the shape in the MapDataset element 
        for features in inputs.take(1):
          n_inputs = int(features.shape[0])
        #  print(n_inputs)

        n_mfs = self.FuzzificationLayer.n_mfs
        n = n_mfs * n_inputs
        k = self.RuleAntecedentLayer.n_participants 
        n_rules = int(coefficient(n, k) - n)

        for i in range(n_rules):
            self.RuleConsequentLayer.dictrules[i] = []
            self.RuleConsequentLayer.tars[i] = []

        # can build fuzzi before
        self.FuzzificationLayer.build(feature_ranges)
        for weird_thingy, weirdtar in (zip(tqdm(inputs, desc='building'), targets)):
            for layer in self.internal_layers:
                if type(layer) == type(self.FuzzificationLayer):
                  #  layer.build(feature_ranges)
                    x = layer(weird_thingy)
                elif type(layer) == type(self.RuleConsequentLayer):
                    x = layer.build(x, weirdtar)
                else:
                    x = layer.build(x)
      #  print("dict", self.RuleConsequentLayer.dictrules)
        # class weights 
        sums = []
        xes = []
        print("\n")
        for ruleID in tqdm(self.RuleConsequentLayer.dictrules, desc="selecting"):
            l = self.RuleConsequentLayer.dictrules[ruleID]
            #print("SUM", sum(l))
            for idx in l:#
                xes.append(idx.numpy())
            sums.append(sum(l).numpy())
            print("/n")
            max_val = max(l)
            idx_max = l.index(max_val)
            # print("l", l)
            # print("max", max_val)
            # print(idx_max)
            
            tar = self.RuleConsequentLayer.tars[ruleID][idx_max]
            # print("tar", self.RuleConsequentLayer.tars[ruleID])
            # print("tar", tar)

            self.RuleConsequentLayer.weights[ruleID] = tar
        # for layer in self.internal_layers:
        #     if type(layer) == type(self.RuleConsequentLayer):
        #         x = layer(x, x_og)
        #     else:
        #         x = layer(x)
            
        save_path = os.path.dirname(__file__) 

        assert  os.path.exists(save_path), f'save_path {save_path} not found'
        print(xes)
        print("type", type(xes))
        file_name = f"testing.yaml"
        full_path = os.path.join(save_path, file_name)
        with open(full_path, 'w') as yaml_file:
            yaml.dump([xes], yaml_file, default_flow_style=False)
        
        self.FuzzificationLayer.save_weights(df_name)
        self.FuzzificationLayer.load_weights(df_name)
        self.RuleConsequentLayer.save_weights(df_name)
        self.RuleConsequentLayer.load_weights(df_name)
        print("centers",self.FuzzificationLayer.centers)
        print("widths",self.FuzzificationLayer.widths)
        print("sums", sums)
        print("building done")
        done = True
        return done       
    

    def build_classweights(self, inputs, targets, feature_ranges, df_name =None):

        done = False


        # calc of amount of rules
        ## HOT FIX ##
        # inputs = <MapDataset element_spec=TensorSpec(shape=(11,), dtype=tf.float64, name=None)>
        # problem: cant get to the shape in the MapDataset element 
        for features in inputs.take(1):
          n_inputs = int(features.shape[0])
        #  print(n_inputs)

        n_mfs = self.FuzzificationLayer.n_mfs
        n = n_mfs * n_inputs
        k = self.RuleAntecedentLayer.n_participants 
        n_rules = int(coefficient(n, k) - n)

        for i in range(n_rules):
            self.RuleConsequentLayer.dictrules[i] = []
            self.RuleConsequentLayer.tars[i] = []

        for weird_thingy, weirdtar in (zip(tqdm(inputs, desc='building'), targets)):
            for layer in self.internal_layers:
                if type(layer) == type(self.FuzzificationLayer):
                    x = layer(weird_thingy)
                elif type(layer) == type(self.RuleConsequentLayer):
                    x = layer.build(x, weirdtar)
                else:
                    x = layer.build(x)
      #  print("dict", self.RuleConsequentLayer.dictrules)
        # class weights 
        for ruleID in tqdm(self.RuleConsequentLayer.dictrules, desc="selecting"):
            l = self.RuleConsequentLayer.dictrules[ruleID]
            max_val = max(l)
            idx_max = l.index(max_val)
          # print("l", l)
           # print("max", max_val)
            
            tar = self.RuleConsequentLayer.tars[ruleID][idx_max]
            #print("tar", tar)
            self.RuleConsequentLayer.weights[ruleID] = tar
        # for layer in self.internal_layers:
        #     if type(layer) == type(self.RuleConsequentLayer):
        #         x = layer(x, x_og)
        #     else:
        #         x = layer(x)        
        self.RuleConsequentLayer.save_weights(df_name)
        self.RuleConsequentLayer.load_weights(df_name)
        print("building done")
        done = True


    def build_MFs(self, feature_ranges, df_name):
        done = False


        self.FuzzificationLayer.build(feature_ranges)
               
        self.FuzzificationLayer.save_weights(df_name)
        self.FuzzificationLayer.load_weights(df_name)
        MFs.visuMFs(self.FuzzificationLayer, dir="after_building", func="InputMFs", max_vals=feature_ranges)
        print("building done")
        done = True
        return done       

        
    def get_params(self):
        """Get stats of params of network
        """
        
        self.trainable_params = 0

        for layer in self.internal_layers:
         #   print(layer)
            if hasattr(layer,'train_params'):
               # print(layer.built)
                assert layer.built == True, f'The layer {type(layer)} has not been built yet'
                for param in layer.train_params:
          #          print(param)
           #         print(layer.train_params[param].size)
                    self.trainable_params += layer.train_params[param].size
    
    def __str__(self) -> str:
        return "MyArc"

        
    def summary(self):
        # get params
        self.get_params()

        column_names = ['Total layer params', 'Trainable layer params', 'Non-trainable layer params']
        d = [self.total_params, self.trainable_params, self.total_params-self.trainable_params]

        df = pd.DataFrame(d, column_names)
        return df