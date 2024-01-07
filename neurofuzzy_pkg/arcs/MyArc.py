# basics
import tensorflow as tf
import pandas as pd

# custom
from neurofuzzy_pkg.fuzzyLayers import *


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

    def build(self, inputs, targets, inputs_mean, df=None):
        """Forward propagating the inputs through the network

        Args: 
            x (tf.Tensor): crisp input (1 row of the dataset)

        Returns: 
            done (boolean): if built
        """
        done = False
        #inputs = inputs[0]
        print("MARK", inputs)
        n_rules = 9
        for i in range(n_rules):
            self.RuleConsequentLayer.dictrules[i] = []
            self.RuleConsequentLayer.tars[i] = []

        for weird_thingy, weirdtar in zip(inputs,targets): 
            for layer in self.internal_layers:
                if type(layer) == type(self.FuzzificationLayer):
                    x = layer.build(inputs_mean, weird_thingy)
                elif type(layer) == type(self.RuleConsequentLayer):
                    x = layer.build(x, weirdtar)
                else:
                    x = layer.build(x)
      #  print("dict", self.RuleConsequentLayer.dictrules)
        # class weights 
        for ruleID in self.RuleConsequentLayer.dictrules:
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
        
        self.FuzzificationLayer.save_weights()
        self.FuzzificationLayer.load_weights()
        self.RuleConsequentLayer.save_weights()
        self.RuleConsequentLayer.load_weights()
        print("building done")
        done = True
        return done       



    def build_MFs(self, inputs, targets, inputs_mean, df=None):
        done = False
        #inputs = inputs[0]
        print("MARK", inputs)
        n_rules = 9
       
        for weird_thingy, weirdtar in zip(inputs,targets): 
            for layer in self.internal_layers:
                if type(layer) == type(self.FuzzificationLayer):
                    x = layer.build(inputs_mean, weird_thingy)
               
        self.FuzzificationLayer.save_weights()
        self.FuzzificationLayer.load_weights()

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