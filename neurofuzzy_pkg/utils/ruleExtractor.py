# basics
import tensorflow as tf
import pandas as pd
import os.path
from numpy import nan
import numpy as np

# custom
from model_pkg import *
from neurofuzzy_pkg import * 

from tqdm import tqdm 

class ruleExtractor():
    """
    The ruleExtractor-Class() is responsible for:
    - extracting the rules from the neurofuzzy model
    - and validating the rulse by testing them with the MLP classifier
    ___________________________________________________________
    """

    def __init__(self, neuro_fuzzy_model, df_name):
        """Init ruleExtractor and calling extractRules()

        Args:
            neuro_fuzzy_model (Custom-CLass()): trained network of the neurofuzzy model
            MLP (Custom-Class()): trained classifier, used to validate rules 
        """

        # arcs used
        self.arc = neuro_fuzzy_model.arc
       # self.classifier = mlp_model 

        # for final Dict of rules
        self.rulesDict = {}
        self.feature_names = neuro_fuzzy_model.data.feature_names
        self.linguistic_mf = ["low","medium","high"]
        self.lingusitic_output = ["bad", "good"]
        
        self.n_outputs = 2# hc neuro_fuzzy_model.arc.RuleConsequentLayer.n_mfs
        self.n_mfs = 3
        self.n_participants = 2
        self.df_name = df_name

        # calling functions
        self.extractRules()
        

    def extractRules(self):
        """Function used for extracting rules from the dicts in the model
        """

        rulesIF = self.arc.RuleAntecedentLayer.rulesIF # 1-90
        # dict that looks like
        # rulesIF {1: [{'xID': 0, 'mfID': 0}, {'xID': 1, 'mfID': 0}], 2: [{'xID': 0, 'mfID': 0}, {'xID': 1, 'mfID': 1}], ...

        rulesTHEN = self.arc.RuleConsequentLayer.rulesTHEN
        # dict that looks like
        ## old:  rulesTHEN {0: [1, 3, ...], 1: [36, 54, ...]
        # {'RS': x, 'target': self.weights}
        # {0: {'RS': x, 'target': self.weights}

        print("rulesIF",rulesIF)
        print("rulesTHEN",rulesTHEN)

        # going through rules for respective outcome 
        # -> per outcome on output mf & one entry final rules Dict
        # good yield -> outmfID = 1 
        # bad yield -> outmfID = 0

            
        # setting up cols of ruleDict
        for para in self.feature_names:
            self.rulesDict[para] = []
        self.rulesDict['Output'] = []

        # iterating over rules 
        for ruleID in rulesIF:
            
                
            # keeping track of which parameters have been used
            # usedNames is needed to fill up the rest of cols with nans
            usedNames = []

            # validate rule with classifier 
           # ruleAcc =  self.checkRule(rulesIF[ruleID], rulesTHEN[ruleID])
          #  self.rulesDict[ruleID] = [rulesIF[ruleID], {"then": rulesTHEN[ruleID], 'acc': ruleAcc}]
                    
             
            for participant in rulesIF[ruleID]:
                # get parameter name
                name = self.feature_names[participant['xID']]
                # add value of mf under parameter name
                self.rulesDict[name].append(self.linguistic_mf[participant['mfID']])
                
                usedNames.append(name)
        
            # determine which parameters to fill with nan
            missingNames = list(set(self.feature_names) - set(usedNames)) 
            for name in missingNames:
                self.rulesDict[name].append(nan)
            
            self.rulesDict['Output'].append(rulesTHEN[ruleID])
            # if rulesTHEN[ruleID] == [1,0]:
            #     self.rulesDict['Output'].append(self.lingusitic_output[0])
            # if rulesTHEN[ruleID] == [0,1]:
            #     self.rulesDict['Output'].append(self.lingusitic_output[1])
        

        print("/n dict", self.rulesDict)
        self.save_results(self.rulesDict)
      #  self.print_results()

        ## bad yield 
        # file_name = "bad_yield.csv"
        # completeName = os.path.join(save_path, file_name)
        # df_bad = pd.DataFrame(self.rulesDict[0])
        # df_bad.to_csv(completeName)

       # self.df_bad = df_bad
       # self.df_good = df_good

        return 0



    
    def get_best_rules(self, inputs, n=10):
        """
        Args:
            inputs ():
            n (int): number of rules to select
        """
        # check if n < generated rules
        acc = []
        x = []
        # get activations per output for inptus
        for input_vec in tqdm(inputs, desc='class selecting'):
            
            x.append(self.arc(input_vec)) 

        x = np.concatenate(x, axis=1)
        #print("actvation np ", activations)

        x = np.sum(x, axis = 1)
        #x = x/np.shape(inputs)[0] # normalize
        best_indeces = np.argsort(x)[-n:]  # get best n indeces, low to high
        best_indeces = np.flip(best_indeces) # reverse so highest activation is first
      #  print("hooonk", best_indeces)
        
        best_rules = {}
        for para in self.feature_names:
            best_rules[para] = []
        best_rules['Output'] = []

        rule=[]
        for key in list(self.rulesDict.keys()):
            for idx in best_indeces:
                best_rules[key].append(self.rulesDict[key][idx])
       # best_rules = self.rulesDict.iloc[best_indeces]
       # best_rules = best_rules.assign( Activations = x[best_indeces] )
        self.save_results(best_rules, best=True)
        return best_rules


    def checkRule(self, elements, target):
        """Check the rule by transforming the rule to a crisp input vector and 
        plugging that into a MLP classifier

        Args:
            elements (list(dict)): list of participating elements of a rule
            target (bool): the output mf of the rule 

        Returns:
            TRUE if 
        """

        # transform to crisp input 
        xIDs = []
        crisp_xs = []
        for element in elements:
            xIDs.append(element['xID'])
            # get crisp value of x (use center of mf)
            crisp_xs.append(self.arc.FuzzificationLayer.centers[element['xID']][element['mfID']])

        # # construct vector of zeros
        # crisp_inputs = []
        # for i in range(len(self.feature_names)):
        #     crisp_inputs.append(0)
        #     # if xID is reached replace zero with crisp value 
        #     for index, xID in enumerate(xIDs):
        #         if i == xID:
        #             crisp_inputs[i] = crisp_xs[index]

        # construct dataset
        test_seq = tf.convert_to_tensor(([crisp_xs]),dtype=tf.float32)
        test_tar = tf.convert_to_tensor(([target]),dtype=tf.float32) 
        ds = tf.data.Dataset.from_tensor_slices((test_seq, test_tar))
        ds = ds.apply(self.pipeline)

        # if self.classifier.validate_input(ds):
        #     return True

        return False  

    def validate_input(self, rule_ds):
        """Validate an input obtained by a rule in ruleExtractor()

        Args: 
            rule_ds (PrefetchDataset): input dataset, created in ruleExtractor()

        Returns:
            (boolean): if a rule has been validated by producing the target
            that was given by the rule
        """
        bias = 0.5
        for (input, targets) in rule_ds:
            input = tf.reshape(input,(len(self.feature_names),))

            # pass forwards
            prediction = self.arc(input)

            # get accuracy
            sample_test_accuracy =  targets == np.round(prediction, 0)
            sample_test_accuracy = np.mean(sample_test_accuracy)
            if sample_test_accuracy < bias:
                return False
            return True

    def save_results(self, rules, best=False):
        """
        Args:
            rules (panda dataframe): 
        """
        # save results to csv files
        save_path = os.path.dirname(__file__) +  f'/../../results/{self.df_name}'
        
        file_name = f"{self.df_name}_rules.csv"
        if best:
            file_name = f"{self.df_name}_best__rules.csv"
        completeName = os.path.join(save_path, file_name)

        df = pd.DataFrame(rules)
        df.index += 1 
        df.to_csv(completeName)
        return 0 


    def print_results(self,):
        """Printing results of rule extraction to console
        """      

        # bad yield
        print("\n┌───────────────────────────────────────────────────────────────┐" + ("\n") +
                "│                           Results                             │" + ("\n") +
                "└───────────────────────────────────────────────────────────────┘\n")

        n_rules_generated = int(self.n_mfs**self.n_participants)
        n_rules_validated = 0

        validation_percentage = 100 * (n_rules_validated/n_rules_generated)

        print(f'# Rules Generated: {n_rules_generated}')
        print(f'# Rules Validated: {n_rules_validated}')
        print(f'Percentage of Validation: {round(validation_percentage,2)}%')
        
        


    def pipeline(self, ds):
        """Transforms crisp input (generated from rules) into usable input for mlp"""
        ds = ds.map(lambda inputs, target: (inputs, tf.one_hot(int(target), 2)))
        ds = ds.batch(1) 
        ds = ds.prefetch(1)

        return ds
