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


class RuleMiner():
    """
    The RuleMiner-Class() is responsible for:
    - extracting the rules from the neurofuzzy model
    - and validating the rulse by testing them with the MLP classifier
    ___________________________________________________________
    """

    def __init__(self, neuro_fuzzy_model, df_name, fuzzy_labels, ling_out):
        """Init ruleExtractor and calling extractRules()

        Args:
            neuro_fuzzy_model (Custom-CLass()): trained network of the neurofuzzy model
            MLP (Custom-Class()): trained classifier, used to validate rules 
        """

        # arcs used
        self.arc = neuro_fuzzy_model.arc
        self.classifier = None#mlp_model 

        # for final Dict of rules
        self.rulesDict = {}
        self.inputs = None
        self.feature_names = neuro_fuzzy_model.data.feature_names
        self.fuzzy_labels = fuzzy_labels #["low","medium","high"]
        self.lingusitic_output =  ling_out #["Setosa", "Versicolour", "Virginica"] #["bad", "good"] # 
        self.n_mfs = len(self.fuzzy_labels)
        self.n_participants = len(neuro_fuzzy_model.data.feature_names)
        self.n_outputs = len(self.lingusitic_output)# hc neuro_fuzzy_model.arc.RuleConsequentLayer.n_mfs 
        self.df_name = df_name

        # calling functions
        self.extractRules()
        

    def extractRules(self):
        """Function used for extracting rules from the dicts in the model
        """

        # get rules and combine
        rulesIF = self.get_if_part(self.fuzzy_labels)
        rulesTHEN = self.get_then_part()
      #  print("huh", rulesIF)
     #   print("ha", rulesTHEN)
        rules = np.concatenate((rulesIF, rulesTHEN), axis=1)
   
        # save as df 
        self.feature_names.append("Class")
        self.rulesDict = pd.DataFrame(rules, columns=self.feature_names)

        self.save_results(self.rulesDict)
        return 0
    

    def get_if_part(self, fuzzy_labels):
        """combine fuzzy labels just like inputs

        Args:
            fuzzy_labels (list(str)):  ["low","medium","high"] or  ["small","medium","large"]
        """
        fuzzy_labels = np.tile(fuzzy_labels, self.n_participants)
        fuzzy_labels = np.array_split(fuzzy_labels, range(self.n_mfs, len(fuzzy_labels), self.n_mfs))
        fuzzy_labels = np.stack(np.meshgrid(*fuzzy_labels, indexing='ij'), axis=-1).reshape(-1, self.n_participants)
        return fuzzy_labels
    
    def get_then_part(self):
        """
        
        """
        weights = self.arc.RuleConsequentLayer.class_weights
        
        output = []
        # go through weights and select the max idx
        # since weights are one-hot encoded this will match the idx of the belonging class
       # print("w", weights)
        for w in weights:
            idx_max = np.argmax(w)
            output.append(self.lingusitic_output[idx_max])
               
        output = np.array(output)
        output = output.reshape(-1, 1)
        return output


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
        x = x/np.shape(inputs)[0] # normalize
        best_indeces = np.argsort(x)[-n:]  # get best n indeces, low to high
        best_indeces = np.flip(best_indeces) # reverse so highest activation is first
      #  print("hooonk", best_indeces)
        best_rules = self.rulesDict.iloc[best_indeces]
        best_rules = best_rules.assign( Activations = x[best_indeces] )
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



        # construct dataset
        test_seq = tf.convert_to_tensor(([crisp_xs]),dtype=tf.float32)
        test_tar = tf.convert_to_tensor(([target]),dtype=tf.float32) 
        ds = tf.data.Dataset.from_tensor_slices((test_seq, test_tar))
        ds = ds.apply(self.pipeline)

        if self.classifier.validate_input(ds):
            return True

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
        save_path = os.path.dirname(__file__) +  f'/../results/{self.df_name}'
        
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
