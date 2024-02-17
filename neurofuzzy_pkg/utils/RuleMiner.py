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

    def __init__(self, neuro_fuzzy_model, df_name):
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
        self.linguistic_mf = ["low","medium","high"]
        self.lingusitic_output =  ["bad", "good"] # ["Setosa", "Versicolour", "Virginica"]
        self.n_mfs = len(self.linguistic_mf)
        self.n_participants = len(neuro_fuzzy_model.data.feature_names)
        self.n_outputs = len(self.lingusitic_output)# hc neuro_fuzzy_model.arc.RuleConsequentLayer.n_mfs 
        self.df_name = df_name

        # calling functions
        self.extractRules()
        

    def get_if_part(self, feature_names, mfs):

        
        feature_names = np.asarray(feature_names)
        feature_names = np.tile(feature_names[:, np.newaxis], self.n_mfs).ravel()

        mfs = np.tile(mfs, self.n_participants)

      #  fuzzy = np.char.add(feature_names, ",") 
       # fuzzy = np.char.add(fuzzy, mfs) 
      #  fuzzy = np.char.add(mfs, ",") 

  
        mfs = np.array_split(mfs, range(3, len(mfs), self.n_mfs))
        #x.reverse()
        mfs = np.stack(np.meshgrid(*mfs, indexing='ij'), axis=-1).reshape(-1, self.n_participants)


        return mfs#.tolist()
    
    def get_then_part(self):
        weights = self.arc.RuleConsequentLayer.weights
      #  print("weights", weights)
        
        output = []
        for w in weights:
            print("w", w)
            idx_max = np.argmax(w)
            output.append(self.lingusitic_output[idx_max])
               
        output = np.array(output)
        output = output.reshape(-1, 1)
        # Reshape the array to have a single column
       # neww = neww.reshape(-1, )
      #  neww = neww.T
      #  neww = neww.reshape(-1, 1)
       # neww = neww.tolist()
        return output


    def extractRules(self):
        """Function used for extracting rules from the dicts in the model
        """

       # rulesIF = self.arc.RuleAntecedentLayer.rulesIF # 1-90
        # dict that looks like
        # rulesIF {1: [{'xID': 0, 'mfID': 0}, {'xID': 1, 'mfID': 0}], 2: [{'xID': 0, 'mfID': 0}, {'xID': 1, 'mfID': 1}], ...

      #  rulesTHEN = self.arc.RuleConsequentLayer.rulesTHEN
        # dict that looks like
        ## old:  rulesTHEN {0: [1, 3, ...], 1: [36, 54, ...]
        # {'RS': x, 'target': self.weights}
        # {0: {'RS': x, 'target': self.weights}

      #  print("rulesIF",rulesIF)
      #  print("rulesTHEN",rulesTHEN)

        # going through rules for respective outcome 
        # -> per outcome on output mf & one entry final rules Dict
        # good yield -> outmfID = 1 
        # bad yield -> outmfID = 0
        rulesIF = self.get_if_part(self.feature_names, self.linguistic_mf)
    #    print("if", rulesIF)
        rulesTHEN = self.get_then_part()
     #   print("then", rulesTHEN)
       # rulesIF = np.char.add(rulesIF, ",") 
        huh = np.concatenate((rulesIF, rulesTHEN), axis=1)# #np.char.add(rulesIF, rulesTHEN) 
      #  print("huh", huh)
        #huh = huh[0]

            
        
        # # Initialize empty lists to store values
        # rules_Dict = {}
        # for i, feature in enumerate(self.feature_names):
        #     rules_Dict[feature] = []

        # rules_Dict['Target'] = []

        # # Process each element in the array
        # for i, element in enumerate(rulesIF):
        #     # Split the string by commas
        #    # elements = element.strip("'").split(',')

        #     # Extract values and append to respective lists
        #     for j,feature in enumerate(self.feature_names):
        #      #   print("i", i)
        #       #  print("keys", rules_Dict.keys())
        #        # print("fea", feature)
             
        #         #    print("HM", elements[i+1])
        #         rules_Dict[feature] = element[j]
        #     rules_Dict['Target'].append(rulesTHEN[i])

        #     # x1_values.append(elements[1])
        #     # x2_values.append(elements[3])
        #     # out_values.append(elements[4])
     

        # Create a pandas DataFrame
        self.feature_names.append("Class")
   #     print("honk", self.feature_names)
        df =      pd.DataFrame(huh, columns=self.feature_names)

        # Display the resulting DataFrame
    #    print(df)
        self.rulesDict = df
       # self.get_best_ruleIDs(self.inputs)

     #   print("/n dict", self.rulesDict)
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
    

    def get_best_rule_IDs(self,inputs, n=5):
        """
        """
        # check if n < generated rules
        acc = []
        activations = []
        for input_vec in tqdm(inputs, desc='class selecting'):
            # append activations of consequent layer
            activations.append(self.arc(input_vec)) 

        print("actvation", activations)
       # activations = np.array(activations
        
        activations = np.concatenate(activations, axis=1)
        print("actvation np ", activations)

        activations = np.sum(activations, axis = 1)
        best_indeces = np.argpartition(activations, -n)[-n:]
        #res = np.sum(*activations, axis = 0)
        print("res", best_indeces)
        best_indeces = best_indeces[np.argsort(activations[best_indeces])]
        print("res ewa ", best_indeces)

        print("here",self.rulesDict)
        best_rules = self.rulesDict.iloc[best_indeces]
        print("best", best_rules)
    #     for ruleID in tqdm(self.arc.RuleConsequentLayer.dictrules, desc="selecting"):
    #         l = self.arc.RuleConsequentLayer.dictrules[ruleID]
          
    #       #  max_val = max(l)
    #        # idx_max = l.index(max_val)

    #         idx_max = np.argmax(l)
            
    #         tar = self.arc.RuleConsequentLayer.tars[ruleID][idx_max]
    #        # print("tar", tar)
    #         self.arc.RuleConsequentLayer.weights[ruleID] = tar
   
    #     self.arc.RuleConsequentLayer.save_weights(df_name)
    #     self.arc.RuleConsequentLayer.load_weights(df_name)
    #    # print("building done")
    #     done = True
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
        # save results to csv files
        save_path = os.path.dirname(__file__) +  f'/../../results/{self.df_name}'

        ## good yield 
        
        file_name = f"{self.df_name}_rules.csv"
        if best:
            file_name = f"{self.df_name}_best__rules.csv"
        completeName = os.path.join(save_path, file_name)

        df_good = pd.DataFrame(rules)
        df_good.to_csv(completeName)
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
