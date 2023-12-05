# basics
import tensorflow as tf
import pandas as pd
import os.path
from numpy import nan

# custom
from model_pkg import *
from neurofuzzy_pkg import * 


class ruleExtractor():
    """
    The ruleExtractor-Class() is responsible for:
    - extracting the rules from the neurofuzzy model
    - and validating the rulse by testing them with the MLP classifier
    ___________________________________________________________
    """

    def __init__(self, neuro_fuzzy_model, mlp_model):
        """Init ruleExtractor and calling extractRules()

        Args:
            neuro_fuzzy_model (Custom-CLass()): trained network of the neurofuzzy model
            MLP (Custom-Class()): trained classifier, used to validate rules 
        """

        # arcs used
        self.mamdaniArc = neuro_fuzzy_model.arc
        self.classifier = mlp_model 

        # for final Dict of rules
        self.rulesDict = {}
        self.feature_names = neuro_fuzzy_model.feature_names
        self.linguistic_mf = ["low","medium","high"]
        self.n_outputs = neuro_fuzzy_model.arc.RuleConsequentLayer.n_mfs 

        # calling functions
        self.extractRules()
        

    def extractRules(self):
        """Function used for extracting rules from the dicts in the model
        """

        rulesIF = self.mamdaniArc.RuleAntecedentLayer.rulesIF # 1-90
        # dict that looks like
        # rulesIF {1: [{'xID': 0, 'mfID': 0}, {'xID': 1, 'mfID': 0}], 2: [{'xID': 0, 'mfID': 0}, {'xID': 1, 'mfID': 1}], ...

        rulesTHEN = self.mamdaniArc.RuleConsequentLayer.rulesTHEN
        # dict that looks like
        # rulesTHEN {0: [1, 3, ...], 1: [36, 54, ...]

      #  print("rulesIF",rulesIF)
       # print("rulesTHEN",rulesTHEN)

        # going through rules for respective outcome 
        # -> per outcome on output mf & one entry final rules Dict
        # good yield -> outmfID = 1 
        # bad yield -> outmfID = 0

        # iterating over rules that are sorted by mf from RuleConsequentLayer
        for outmfID in range(self.n_outputs):

            # setting up ruleDict
            self.rulesDict[outmfID] = {}
            self.rulesDict[outmfID]["ruleID"] = []

            # setting up cols of ruleDict
            for para in self.feature_names:
                self.rulesDict[outmfID][para] = []

            # iterate over rules per mf 
            for ruleID in rulesTHEN[outmfID]:
                
                # keeping track of which parameters have been used
                # usedNames is needed to fill up the rest of cols with nans
                usedNames = []

                # getting information on participating elements of a rule  
                elements_from_rule = []
                for element in rulesIF[ruleID]:
                    elements_from_rule.append(element)

                # validate rule with classifier 
                if self.checkRule(elements_from_rule, outmfID):
                    # adding ruleID from from validated rule from rulesTHEN to rulesDict
                    self.rulesDict[outmfID]["ruleID"].append(ruleID)

                    for element in elements_from_rule:
                        # get parameter name
                        name = self.feature_names[element['xID']]
                        # add value of mf under parameter name
                        self.rulesDict[outmfID][name].append(self.linguistic_mf[element['mfID']])
                        usedNames.append(name)
                
                    # determine which parameters to fill with nan
                    missingNames = list(set(self.feature_names) - set(usedNames)) 
                    for name in missingNames:
                        self.rulesDict[outmfID][name].append(nan)

                # if rule was not validated, do not add 
                else:
                    continue
            
        # save results to csv files
        save_path = os.path.dirname(__file__) +  '/../../results'

        ## good yield 
        file_name = "good_yield.csv"
        completeName = os.path.join(save_path, file_name)
        df_good = pd.DataFrame(self.rulesDict[1])
        df_good.to_csv(completeName)

        ## bad yield 
        file_name = "bad_yield.csv"
        completeName = os.path.join(save_path, file_name)
        df_bad = pd.DataFrame(self.rulesDict[0])
        df_bad.to_csv(completeName)

        self.df_bad = df_bad
        self.df_good = df_good

        return 0


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
            crisp_xs.append(self.mamdaniArc.FuzzificationLayer.centers[element['xID']][element['mfID']])

        # construct vector of zeros
        crisp_inputs = []
        for i in range(len(self.feature_names)):
            crisp_inputs.append(0)
            # if xID is reached replace zero with crisp value 
            for index, xID in enumerate(xIDs):
                if i == xID:
                    crisp_inputs[i] = crisp_xs[index]

        # construct dataset
        test_seq = tf.convert_to_tensor(([crisp_inputs]),dtype=tf.float32)
        test_tar = tf.convert_to_tensor(([target]),dtype=tf.float32) 
        ds = tf.data.Dataset.from_tensor_slices((test_seq, test_tar))
        ds = ds.apply(self.pipeline)

        if self.classifier.validate_input(ds):
            return True

        return False  


    def print_results(self,):
        """Printing results of rule extraction to console
        """      

        # bad yield
        print("\n┌────────────────────────────────────────────────────────────────┐" + ("\n") +
                "│                         OUTCOME: 0                             │" + ("\n") +
                "└────────────────────────────────────────────────────────────────┘\n")
        print(self.df_bad)

        rules_generated_bad = len(self.mamdaniArc.RuleConsequentLayer.rulesTHEN[0]) 
        rules_validated_bad = len(self.rulesDict[0]['ruleID']) 
        if rules_generated_bad == 0:
            accuracy_bad = 0
        else:
            accuracy_bad = 100 * (rules_validated_bad/rules_generated_bad)

        print(f'Rules Generated for Bad Yield: {rules_generated_bad}')
        print(f'Rules Validated for Bad Yield: {rules_validated_bad}')
        print(f'Accuracy of Rules Bad Yield: {round(accuracy_bad,2)}%')
        



        # good yield
        print("\n┌────────────────────────────────────────────────────────────────┐" + ("\n") +
                "│                         OUTCOME: 1                             │" + ("\n") +
                "└────────────────────────────────────────────────────────────────┘\n")
        print(self.df_good)
        
        rules_generated_good = len(self.mamdaniArc.RuleConsequentLayer.rulesTHEN[1]) 
        rules_validated_good = len(self.rulesDict[1]['ruleID']) 
        if rules_generated_good == 0: # catch divide by zero error
            accuracy_good = 0
        else:
            accuracy_good = 100 * (rules_validated_good/rules_generated_good)
        
        print(f'Rules Generated for Good Yield: {rules_generated_good}')
        print(f'Rules Validated for Good Yield: {rules_validated_good}')
        print(f'Accuracy of Rules Good Yield: {round(accuracy_good,2)}%')
   

        # in total
        total_rules_generated = rules_generated_bad + rules_generated_good
        total_rules_validated = rules_validated_bad + rules_validated_good
        if total_rules_generated == 0: # catch divide by zero error
            total_accuracy = 0
        else:
            total_accuracy = 100 * (total_rules_validated/total_rules_generated)
        total_possible_rules = self.mamdaniArc.RuleAntecedentLayer.n_rules*2

        print("\n┌────────────────────────────────────────────────────────────────┐" + ("\n") +
                "│                         IN TOTAL                               │" + ("\n") +
                "└────────────────────────────────────────────────────────────────┘\n")
        print(f'Total Rules being possible by combination: {total_possible_rules}')
        print(f'Total Rules Generated: {total_rules_generated}')
        print(f'Total Rules Validated: {total_rules_validated}')
        print(f'Total Accuracy of Rules: {round(total_accuracy,2)}%')
        


    def pipeline(self, ds):
        """Transforms crisp input (generated from rules) into usable input for mlp"""
        ds = ds.map(lambda inputs, target: (inputs, tf.one_hot(int(target), 2)))
        ds = ds.batch(1) 
        ds = ds.prefetch(1)

        return ds
