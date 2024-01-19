# basics
import tensorflow as tf
import numpy as np
import pandas as pd
# specifics
from tensorflow.keras.optimizers import *
import os.path

# custom
from model_pkg import *


class Model():
    """Masterclass for combining data, architecture and trainer.
    """

    def __init__(self, data, arc, trainer):
        """Initializes Model()-Object

        Args:
            data (DataPipeline()): provides the data for the model
            arc (Custom-Class): architecture of model
            trainer (Trainer()): trainer of model

        Attributes:
            train_ds (tf.PrefetchDataset): training dataset
            test_ds (tf.PrefetchDataset): testing dataset
            validation_ds (tf.PrefetchDataset): validating dataset
            parameter_names (list(str)): column names of dataset, needed for extracting rules later
        """

        # data
        self.data = data

        # architecture
        self.arc = arc
        

        # trainer
        self.trainer = trainer

    def run(self):
        """
        """
        self.build()
        self.train()
        self.summary()

    def build_MyArc(self):
        # load data for building my arc
        self.data.load_data_for_building()
        self.feature_names = self.data.feature_names
        self.arc.feature_names = self.feature_names
        self.arc.feature_ranges = self.data.feature_ranges
        self.arc.build(self.data.inputs, self.data.targets, self.data.feature_ranges, self.data.df_name)
        print("Build done")

    def build_MyArc_CW(self):
        # load data for building my arc
        self.data.load_data_for_building()
        self.feature_names = self.data.feature_names
        self.arc.feature_names = self.feature_names
        self.arc.feature_ranges = self.data.feature_ranges
        self.arc.build_classweights(self.data.inputs, self.data.targets, self.data.feature_ranges, self.data.df_name)
        print("Build done")

    def build_MyArc_MF(self):
        # load data for building my arc
        self.data.load_data_for_building()
        self.arc.build_MFs(self.data.feature_ranges, self.data.df_name)
        print("Build done")




    def build(self):
        """
        """
        # todo just do example datastet
        #print("self.inputs_mean", self.inputs_mean)
        # get feature_names names 
        self.feature_names = self.data.feature_names
        self.arc.feature_names = self.feature_names
        self.arc.build(self.inputs_mean, self.feature_names)
       # self.trainer.test(self.inputs_mean)
        self.built = True
        return True

    def trainMyArc(self):
        # get feature_names names 
        
        self.arc.FuzzificationLayer.load_weights(self.data.df_name)
        self.arc.RuleConsequentLayer.load_weights(self.data.df_name)
        
        self.train()

    def train(self):
        """Calling trainer
        """
        # loading data, performing datapipeline and getting datasets
        self.data.load_data_for_training()
        self.train_ds = self.data.train_ds
        self.test_ds = self.data.test_ds
        self.validation_ds = self.data.validation_ds
        self.feature_names = self.data.feature_names
        # passing parameter names onto trainer
        self.trainer.arc = self.arc
        self.trainer.feature_ranges = self.data.feature_ranges 
       # print("here", self.data.feature_ranges)
        tf.keras.backend.clear_session()
        # trainig model
        self.trainer(self.train_ds,  self.test_ds, self.validation_ds)

    

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
            return sample_test_accuracy


    def evalutate(self):
        """Testing the model with the validation dataset
        """
        # results from using validation data
        eval_loss, eval_accuracy =  self.trainer.test(self.validation_ds)

        print("\nEvaluation")
        print(f'accuracy: {eval_loss}%')
        print(f'loss:     {eval_accuracy}%')

    def save_params(self):
        """
        """
        # save results to csv  
        save_path = os.path.dirname(__file__) +  '/../results'        
        file_name = str ("Params" + self.arc.Name + ".csv")
        completeName = os.path.join(save_path, file_name)
        self.result.to_csv(completeName)

    def summary(self):
        """Printing & Saving all important information of a model
        """
        
        data = self.data.summary()
        arc = self.arc.summary()
        trainer = self.trainer.summary()

        frames = [data, arc, trainer]
        self.result = pd.concat(frames)

        print(f'\n=================================================================\
                \n Network Architecure: {self.arc}  \
                \n {self.result}                         \
                \n=================================================================')

        self.save_params()
        
