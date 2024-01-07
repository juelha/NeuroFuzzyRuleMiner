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

        # get feature_names names 
        self.feature_names = data.feature_names

        # architecture
        self.arc = arc
        self.arc.feature_names = self.feature_names

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
        self.arc.build(self.data.inputs, self.data.targets, self.data.inputs_mean)
        print("Build done")

    def build_MyArc_MF(self):
        # load data for building my arc
        self.data.load_data_for_building()
        self.arc.build_MFs(self.data.inputs, self.data.targets, self.data.inputs_mean)
        print("Build done")

    def build(self):
        """
        """
        # todo just do example datastet
        print("self.inputs_mean", self.inputs_mean)
        self.arc.build(self.inputs_mean, self.feature_names)
       # self.trainer.test(self.inputs_mean)
        self.built = True
        return True

    def trainMyArc(self):
        self.arc.FuzzificationLayer.load_weights()
        self.arc.RuleConsequentLayer.load_weights()
        self.train()

    def train(self):
        """Calling trainer
        """
        # loading data, performing datapipeline and getting datasets
        self.trainer.arc = self.arc

        self.data.load_data_for_training()
        self.train_ds = self.data.train_ds
        self.test_ds = self.data.test_ds
        self.validation_ds = self.data.validation_ds
        # passing parameter names onto trainer
        self.trainer.feature_names = self.data.feature_names 

        tf.keras.backend.clear_session()
        # trainig model
        self.trainer(self.train_ds,  self.test_ds, self.validation_ds)


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
        
