# basics
import tensorflow as tf
import numpy as np
import pandas as pd
# specifics
from tensorflow.keras.optimizers import *
import os.path

# custom
from model_pkg import *
import neurofuzzy_pkg.utils.MFs as MFs
from neurofuzzy_pkg.utils.WeightManager import load_weights


class Model():
    """Masterclass for combining data, architecture and trainer.
    """

    def __init__(self, data, arc, trainer, builder=None, classifier=None):
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
        self.data = data
        self.arc = arc
        self.trainer = trainer
        self.builder = builder
        self.classifier = classifier

    def run(self):
        """
        """
        self.build()
        self.train()
        self.summary()

    def build_MyArc(self):
        # load data for building my arc
        self.data.load_data_for_building()
        self.builder.arc = self.arc#
        #self.arc.RuleAntecedentLayer.n_features = self.data.n_features
       # self.arc.RuleConsequentLayer.n_classes = self.data.n_classes
        
        self.builder(self.data.inputs, self.data.targets, self.data.feature_maxs, self.data.feature_mins, self.data.df_name, self.arc.n_mfs)

        print("Build done")

    def build_MyArc_CW(self):
        # load data for building my arc
        self.data.load_data_for_building()
       # self.feature_names = self.data.feature_names
        #self.arc.feature_names = self.feature_names
        #self.arc.feature_ranges = self.data.feature_ranges
        self.arc.build_classweights(self.data.inputs, self.data.targets, self.data.df_name)
        print("Build done")

    def build_MyArc_MF(self):
        # load data for building my arc
        self.data.load_data_for_building()
        self.builder.arc = self.arc
        self.arc.build_MFs(self.data.feature_ranges, self.data.df_name)
        print("Build done")

    def class_acc(self):
       
        self.classifier.arc = self.arc
        return self.classifier.get_class_accuracy(self.data.inputs, self.data.targets, self.data.df_name)



    def trainMyArc(self):
        # get feature_names names 
        load_weights(self.arc.FuzzificationLayer, "centers", self.data.df_name)
        load_weights(self.arc.FuzzificationLayer, "widths", self.data.df_name)
        load_weights(self.arc.RuleConsequentLayer, "class_weights", self.data.df_name)
     #   self.arc.FuzzificationLayer.load_weights(self.data.df_name)
      #  self.arc.RuleConsequentLayer.load_weights(self.data.df_name)
        self.data.load_data_for_training() # doubled ! hc
        self.trainer.builder = self.builder
        MFs.visuMFs(self.arc.FuzzificationLayer, df_name= self.data.df_name, dir="before_training", max_vals=self.data.feature_maxs, min_vals= self.data.feature_mins, mf_names=self.arc.fuzzy_labels )
        self.trainer.max_vals = self.data.feature_maxs
        self.trainer.min_vals = self.data.feature_mins
        self.trainer.n_mfs = self.arc.n_mfs
      #  print("???", self.data.feature_ranges)
       # print("!!!", self.trainer.max_vals)
        self.train()
        MFs.visuMFs(self.arc.FuzzificationLayer, df_name=self.data.df_name, dir="after_training", max_vals=self.data.feature_maxs,  min_vals= self.data.feature_mins,mf_names=self.arc.fuzzy_labels )
        #self.arc.FuzzificationLayer.save_weights(self.data.df_name)


    def train(self):
        """Calling trainer
        """
        # loading data, performing datapipeline and getting datasets
        self.data.load_data_for_training()
        self.trainer.arc = self.arc
       # tf.keras.backend.clear_session()
        # trainig model
        self.trainer(self.data.train_ds,  self.data.test_ds, self.data.validation_ds)
        # saving figs after training
        self.trainer.visualize_training(df_name=self.data.df_name, type_model=self.arc.Name)

    

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
            input = tf.reshape(input,(len(self.data.feature_names),))

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
        
