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
    """    
    The Model()-Class is combinating all parts needed to make a model:
    - datasets (train, test, evaluation)
    - an architecture
    - a trainer
    ___________________________________________________________
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
        # load data for building my arc
        self.data.load_data_for_building()
        self.inputs = data.inputs
        self.targets = data.targets
        # loading data, performing datapipeline and getting datasets
        self.data.load_data_for_training()
        self.train_ds = data.train_ds
        self.test_ds = data.test_ds
        self.validation_ds = data.validation_ds
        # get feature_names names 
        self.feature_names = data.feature_names

        # architecture
        self.arc = arc
        self.arc.feature_names = self.feature_names

        # trainer
        self.trainer = trainer
        # passing arc onto trainer
        self.trainer.arc = self.arc
        # passing parameter names onto trainer
        trainer.feature_names = self.feature_names 
        trainer.inputs_mean = data.inputs_mean 

        # for building 
        self.inputs_mean = data.inputs_mean 
        self.built = False


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
        



    def save_params(self):
        """
        """
        
        # save results to csv  
        save_path = os.path.dirname(__file__) +  '/../results'        
        file_name = str ("Params" + self.arc.Name + ".csv")
        completeName = os.path.join(save_path, file_name)
        self.result.to_csv(completeName)


    def run(self):
        """
        """
        self.build()
        # asset ? 
        self.train()
        self.summary()

    def build_MyArc(self):
      # todo just do example datastet
      #  print("self.inputs_mean", self.inputs_mean)
        self.arc.build(self.inputs, self.targets, self.inputs_mean)
       # self.trainer.test(self.inputs_mean)
        self.built = True
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
        self.trainer.arc = self.arc

        self.train()

    def train(self):
        """Calling trainer
        """
        tf.keras.backend.clear_session()
        # trainig model
        self.trainer(self.train_ds,  self.test_ds, self.validation_ds)


    def evalutate(self):
        """Testing the model with the validation dataset
        """
        # results from using validation data
        test_loss, test_accuracy =  self.trainer.test(self.validation_ds)

        print("\nEvaluation")
        print(f'accuracy: {test_accuracy}%')
        print(f'loss:     {test_loss}%')



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


    def run_backwards(self):
        """Inversion of a model 
        """
        self.train()

        target = tf.convert_to_tensor([[1] * 11],dtype=tf.float32)
        print("target       ", target)

        best_input  =  self.arc.invert(target)
        print("best_input    ", best_input)

        # test best_input
        prediction = self.arc(best_input)


        print("prediciton from best input: ", prediction)
        print("loss of best input:   ", 1-prediction)


        save_path = os.path.dirname(__file__) +  '/../results'
        file_name = "results.csv"
        completeName = os.path.join(save_path, file_name)
        new_file = open(completeName, "a")

        #tf.io.write_file(file_name, output)
        text =  str(best_input)
        new_file.write(text)

        new_file.close()


