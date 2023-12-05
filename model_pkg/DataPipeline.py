# basics
#from jsonschema import draft201909_format_checker
import tensorflow as tf
import numpy as np
import pandas as pd
import os

# specifics
from tensorflow.keras import layers
from tensorflow.keras.optimizers import *


class DataPipeline():
    """
    The DataPipeline()-Class is responsible for:
    - loading the dataset
    - splitting it into datsets used for training
    - performing the datapipeline
    ___________________________________________________________
    """

    def __init__(self, n_batches=3, online=False):
        """Initializes DataPipeline()-Object
        Args:
            n_batches (int): number of batches of a dataset
        Note:
            split of dataset is 80:10:10
        Attributes:
            train_ds (tf.PrefetchDataset): dataset for training
            test_ds (tf.PrefetchDataset): dataset for testing
            validation_ds (tf.PrefetchDataset): dataset for validating the model
            parameter_names (list(str)): column names of dataset, needed for extracting rules later
        """

        self.n_batches = n_batches
        self.params = [self.n_batches]
        
        self.online = online

        self.train_ds = None
        self.test_ds = None
        
        self.validation_ds = None
        self.feature_names = None


    def __call__(self):
        """Calling Object of Class will load data 
        """
        self.load_data()
        return 0


    def summary(self):
        """Get parameters of class and return in pd.DataFrame
        
        Returns:
            df (pd.DataFrame): all params with names  
        """  
        column_names = ['n_batch']
        df = pd.DataFrame( self.params, column_names)
        return df


    def pipeline(self, ds):
        """Performs the needed operations to prepare the datasets for training
        Args:
            ds (tf.data.Dataset): dataset to prepare
        
        Returns: 
            ds (tf.PrefetchDataset): prepared dataset
        """
        # target is one-hot-encoded to have two outputs, representing two output perceptrons
        ds = ds.map(lambda inputs, target: (inputs, tf.one_hot(int(target), 2)))
        # cache this progress in memory
        ds = ds.cache()
        # shuffle, batch, prefetch
        ds = ds.shuffle(50)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=1)
        return ds


    def load_data(self):
        """Initializes the datasets: train_ds, test_ds, validation_ds,
        which have been split from the original dataset using the ratio 
        = 80:10:10
        """

        if self.online:
            df_merged = pd.read_csv('https://raw.githubusercontent.com/juelha/IANNWTF_FINAL/main/data/df_merged.csv?token=GHSAT0AAAAAABOICJN2K2HDESZ7CSHXL2MGYSTLCOA')
            df_merged_aug = pd.read_csv('https://raw.githubusercontent.com/juelha/IANNWTF_FINAL/main/data/df_merged_aug.csv?token=GHSAT0AAAAAABOICJN3CABVYOG5H2KZMVBQYSTLDHQ')

        else:
            # import original and augmented citrus data sets
            # get save path 
            file_name = 'df_merged' + '.csv'
            file_name_aug = 'df_merged' + '.csv'
            save_path = os.path.dirname(__file__) +  '/../data'
            full_path = os.path.join(save_path, file_name)
            full_path_aug = os.path.join(save_path, file_name_aug)

            assert  os.path.exists(full_path), f'File {file_name} is not downloaded to correct folder'
            assert  os.path.exists(full_path_aug), f'File {file_name_aug} is not downloaded to correct folder'

            df_merged = pd.read_csv(full_path)
            df_merged_aug = pd.read_csv(full_path_aug)
            

        # create features and labels
        df_merged_features = df_merged.copy()
        targets = df_merged_features.pop('yield')

        # df_merged_features.pop('Unnamed: 0')
        df_merged_features.pop('region')
        df_merged_features.pop('season')

        df_merged_aug.pop('season')
        # merge augmented to the rest
        frames = [df_merged, df_merged_aug]
        df_merged = pd.concat(frames)
        
        self.batch_size = 32# int(len(df_merged)/self.n_batches)
        self.feature_names = df_merged_features.columns

        inputs = {}    # Building a set of symbolic keras.Input objects

        # iterating over the columns and names
        for name, column in df_merged_features.items():
            # check data type
            dtype = column.dtype
            # matching the names and data-types of the CSV columns
            if dtype == object:
                dtype = tf.string
            else:
                dtype = tf.float32
            inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

        numeric_inputs = {name: input for name, input in inputs.items()
                        if input.dtype == tf.float32}
        x = layers.Concatenate()(list(numeric_inputs.values()))
        all_numeric_inputs = x

        # Collecting the symbolic preprocessing results, to concatenate them later.
        preprocessed_inputs = [all_numeric_inputs]

        # concatenate all the preprocessed inputs together
        preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)
        # build a model that handles the input preprocessing
        df_merged_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)
        # Convert the data set to a dictionary of tensors:
        df_merged_features_dict = {name: np.array(value) for name, value in df_merged_features.items()}
        # Slice out the first training example
        # features_dict = {name: values[:1] for name, values in df_merged_features_dict.items()}
        # pass the encoded features to the preprocessing model
        encoded_features = df_merged_preprocessing(df_merged_features_dict)
        # convert the dictionary to a numpy array
        feature_array = encoded_features.numpy()

        # get mean of all cols
        print(feature_array)
        self.inputs_mean = np.mean(feature_array, axis=0)
        print(self.inputs_mean)


        # Split the dataset into a train, test and validation split
        # ratio is 80:10:10
        train_ds, test_ds, validation_ds = np.split(feature_array, [int(.8*len(feature_array)), int(.9*len(feature_array))])
        train_tar, test_tar, validation_tar = np.split(targets, [int(.8*len(targets)), int(.9*len(targets))])

        # convert to tensor dataset
        training_ds = tf.data.Dataset.from_tensor_slices((train_ds, train_tar))
        testing_ds = tf.data.Dataset.from_tensor_slices((test_ds, test_tar))
        validating_ds = tf.data.Dataset.from_tensor_slices((validation_ds, validation_tar))

        # pipeline and one-hot encoding target vector
        self.train_ds = training_ds.apply(self.pipeline)
        self.test_ds = testing_ds.apply(self.pipeline)
        self.validation_ds = validating_ds.apply(self.pipeline)