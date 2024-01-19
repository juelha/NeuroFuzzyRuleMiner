# basics
#from jsonschema import draft201909_format_checker
import tensorflow as tf
import numpy as np
import pandas as pd
import os


class DataPipeline():
    """Loading and preparing the dataset for building and training respectively. 

    Attributes:
        train_ds (tf.PrefetchDataset): dataset for training
        test_ds (tf.PrefetchDataset): dataset for testing
        validation_ds (tf.PrefetchDataset): dataset for validating the model
        parameter_names (list(str)): column names of dataset, needed for extracting rules later
    """

    def __init__(self, df_name="dummy", batch_size=64):
        """Initializes DataPipeline()-Object

        Args:
            df_name (str): name of dataframe to be used
            batch_size (int): size of batch used for SGD 
        Note:
            split of dataset is 80:10:10
        """
        self.df_name = df_name
        self.batch_size = batch_size
        self.params = [self.batch_size]
        
        self.train_ds = None
        self.test_ds = None
        
        self.validation_ds = None
        self.feature_names = None


    def __call__(self):
        """Calling Object of Class will load data"""
        self.load_data()

    def summary(self):
        """Get parameters of class and return in pd.DataFrame
        
        Returns:
            df (pd.DataFrame): all params with names  
        """  
        column_names = ['batch_size']
        df = pd.DataFrame(self.params, column_names)
        return df
    
    def loader(self):
        """Load specified dataset
        
        Args:
            df_name (str): dataset to load

        Returns:
            df, targets
        """
        if self.df_name == None:
            print("no df specified")
        elif self.df_name == "wine":
            df, targets = self.load_wine()
        elif self.df_name == "dummy":
            df, targets = self.load_dummy()
        return df,targets

    def load_wine(self):
        """Loads wine quality dataset
        
        Returns:
            df, targets
        """
        # load from website 
        df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", 
        delimiter=";")
        # shuffle first so inputs and targets stay on same row
        df = df.sample(frac=1)
        # separate into input and targets 
        targets = df.pop('quality')
        self.feature_names = list(df.columns)
        # get max value of each feature <- center init
        self.feature_ranges = df.max()
        print(df.max())

        print("feature", self.feature_names)

        return df, targets

    def load_dummy(self):
        """Loads dummy dataset 
        
        """
        # get save path 
        file_name = 'dummy_df.csv'
        save_path = os.path.dirname(__file__) +  '/../data'
        full_path = os.path.join(save_path, file_name)
        assert  os.path.exists(full_path), f'File {file_name} not found'
        df = pd.read_csv(full_path)
        # shuffle first so inputs and targets stay on same row
        df = df.sample(frac=1)
        # separate into input and targets 
        targets = df.pop('out')
        print("df", df)
        self.feature_names = list(df.columns)
        print("feature", self.feature_names)

        # get mean of all cols
        self.inputs_mean = np.mean(df, axis=0)      
        return df, targets

    def load_data_for_training(self):
        """Initializes the datasets: train_ds, test_ds, validation_ds,
        which have been split from the original dataset using the ratio 
        = 80:10:10
        """
        
        df, targets = self.loader()
    
        

        # Split the dataset into a train, test and validation split
        # ratio is 80:10:10
        train_ds, test_ds, validation_ds = np.split(df, [int(.8*len(df)), int(.9*len(df))])
        train_tar, test_tar, validation_tar = np.split(targets, [int(.8*len(targets)), int(.9*len(targets))])

        # convert to tensor dataset
        df = tf.data.Dataset.from_tensor_slices((df.values, targets.values))

        # stitch inputs and targets back together
        training_ds = tf.data.Dataset.from_tensor_slices((train_ds, train_tar))
        testing_ds = tf.data.Dataset.from_tensor_slices((test_ds, test_tar))
        validating_ds = tf.data.Dataset.from_tensor_slices((validation_ds, validation_tar))

        # calculate treshhold
        self.treshhold = np.mean(targets)    

        # pipeline and one-hot encoding target vector
        self.train_ds = training_ds.apply(self.pipeline_for_training)
        self.test_ds = testing_ds.apply(self.pipeline_for_training)
        self.validation_ds = validating_ds.apply(self.pipeline_for_training)

    def pipeline_for_training(self, df):
        """Performs the needed operations to prepare the datasets for training

        Args:
            df (tf.data.Dataset): dataset to prepare
        
        Returns: 
            df (tf.PrefetchDataset): prepared dataset
        """
        # target is one-hot-encoded to have two outputs, representing two output perceptrons
        df = df.map(lambda features, target: (features, self.make_binary(target)))
        df = df.map(lambda inputs, target: (inputs, tf.one_hot(int(target), 2)))
        # cache this progress in memory
       # df = df.cache()
        # shuffle, batch, prefetch
        df = df.shuffle(50)
        df = df.batch(self.batch_size)
        df = df.prefetch(buffer_size=1)
        return df
    
    def load_data_for_building(self, df_name="dummy"):
        """
        one batch dataset, input and target 

        """

        df, targets = self.loader()
        
        # make targets binary 
        treshhold = np.mean(targets)        
        targets = targets.apply(lambda x: int(x >= treshhold))
        print(type(targets))
        print(targets.head())
        
        self.feature_names = list(df.columns)

        # get mean of all cols
        self.inputs_mean = np.mean(df, axis=0)      

        # convert to tensor dataset
        df = tf.data.Dataset.from_tensor_slices((df.values, targets.values))

        # pipeline and one-hot encoding target vector
        df = df.apply(self.pipeline_for_building)

        for features, targets in df.take(5):
          print ('Features: {}, Target: {}'.format(features, targets))

        self.inputs = df.map(lambda x,y: x)
        self.targets = df.map(lambda x,y: y) # doing weird shit  
        print("over here", df)

        return 0#self.inputs, self.targets

    def pipeline_for_building(self, df):
        """
        input: tensorflow dataset
        returns: preprocessed and prepared dataset
        """
        #df = df.map(lambda features, target: (features, self.make_binary(target)))
        # note: perfomance is better without converting to one_hot
        df = df.map(lambda inputs, target: (inputs, tf.one_hot(target,2)))
        df = df.shuffle(50)

        return df
    
    def make_binary(self,target):
        """
        is needed to make the non-binary classification problem binary
        input: the target to be simplified 
        returns: boolean 
        """
        # note: casting to integers lowers accuracy
        return(tf.expand_dims(int(target >= self.treshhold), -1))
        #return(tf.expand_dims(target >= self.treshhold, -1))