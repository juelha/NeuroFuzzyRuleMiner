# basics
#from jsonschema import draft201909_format_checker
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from neurofuzzy_pkg.utils.DirStructManger import generate_folders


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
        self.df_names =  ['dummy2', 'dummy3', 'dummy4', 'xor', 'iris']
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
                           options: ['dummy2', 'dummy3', 'dummy4', 'xor', 'iris']

        Returns:
            df, targets ('pandas.core.frame.DataFrame')
        
        Raises:
            ValueError: if no df_name was given
            
        Note: assumes last column of df is the target vector
        """
        # check if str is in possible datasets
        
        if self.df_name not in self.df_names:
            raise ValueError(
                f"Valid values for df name are {self.df_names}.")
    
        file_name = self.df_name + '_df.csv'
        save_path = os.path.dirname(__file__) +  '/../data'
        full_path = os.path.join(save_path, file_name)
        assert  os.path.exists(full_path), f'File {file_name} not found'
        df = pd.read_csv(full_path)
        # shuffle first so inputs and targets stay on same row
        df = df.sample(frac=1) # do we need to shuffle here? 
        # separate into input and targets 
        targets = df.pop(df.columns[-1]) # always use last column
        # get featuer names <- documenting MFs
        self.feature_names = list(df.columns)
        # get max value of each feature <- center init        
        self.feature_maxs = df.max()
        self.feature_mins = df.min()
        self.n_features = len(self.feature_names)
        self.n_classes = len(np.unique(targets))
      #  print(f"Dataset {self.df_name} loaded: \n {df.head()} \n")

        return df,targets
    
    

    def load_data_for_building(self):
        """
        one batch dataset, input and target 

        """
        df, targets = self.loader()
        self.generate_folders(self.df_name) 

        b = tf.one_hot(targets, self.n_classes)
        targets = b.numpy()

        self.inputs = df.to_numpy()# df.map(lambda x,y: x)
        self.targets = targets #df.map(lambda x,y: y) # doing weird shit  
       # print("over here", df)

        return 0#self.inputs, self.targets


    def load_data_for_training(self):
        """Initializes the datasets: train_ds, test_ds, validation_ds,
        which have been split from the original dataset using the ratio 
        = 80:10:10
        """
        df, targets = self.loader()
        # calculate treshhold
       # self.treshhold = np.mean(targets)  
        #generate_folders(self.df_name) 

        # one hot encoding
        b = tf.one_hot(targets, self.n_classes)
        targets = b.numpy()

        df = df.to_numpy()

        # Split the dataset into a train, test and validation split
        # ratio is 80:10:10
        train_ds, test_ds = np.split(df, [int(.8*len(df))])#, int(.9*len(df))])
        train_tar, test_tar = np.split(targets, [int(.8*len(targets))])#, int(.9*len(targets))])

        self.train_ds = (train_ds, train_tar)
        self.test_ds = (test_ds, test_tar)# test_ds.apply(self.pipeline_for_training)
        #self.validation_ds = (validation_ds, validation_tar)#validation_ds.apply(self.pipeline_for_training)


    def generate_folders(self,df_name):
        self.generate_folders_config(df_name)
        self.generate_folders_results(df_name)

    def generate_folders_config(self,df_name):
        """
        Args: 
            df_name (str): name of loaded dataset
        
        Dir structure:
        ├── config 
        |   ├── [df_name]       <- name of given dataframe
        │           └── weights <- where weights of Fuzzification- and ConsequentLayer will be saved
        """
        relative_path = f"/../config/{df_name}"
        save_path = os.path.dirname(__file__) + relative_path
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            save_path += "/weights"
            os.mkdir(save_path)
        print(f'Directory {df_name} created in config, full path is {save_path}') 


    def generate_folders_results(self,df_name):
        """
        Args: 
            df_name (str): name of loaded dataframe
            
        Dir structure:
        ├── results 
        |   ├── [df_name]       <- name of given dataframe
        │           └── figures <- MFs before and after training and performance of arc
        """
        relative_path = f"/../results/{df_name}"
        save_path = os.path.dirname(__file__) + relative_path
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            save_path += "/figures"
            os.mkdir(save_path)
            save_path1 = save_path + "/before_training"
            os.mkdir(save_path1)
            save_path2 = save_path + "/after_training"
            os.mkdir(save_path2)
        print(f'Directory {df_name} created in results, full path is {save_path}') 