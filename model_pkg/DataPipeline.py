# basics
#from jsonschema import draft201909_format_checker
import tensorflow as tf
import numpy as np
import pandas as pd
import os
#from neurofuzzy_pkg.utils.DirStructManger import generate_folders


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
                           options: ["dummy", "wine"]

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
        return df, targets

    def load_dummy(self):
        """Loads dummy dataset 
        
        'pandas.core.frame.DataFrame'
        """
        # get save path 
        file_name = 'dummy_df.csv'
        save_path = os.path.dirname(__file__) +  '/../data'
        full_path = os.path.join(save_path, file_name)
        assert  os.path.exists(full_path), f'File {file_name} not found'
        df = pd.read_csv(full_path)
        # shuffle first so inputs and targets stay on same row
        df = df.sample(frac=1) # do we need to shuffle here? 
        # separate into input and targets 
        targets = df.pop("out")
        # get featuer names <- documenting MFs
        self.feature_names = list(df.columns)
        # get max value of each feature <- center init        
        self.feature_ranges = df.max()
        print(f"Dataset {self.df_name} loaded: \n {df.head()} \n")
        return df, targets

    def load_data_for_building(self):
        """
        one batch dataset, input and target 

        """
        df, targets = self.loader()
        self.generate_folders(self.df_name) 
        # make targets binary 
        treshhold = np.mean(targets)        
       # targets = targets.apply(lambda x: int(x >= treshhold))

        # one hot encoding
        depth = 2
        b = tf.one_hot(targets, depth)
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
        self.generate_folders(self.df_name) 

        # one hot encoding
        depth = 2
        b = tf.one_hot(targets, depth)
        targets = b.numpy()

        df = df.to_numpy()

        # Split the dataset into a train, test and validation split
        # ratio is 80:10:10
        train_ds, test_ds, validation_ds = np.split(df, [int(.8*len(df)), int(.9*len(df))])
        train_tar, test_tar, validation_tar = np.split(targets, [int(.8*len(targets)), int(.9*len(targets))])

        self.train_ds = (train_ds, train_tar)
        self.test_ds = (test_ds, test_tar)# test_ds.apply(self.pipeline_for_training)
        self.validation_ds = (validation_ds, validation_tar)#validation_ds.apply(self.pipeline_for_training)

    def pipeline_for_training(self, df):
        """Performs the needed operations to prepare the datasets for training

        Args:
            df (tf.data.Dataset): dataset to prepare
        
        Returns: 
            df (tf.PrefetchDataset): prepared dataset
        """
        # target is one-hot-encoded to have two outputs, representing two output perceptrons
      #  df = df.map(lambda features, target: (features, self.make_binary(target)))
       # df = df.map(lambda inputs, target: (inputs, tf.one_hot(int(target), 2)))
        # cache this progress in memory
       # df = df.cache()
        # shuffle, batch, prefetch
        df = df.shuffle(50)
        df = df.batch(self.batch_size)
        df = df.prefetch(buffer_size=1)
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
    
    def generate_folders(self, df_name):
        self.generate_folders_config(df_name)
        self.generate_folders_results(df_name)

    def generate_folders_config(self, df_name):
        """
        Args: 
            df_name (str): name of loaded dataset
        
        ├── config 
        |   ├── [df_name]       <- name of given dataframe
        │           └── weights <- where weights of Fuzzification- and ConsequentLayer will be saved
        """
        relative_path = f"/../config/{df_name}/weights/"
        save_path = os.path.dirname(__file__) + relative_path
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            print(f"Directory {df_name} created in config, full path is {save_path}\n") 


    def generate_folders_results(self, df_name):
        """
        Args: 
            df_name (str): name of loaded dataframe
        
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
            save_path_fig_1 = save_path + "/before_training"
            os.mkdir(save_path_fig_1)
            save_path_fig2 = save_path + "/after_training"
            os.mkdir(save_path_fig2)
            print(f'Directory {df_name} created, full path is {save_path}\n') 