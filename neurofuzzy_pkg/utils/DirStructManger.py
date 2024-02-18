# basics
import numpy as np
import matplotlib.pyplot as plt
import os 

"""
Collection of 
- functions that generate folders once a new dataset is loaded 

"""

def generate_folders(df_name):
    generate_folders_config(df_name)
    generate_folders_results(df_name)

def generate_folders_config(df_name):
    """
    Args: 
        df_name (str): name of loaded dataset
    
    Dir structure:
    ├── config 
    |   ├── [df_name]       <- name of given dataframe
    │           └── weights <- where weights of Fuzzification- and ConsequentLayer will be saved
    """
    relative_path = f"/../../config/{df_name}"
    save_path = os.path.dirname(__file__) + relative_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        save_path += "/weights"
        os.mkdir(save_path)
    print("f'Directory {df_name} created in config, full path is {save_path}'") 


def generate_folders_results(df_name):
    """
    Args: 
        df_name (str): name of loaded dataframe
        
    Dir structure:
    ├── results 
    |   ├── [df_name]       <- name of given dataframe
    │           └── figures <- MFs before and after training and performance of arc
    """
    relative_path = f"/../../results/{df_name}"
    save_path = os.path.dirname(__file__) + relative_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        save_path += "/figures"
        os.mkdir(save_path)
        save_path1 = save_path + "/before_training"
        os.mkdir(save_path1)
        save_path2 = save_path + "/after_training"
        os.mkdir(save_path2)
    print("f'Directory {df_name} created, full path is {save_path}'") 