
import os
import sys
import pandas as pd
import numpy as np


# Read dataset = function
def read_dataset(dataset_path: str) -> pd.DataFrame: 
    """ This function is to read the dataset
    """
    dataset = pd.read_csv(filepath_or_buffer=dataset_path)
    return dataset

# Describe dataset = function
def describe(dataset: pd.DataFrame): 
    """ This function is to describe the dataset
    """
    print(dataset.describe()) 
    
# Class 1 = Data Manipulation any below are members

class DataManipulation:
    def __init__(self, df: pd.DataFrame, column_name: str):
        self.df = df.copy()

    
    def drop_missing_data(self, column_name: str) -> pd.DataFrame: 
        """Drops rows with missing values in the specified column.

        Args:
            column_name (str): The name of the column to drop missing values from.

        Returns: pd.DataFrame: The DataFrame with missing values dropped.
        """
        self.df = self.df.dropna(subset=[column_name])
        return self.df





if __name__== "__main__":

    absolute_path= 'E:/Repos/capstone_project/capstone_project/diamonds.csv'
    df = read_dataset(dataset_path=absolute_path)
    describe(dataset=df)
    new_dataset = DataManipulation(column_name="Unnamed: 0")
    #print(drop_missing_data(column_name="Unnamed: 0"))
    

#Data Manipulation
#Data Visualization
# member output is dataframe as attribute






# Class 2 = Plotting and any below plots are members

