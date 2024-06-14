
import os
import sys
import pandas as pd
import numpy as np
from random import sample

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler


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
    def __init__(self, df: pd.DataFrame) -> pd.DataFrame:
        self.df_raw = df
        self.df = df.copy()

    
    def drop_missing_data(self, df: pd.DataFrame) -> pd.DataFrame: 
        """Drops rows with missing values in the specified column.

        Args:
            column_name (str): The name of the column to drop missing values from.

        Returns: pd.DataFrame: The DataFrame with missing values dropped.
        """
        self.df = self.df.dropna()
        return self.df
    
    def drop_column(self, column_names: str) -> pd.DataFrame:
        """Drop entire specified column
    
        Args:
        column_name (str): The name of the column to completely drop.
        
        Returns: pd.DataFrame: The DataFrame with the selected column removed
        """
        self.df = self.df.drop([column_names], axis=1)
        return self.df
    
    def encode_features(self, column_names: str) -> pd.DataFrame:
        """One hot encodes features of the specified column.

        Args:
            column_name (str): The name of the column to encode values from.

        Returns: pd.DataFrame: The DataFrame with features encoded.
        """
        enc = OneHotEncoder()
        self.df_onehot = pd.get_dummies(self.df[[column_names]], dtype=int)
        self.df = pd.concat([self.df, self.df_onehot], axis=1) 
        self.df = self.df.drop([column_names], axis=1)
        return self.df
    
    def encode_label(self, column_names: str) -> pd.DataFrame:
        """Label encodes in the specified column.

        Args:
            column_name (str): The name of the column to label encode values from.

        Returns: pd.DataFrame: The DataFrame with label encoded.
        """
        lbl_enc = LabelEncoder()
        self.df.loc[:, column_names] = lbl_enc.fit_transform(self.df[column_names])
        return self.df
    
    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize data in the dataset.

        Args:
            df (pd.DataFrame): The name of the dataset to standardize.

        Returns: pd.DataFrame: The DataFrame with standardization.
        """
        scaler = StandardScaler()
        self.df = scaler.fit_transform(self.df)
        return self.df

    def shuffle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shuffle the dataset

        Args:
            df (pd.DataFrame): The name of the dataset to shuffle.

        Returns: pd.DataFrame: The DataFrame shuffled.
        """
        shuffled_df = self.df.sample(n=len(self.df))
        return shuffled_df
    
    def sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get 50% sample (0.5 frac) of the dataset

        Args:
            df (pd.DataFrame): The name of the dataset to shuffle.

        Returns: 50% of the pd.DataFrame:
        """
        self.df = self.df.sample(frac=0.5, replace=True, random_state=1)
        return self.df
    
    def retrieve_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Retrieve data at it's current

        Args:
            df (pd.DataFrame): The name of the dataset to retrieve.

        retrieved current (transformed) verion of pd.DataFrame:
        """
        return self.df
    
if __name__== "__main__":

    #testing read dataset function
    absolute_path= 'E:/Repos/capstone_project/capstone_project/diamonds.csv'
    df = read_dataset(dataset_path=absolute_path)

    #testing describe function
    describe(dataset=df)
    
    #testing DataManipulation class
    my_data_manipulation = DataManipulation(df=df)

    #testing drop na
    data_df = my_data_manipulation.drop_missing_data(df)
    print(data_df)

    #testing drop column
    cleaned_df = my_data_manipulation.drop_column(column_names="Unnamed: 0")
    print(cleaned_df)

    #testing feature one hot encoder
    encoded_df = my_data_manipulation.encode_features(column_names="color")
    print(encoded_df)

    #testing label encoder
    lblencoded_df = my_data_manipulation.encode_label(column_names="cut")
    lblencoded_df = my_data_manipulation.encode_label(column_names="clarity")
    print(lblencoded_df)

    #testing label encoder
    scaled_df = my_data_manipulation.standardize(df=df)
    print(scaled_df)
    
    #testing dataset shuffle
    #shuffled_df = my_data_manipulation.shuffle(df=df)
    #print(shuffled_df)

    #testing dataset 50% sample
   # sampled_df = my_data_manipulation.sample(df=df)
    #print(sampled_df)

    #testing retrieve data
   # retrieved_df = my_data_manipulation.retrieve_data(df=df)
    #print(retrieved_df)

#Data Manipulation
#Data Visualization
# member output is dataframe as attribute






# Class 2 = Plotting and any below plots are members

