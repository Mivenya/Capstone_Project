import os
import sys
import pandas as pd
import numpy as np

# Fit = function
def fit(dataset_path: str) -> pd.DataFrame: 
    """ This function is to fit the dataset
        
                             Args:
            df (pd.Dataframe): The name of the dataframe to fit

        Returns: df (pd.Dataframe): The dataframe after fit
        """   
    dataset = pd.read_csv(filepath_or_buffer=dataset_path)
    return dataset

# Predict = function
def predict(dataset_path: str) -> pd.DataFrame: 
    """ This function is to predict based on data within the dataset
        
                             Args:
            df (pd.Dataframe): The name of the dataframe to predict

        Returns: predict(): The prediction based on data within the dataframe
        """       
    return predict()
              
# Score = function
def score(dataset_path: str) -> pd.DataFrame: 
    """ This function is to score based on data within the dataset
        
                             Args:
            df (pd.Dataframe): The name of the dataframe containing data being scored

        Returns: score(): The score of the data
        """   
    return score()

# Class of Regression Estimators =  any below are members 

class RegressionEstimators:
    def __init__(self, df: pd.DataFrame, column_name: str):
        self.df = df.copy()

    
    def linear_regression(self, column_name: str) -> pd.DataFrame: 
        """Performs regression with method of -> linear regression

        Args:
            column_name (str): The name of the column to drop missing values from.

        Returns: R2_score, Mean Squared error, Root Mean Squared Error, and Mean Absolute error of the regression model of the method -> linear regression
        """
        
        return linear_regression_model()
                             
    def knn_regression(self, column_name: str) -> pd.DataFrame: 
        """Performs regression with method of -> KNN regression

        Args:
            column_name (str): The name of the column to drop missing values from.

        Returns: R2_score, Mean Squared error, Root Mean Squared Error, and Mean Absolute error of the regression model of the method -> KNN regression
        """
        
        return knn_regression_model()
                             
    def decision_tree(self, column_name: str) -> pd.DataFrame: 
        """Performs regression with method of -> decision tree

        Args:
            column_name (str): The name of the column to drop missing values from.

        Returns: R2_score, Mean Squared error, Root Mean Squared Error, and Mean Absolute error of the regression model of the method -> decision tree
        """
        
        return decision_tree_model()

    def random_forest(self, column_name: str) -> pd.DataFrame: 
        """Performs regression with method of -> random forest

        Args:
            column_name (str): The name of the column to drop missing values from.

        Returns: R2_score, Mean Squared error, Root Mean Squared Error, and Mean Absolute error of the regression model of the method -> random forest
        """
        
        return random_forest_model()
                             
    def svr(self, column_name: str) -> pd.DataFrame: 
        """Performs regression with method of -> SVR

        Args:
            column_name (str): The name of the column to drop missing values from.

        Returns: R2_score, Mean Squared error, Root Mean Squared Error, and Mean Absolute error of the regression model of the method -> SVC
        """
        
        return svr_model()
                             
    def ann_regression(self, column_name: str) -> pd.DataFrame: 
        """Performs regression with method of -> artificial neural networks

        Args:
            column_name (str): The name of the column to drop missing values from.

        Returns: R2_score, Mean Squared error, Root Mean Squared Error, and Mean Absolute error of the regression model of the method -> artificial neural networks
        """
        
        return ann_regression_model()
