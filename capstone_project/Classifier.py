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

# Class of Classification Estimators =  any below are members 

class ClassificationEstimators:
    def __init__(self, df: pd.DataFrame, column_name: str):
        self.df = df.copy()

    
    def logistical_regression(self, column_name: str) -> pd.DataFrame: 
        """Performs classification with method of -> logistical regression

        Args:
            column_name (str): The name of the column to drop missing values from.

        Returns: Confusion matrix of accuracy score for model of the method -> logistical regression
        """
        
        return logistical_regression_accuracy_score()
                             
    def knn_classifier(self, column_name: str) -> pd.DataFrame: 
        """Performs classification with method of -> KNN classifier

        Args:
            column_name (str): The name of the column to drop missing values from.

        Returns: Confusion matrix of accuracy score for model of the method -> KNN classifier
        """
        
        return knn_classifier_accuracy_score()
                             
    def decision_tree(self, column_name: str) -> pd.DataFrame: 
        """Performs classification with method of -> decision tree

        Args:
            column_name (str): The name of the column to drop missing values from.

        Returns: Confusion matrix of accuracy score for model of the method -> decision tree
        """
        
        return decision_tree_accuracy_score()

    def random_forest(self, column_name: str) -> pd.DataFrame: 
        """Performs classification with method of -> random forest

        Args:
            column_name (str): The name of the column to drop missing values from.

        Returns: Confusion matrix of accuracy score for model of the method -> random forest
        """
        
        return random_forest_accuracy_score()
                             
    def svc(self, column_name: str) -> pd.DataFrame: 
        """Performs classification with method of -> SVC

        Args:
            column_name (str): The name of the column to drop missing values from.

        Returns: Confusion matrix of accuracy score for model of the method -> SVC
        """
        
        return svc_accuracy_score()
                             
    def ann_classifier(self, column_name: str) -> pd.DataFrame: 
        """Performs classification with method of -> artificial neural networks

        Args:
            column_name (str): The name of the column to drop missing values from.

        Returns: Confusion matrix of accuracy score for model of the method -> artificial neural networks
        """
        
        return ann_accuracy_score()
