import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from capstone_project import Analyzer

analyzer.read_dataset



# fit_predict_score class
class FitPredictScore():
    def __init__(self, model: array, y_pred: arrray, score: int, accuracy_score: float):    
        self.model = self.model
        self.y_pred = self.y_pred
        self.score = self.score
        self.accuracy_score = self.accuracy_score

class Classifier(FitPredictScore):
# Fit = function
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame): 
        super().__init__(model=model)
        """ This function is to fit the dataset
                
                                Args:
                df (pd.Dataframe): The name of the dataframe to fit

            Returns: df (pd.Dataframe): The dataframe after fit
        """   
        self.model = model.fit(df(X_train, y_train))
            
    # Predict = function
    def predict(self, X_test: pd.DataFrame, y_pred: pd.array) -> pd.array: 
        super().__init__(model=model)
        """ This function is to predict based on data within the dataset
            
                                Args:
                df (pd.Dataframe): The name of the dataframe to predict

            Returns: predict(): The prediction based on data within the dataframe
            """       
        y_pred = model.predict(X_test)
        return y_pred
                
    # Score = function
    def score(self) -> accuracy_score: 
        super().__init__(model=model, y_pred=y_pred)
        """ This function is to score based on data within the dataset
            
                                Args:
                df (pd.Dataframe): The name of the dataframe containing data being scored

            Returns: score(): The score of the data
            """   
        #matrix = confusion_matrix(y_test,y_pred)
        accuracy_score = accuracy_scores(model, y_pred)
        return accuracy_score

# Class of Classification Estimators =  any below are members 

class ClassificationEstimators:
    def __init__(self, df: pd.DataFrame, column_name: str):
        self.df = self.df

    
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
