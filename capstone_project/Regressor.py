import os
import sys
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from capstone_project import Analyzer

#in pipeline the dictionary #models.append(estimator, accuracy_score)


# fit_predict_score class
class FitPredictScore():
    def __init__(self, random_state: int, model: dict):
        self.random_state = random_state
        self.model = model
        #self.params = params()

    # way to see the results of multiple models and compare - to build out and plot the results
   # def create_model():
       # return None

    # Fit = function
    def fit(self, x_train: np.array, y_train: np.array) -> np.array:
        """ This function is to fit the data with x and y training arrays
              Args:
                x_train(np.array): The name of the x array to fit
                y_train(np.array): The name of the y array to fit

            Returns: model(np.array): The arrays after fit
        """
        self.model.fit(x_train, y_train)

    # Predict = function
    def predict(self, x: np.array) -> np.array:
        """ This function is to predict based on data within the given array
            Args:
                x(np.array): The name of the array to predict on

            Returns: model_predict(np.array): The prediction on the array data
        """
        model_predict = self.model.predict(x)
        return model_predict

    # Score = function
    def score(self, y_true: np.array, log_reg_predict: float) -> float:
        """ This function is to score based on data within the array, and append the model name and score for a comparison
            Args:
                y_predicted(np.array): The name of the array data predicted
                accuracy_score: the score based on the true vs predicted results

            Returns: accuracy_score: The score of the data
            """
            #### QUESTION - best way to also append the estimator/model name from the class?
        #y_predicted = self.predict(x)
        class_score = accuracy_score(y_true, log_reg_predict)
        return class_score

# Class of Classification Estimators =  any below are members

#class ClassificationEstimators(FitPredictScore):
 #   estimator = estimator
 #   def __init__(self):


class CustomLogiscticRegression(FitPredictScore):
    #estimator = "Logistic Regression"
    def __init__(self, random_state: int, params: dict):
        model = LogisticRegression(**params)
        super().__init__(random_state=random_state, model=model)

   #def create_model(self):
        """Performs classification with method of -> logistical regression

        Args:
            model(): The model with the selected classification method.

        Returns: model for the method -> logistical regression
        """
       #model = LogisticRegression(**self.params)
       #return model

class CustomKNN_Classifier(FitPredictScore):
    #estimator = "KNN"
    def __init__(self, n_neighbours: int, params: dict):
        model = KNeighborsClassifier(n_neighbours,**params)
        super().__init__(random_state=None,model=model)

    #def create_model(self):
        """Performs classification with method of -> logistical regression

        Args:
            model(): The model with the selected classification method.

        Returns: model for the method -> logistical regression
        """

        #neighbour = KNeighborsClassifier(n_neighbors = best_knn)


        #return model

class CustomDecisionTree(FitPredictScore):
    #estimator = "Decision Tree"
    def __init__(self, params: dict):
        model = DecisionTreeClassifier(**params)
        super().__init__(random_state=None, model=model)


    #def create_model(self):
        """Performs classification with method of -> logistical regression

        Args:
            model(): The model with the selected classification method.

        Returns: model for the method -> logistical regression
        """
        #model = decision_tree(params)
        #return model

class CustomRandomForest(FitPredictScore):
    #estimator = "Random Forest"
    def __init__(self, n_estimators: int, random_state: int, params: dict):
        model = RandomForestClassifier(n_estimators,**params)
        super().__init__(random_state=random_state, model=model)


    #def create_model(self):
        """Performs classification with method of -> logistical regression

        Args:
            model(): The model with the selected classification method.

        Returns: model for the method -> logistical regression
        """
        #model = random_forest(**params)
        #return model

class CustomSVC(FitPredictScore):
    #estimator = "SVC"
    def __init__(self, random_state: int, params: dict):
        model = SVC(**params)
        super().__init__(random_state=random_state, model=model)


   # def create_model(self):
        """Performs classification with method of -> logistical regression

        Args:
            model(): The model with the selected classification method.

        Returns: model for the method -> logistical regression
        """
        #model = svc(**params)
        #return model

class CustomANN_Classifier(FitPredictScore):
    #estimator = "ANN"
    def __init__(self, random_state: int, params: dict):
        model = MLPClassifier(**params)
        super().__init__(model=model, random_state=random_state)


   # def create_model(self):
        """Performs classification with method of -> logistical regression

        Args:
            model(): The model with the selected classification method.

        Returns: model for the method -> logistical regression
        """
        #model = MLPClassifier(**params)
        #return model


## OLD BELOW

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
