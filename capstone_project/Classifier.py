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
    def score(self, x: np.array, y_true: np.array) -> float:
        """ This function is to score based on data within the array, and append the model name and score for a comparison
            Args:
                y_predicted(np.array): The name of the array data predicted
                accuracy_score: the score based on the true vs predicted results

            Returns: accuracy_score: The score of the data
            """
            #### QUESTION - best way to also append the estimator/model name from the class?
        y_predicted = self.predict(x)
        accuracy_score = accuracy_score(y_true, y_predicted)
        return accuracy_score

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
       # model = LogisticRegression(**self.params)
       # return model

class CustomKNN_Classifier(FitPredictScore):
    #estimator = "KNN"
    def __init__(self, random_state: int, params: dict):
        super().__init__(random_state=random_state, params=params)

    def create_model(self):
        """Performs classification with method of -> logistical regression

        Args:
            model(): The model with the selected classification method.

        Returns: model for the method -> logistical regression
        """
        model = knn_classifier(**self.params)
        return model

class CustomDecisionTree(FitPredictScore):
    #estimator = "Decision Tree"
    def __init__(self, random_state: int, params: dict):
        super().__init__(random_state=random_state, params=params)


    def create_model(self):
        """Performs classification with method of -> logistical regression

        Args:
            model(): The model with the selected classification method.

        Returns: model for the method -> logistical regression
        """
        model = decision_tree(**self.params)
        return model

class CustomRandomForest(FitPredictScore):
    #estimator = "Random Forest"
    def __init__(self, random_state: int, params: dict):
        super().__init__(random_state=random_state, params=params)


    def create_model(self):
        """Performs classification with method of -> logistical regression

        Args:
            model(): The model with the selected classification method.

        Returns: model for the method -> logistical regression
        """
        model = random_forest(**self.params)
        return model

class CustomSVC(FitPredictScore):
    estimator = "SVC"
    def __init__(self, random_state: int, params: dict):
        super().__init__(random_state=random_state, params=params)


    def create_model(self):
        """Performs classification with method of -> logistical regression

        Args:
            model(): The model with the selected classification method.

        Returns: model for the method -> logistical regression
        """
        model = svc(**self.params)
        return model

class CustomANN_Classifier(FitPredictScore):
    estimator = "ANN"
    def __init__(self, random_state: int, params: dict):
        super().__init__(random_state=random_state, params=params)


    def create_model(self):
        """Performs classification with method of -> logistical regression

        Args:
            model(): The model with the selected classification method.

        Returns: model for the method -> logistical regression
        """
        model = ann_classifier(**self.params)
        return model
