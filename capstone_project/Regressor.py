import os
import sys
import pandas as pd
import numpy as np

from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

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
    def score(self, y_true: np.array, x: np.array) -> float:
        """ This function is to score based on data within the array, and append the model name and score for a comparison
            Args:
                y_predicted(np.array): The name of the array data predicted
                accuracy_score: the score based on the true vs predicted results

            Returns: accuracy_score: The score of the data
            """
        log_reg_predict = self.predict(x)
        reg_score = {}   
        reg_score["R2 Score"] = r2_score(y_true, log_reg_predict)
        regress_r2_score = r2_score(y_true, log_reg_predict)
        regress_mean_score = mean_squared_error(y_true, log_reg_predict)
        regress_root_mean_score = root_mean_squared_error(y_true, log_reg_predict)
        regress_mean_absolute_score = mean_absolute_error(y_true, log_reg_predict)
        print(reg_score)
        return reg_score
# Class of Classification Estimators =  any below are members

#class ClassificationEstimators(FitPredictScore):
 #   estimator = estimator
 #   def __init__(self):


class CustomLinearRegression(FitPredictScore):
    #estimator = "Logistic Regression"
    def __init__(self, random_state: int, params: dict):
        model = LinearRegression(**params)
        super().__init__(random_state=random_state, model=model)

   #def create_model(self):
        """Performs classification with method of -> logistical regression

        Args:
            model(): The model with the selected classification method.

        Returns: model for the method -> logistical regression
        """
       #model = LogisticRegression(**self.params)
       #return model

class CustomKNN_Regressor(FitPredictScore):
    #estimator = "KNN"
    def __init__(self, n_neighbours: int, params: dict):
        model = KNeighborsRegressor(n_neighbours,**params)
        super().__init__(random_state=None,model=model)

    #def create_model(self):
        """Performs classification with method of -> logistical regression

        Args:
            model(): The model with the selected classification method.

        Returns: model for the method -> logistical regression
        """

        #neighbour = KNeighborsClassifier(n_neighbors = best_knn)


        #return model

class CustomDecisionTreeReg(FitPredictScore):
    #estimator = "Decision Tree"
    def __init__(self, params: dict):
        model = DecisionTreeRegressor(**params)
        super().__init__(random_state=None, model=model)


    #def create_model(self):
        """Performs classification with method of -> logistical regression

        Args:
            model(): The model with the selected classification method.

        Returns: model for the method -> logistical regression
        """
        #model = decision_tree(params)
        #return model

class CustomRandomForestReg(FitPredictScore):
    #estimator = "Random Forest"
    def __init__(self, n_estimators: int, random_state: int, params: dict):
        model = RandomForestRegressor(n_estimators,**params)
        super().__init__(random_state=random_state, model=model)


    #def create_model(self):
        """Performs classification with method of -> logistical regression

        Args:
            model(): The model with the selected classification method.

        Returns: model for the method -> logistical regression
        """
        #model = random_forest(**params)
        #return model

class CustomSVR(FitPredictScore):
    #estimator = "SVC"
    def __init__(self, random_state: int, params: dict):
        model = SVR(**params)
        super().__init__(random_state=random_state, model=model)


   # def create_model(self):
        """Performs classification with method of -> logistical regression

        Args:
            model(): The model with the selected classification method.

        Returns: model for the method -> logistical regression
        """
        #model = svc(**params)
        #return model

class CustomANN_Regressor(FitPredictScore):
    #estimator = "ANN"
    def __init__(self, random_state: int, params: dict):
        model = MLPRegressor(**params)
        super().__init__(model=model, random_state=random_state)


   # def create_model(self):
        """Performs classification with method of -> logistical regression

        Args:
            model(): The model with the selected classification method.

        Returns: model for the method -> logistical regression
        """
        #model = MLPClassifier(**params)
        #return model

