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
from sklearn.model_selection import train_test_split

from capstone_project import Analyzer

#in pipeline the dictionary #models.append(estimator, accuracy_score)


# fit_predict_score class
class FitPredictScore():
    def __init__(self, random_state: int, model: dict, params: dict):
        self.random_state = random_state
        self.model = self.create_model()
        self.params = params()

    # way to see the results of multiple models and compare - to build out and plot the results
    def create_model():
        return None

    # Fit = function
    def fit(self, x_train: np.array, y_train: np.array, parameters: dict) -> np.array:
        """ This function is to fit the data with x and y training arrays
              Args:
                x_train(np.array): The name of the x array to fit
                y_train(np.array): The name of the y array to fit

            Returns: model(np.array): The arrays after fit
        """
        self.model.fit(x_train, y_train, **parameters)

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


class LogiscticRegression(FitPredictScore):
    #estimator = "Logistic Regression"
    def __init__(self, random_state: int, params: dict):
        super().__init__(random_state=random_state, params=params)

    def create_model(self):
        """Performs classification with method of -> logistical regression

        Args:
            model(): The model with the selected classification method.

        Returns: model for the method -> logistical regression
        """
        model = logistic_regression(**self.params)
        return model

class KNN_Classifier(FitPredictScore):
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

class DecisionTree(FitPredictScore):
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

class RandomForest(FitPredictScore):
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

class SVC(FitPredictScore):
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

class ANN_Classifier(FitPredictScore):
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


if __name__ == "__main__":
   # df = analyzer.read_dataset()





    absolute_path= 'E:/Repos/capstone_project/capstone_project/diamonds.csv'
    #absolute_path= 'C:/Users/hyppi/Repos/capstone_project/capstone_project/diamonds.csv'
    df = Analyzer.read_dataset(dataset_path=absolute_path)

    #def score(y_true: np.array, y_predicted: np.array) -> float: # non member function version
#      accuracy_score = accuracy_score(y_true, y_predicted)
#      return accuracy_score

#Before fitting and training we need to split the data

# train is now 80% of the entire data set
    x_train, x, y_train, y = train_test_split(x, y, random_state=0, test_size=0.2)

# test is now 10% of the initial data set
# validation is now 10% of the initial data set
    x_val, x, y_val, y = train_test_split(x, y, test_size=.5)

   #testing Logistic Regression
    params = {
    "criterion": "lbgs",
    "solver": 100
        }

    logistic_regression = LogisticRegression(params=params)

    #testing KNN Classifier
    params = {
        "criterion": "lbgs",
        "solver": 100
    }

    knn_classifier = KNeighborsClassifier(params=params)

    #testing Decision Tree
    params = {
    "gamma":"auto"
    }

    decision_tree = DecisionTreeClassifier(params=params)

    #testing Random Forest
    params = {
    "gamma":"auto"
    }

    random_forest = RandomForestClassifier(params=params)

    #testing SVC
    params = {
    "gamma":"auto"
    }

    svc = SVC(params=params)

    #testing ANN
    params = {
    "max_iter": 1
    }
    ann_classifier = MLPClassifier(params=params)

    #Plot of end results
    #plt.figure(figsize=(12,8))
    #plt.ylim(.5, 1)
    #sns.barplot(x=estimator, y= accuracy_score)