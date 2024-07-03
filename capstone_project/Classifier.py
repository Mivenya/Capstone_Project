import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier



# fit_predict_score class
class FitPredictScore():
    def __init__(self, random_state: int, model: dict):
        self.random_state = random_state
        self.model = model

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
                log_reg_predict(np.array): The name of the array data predicted
                accuracy_score: the score based on the true vs predicted results

            Returns: accuracy_score: The score of the data
            """
        log_reg_predict = self.predict(x)
        class_score = {}  
        class_score["Accuracy Score"] = accuracy_score(y_true, log_reg_predict)
        print(class_score)
        return class_score

# Class of Classification Estimators =  any below are members

#class ClassificationEstimators(FitPredictScore):
 #   estimator = estimator
 #   def __init__(self):


class CustomLogisticRegression(FitPredictScore):
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
