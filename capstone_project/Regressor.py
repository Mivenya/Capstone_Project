import numpy as np

from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor



# fit_predict_score class adjusting score for regression
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
        """ This function is to score based on data within the array with 4 metrics, and list all 4 metrics in a dictionary
            
            Args:
                x(np.array): The name of the array for testing/prediction
                log_reg_predict(np.array): The result of the prediction to compare with y_true values
                y_true(np.array): The name of the array of true values for comparison with prediction
                r2_score(float): Score of this metric name
                mean_squared_error(float): Score of this metric name
                root_mean_squared_error(float): Score of this metric name
                mean_absolute_error(float): Score of this metric name

            Returns: reg_score(dict): The printout of the dictionary showing the resulting metrics of the model
            """
        log_reg_predict = self.predict(x)
        reg_score = {}   
        reg_score["R2 Score"] = r2_score(y_true, log_reg_predict)
        reg_score["Mean Squared Error"] = mean_squared_error(y_true, log_reg_predict)
        reg_score["Root Mean Squared Error"] =  root_mean_squared_error(y_true, log_reg_predict)
        reg_score["Mean Absolute Error"] = mean_absolute_error(y_true, log_reg_predict)
        print(reg_score)
        return reg_score

# Classes of Regressors 

#Class for Linear Regression

class CustomLinearRegression(FitPredictScore):
    def __init__(self, random_state: int, params: dict):
        model = LinearRegression(**params)
        super().__init__(random_state=random_state, model=model)

    """Performs Regression with method of -> linear regression

        Args:
            model(): The model with the selected classification method.

        Returns: None
        """

class CustomKNN_Regressor(FitPredictScore):
    def __init__(self, n_neighbors: int, params: dict):
        model = KNeighborsRegressor(n_neighbors,**params)
        super().__init__(random_state=None,model=model)

    """Performs regression with method of -> KNN regression

        Args:
            n_neighbors(int): the best neighbours
            model(): The model with the selected classification method.

        Returns: None
        """

class CustomDecisionTreeReg(FitPredictScore):
    def __init__(self, params: dict):
        model = DecisionTreeRegressor(**params)
        super().__init__(random_state=None, model=model)

        """Performs regression with method of -> Decision Tree regression

        Args:
            model(): The model with the selected classification method.

        Returns: None
        """

class CustomRandomForestReg(FitPredictScore):
    def __init__(self, n_estimators: int, random_state: int, params: dict):
        model = RandomForestRegressor(n_estimators,**params)
        super().__init__(random_state=random_state, model=model)

        """Performs regression with method of -> random forest regression

        Args:
            model(): The model with the selected classification method.

        Returns: None
        """

class CustomSVR(FitPredictScore):
    def __init__(self, random_state: int, params: dict):
        model = SVR(**params)
        super().__init__(random_state=random_state, model=model)

        """Performs regression with method of -> SVR regression

        Args:
            model(): The model with the selected classification method.

        Returns: None
        """

class CustomANN_Regressor(FitPredictScore):
    def __init__(self, random_state: int, params: dict):
        model = MLPRegressor(**params)
        super().__init__(model=model, random_state=random_state)

        """Performs regression with method of -> ANN regression

        Args:
            model(): The model with the selected classification method.

        Returns: None
        """