import numpy as np

from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift


# fit_predict_score class adjusting score for regression
class FitPredict():
    def __init__(self, random_state: int, model: dict):
        self.random_state = random_state
        self.model = model

    # Fit = function
    def fit(self, x: np.array) -> np.array:
        """ This function is to fit the data with x and y training arrays
              Args:
                x_train(np.array): The name of the x array to fit
                y_train(np.array): The name of the y array to fit

            Returns: model(np.array): The arrays after fit
        """
        model = self.model.fit(x)
        #centres=model.cluster_centers_
        #return centres

    # Predict = function
    def predict(self, x: np.array) -> np.array:
        """ This function is to predict based on data within the given array
            Args:
                x(np.array): The name of the array to predict on

            Returns: model_predict(np.array): The prediction on the array data
        """
        model = self.model.predict(x)
        return model

  
# Classes of Clustering 

#Class for K-Means Clustering

class CustomKMeans(FitPredict):
    def __init__(self, n_clusters: int, random_state: int, params: dict):
        model = KMeans(n_clusters, random_state=random_state,**params)
        super().__init__(random_state=0, model=model)

    """Performs Clustering with method of -> K-Means regression

        Args:
            model(): The model with the selected K-Means method.

        Returns: Labels and inertia
        """
    
class CustomAgglomerativeClustering(FitPredict):
    def __init__(self, params: dict):
        model = AgglomerativeClustering(**params)
        super().__init__(random_state=None,model=model)

    """Performs clustering with method of -> Agglomerative Hierarchal Clustering

        Args:
            n_neighbors(int): the best neighbours
            model(): The model with the selected clustering method.

        Returns: None
        """

class CustomMeanShift(FitPredict):
    def __init__(self, params: dict):
        model = MeanShift(**params)
        super().__init__(random_state=None, model=model)

        """Performs clustering with method of -> Mean Shift Clustering

        Args:
            model(): The model with the selected regression method.

        Returns: None
        """

# centres = kmeans_predict.cluster_centers_ #  having trouble with this line to get it working
    # centres = []
        
    # for i in range(1, 3):
    #     model = KMeans(n_clusters=3, random_state=0)
    #     model.fit(x)
    #     centres.append(model.cluster_centers_)
 
    # colors = ['orange', 'blue', 'green', 'magenta', 'cyan']
    # for i in range(3):
    #     plt.scatter(x[kmeans_predict == i, 0], x[kmeans_predict == i, 1], c=colors[i])
    # plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], color='red', marker='+', s=100)
    # plt.title('K-Means Clustering')
    # plt.xlabel('Annual Income (k$)')
    # plt.ylabel('Spending Score (1-100)')
    # plt.show()

