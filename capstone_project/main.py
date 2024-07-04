from capstone_project import Analyzer, Classifier, Regressor, Clustering

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift
#from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

def main(): 
#read data
    #absolute_path= 'E:/Repos/capstone_project/capstone_project/diamonds.csv'
    path = 'capstone_project/diamonds.csv'
    df = Analyzer.read_dataset(dataset_path=path)
#Testing for analyzer

#data exploration and preprocessing
    my_data_manipulation = Analyzer.DataManipulation(df=df)

#describe dataset
    #Analyzer.describe(dataset=df)

# Drop Missing Data
    data_df = my_data_manipulation.drop_missing_data(df)
    #print(data_df)

# Drop columns if needed
    cleaned_df = my_data_manipulation.drop_column(column_names="Unnamed: 0")
    #print(cleaned_df)

# One hot encoder
    #encoded_df = my_data_manipulation.encode_features(column_names="color")
    #print(encoded_df)

# Label encoder
    lblencoded_df = my_data_manipulation.encode_label(column_names="color")
    lblencoded_df = my_data_manipulation.encode_label(column_names="cut")
    lblencoded_df = my_data_manipulation.encode_label(column_names="clarity")
    #print(lblencoded_df)

# Scale encoded data
    scaled_df = my_data_manipulation.standardize(df=df)
    #print(scaled_df)

# Shuffle dataset
    shuffled_df = my_data_manipulation.shuffle(df=df)
    #print(shuffled_df)

# Get 50% sample of dataset
    sampled_df = my_data_manipulation.sample(df=df)
    #print(sampled_df)

# Retrieve data
    retrieved_df = my_data_manipulation.retrieve_data(df=df)
    #print(retrieved_df)

# Description after cleaning/preprocessing
    #Analyzer.describe(dataset=retrieved_df)

#testing DataVisualization class

    #my_data_visualization = Analyzer.DataVisualization(df=df)

# Plot correlation matrix
    #orr_matrix = my_data_visualization.plot_correlationMatrix(df=retrieved_df)
    #print(corr_matrix)

# Pair Plot
    #pair_plot = my_data_visualization.plot_pairplot(df=retrieved_df)
    #print(pair_plot)

# Plotting of histogram grouped - not utilized currently
    #histogram_group = my_data_visualization.plot_histograms_group(df=retrieved_df)
    #print(histogram_group)

#testing plotting of histogram categorical
    #histogram_cat = my_data_visualization.plot_histograms_categorical(column_name1="clarity")
    #print(histogram_cat)

# Box plot categorical
    #box_plotting = my_data_visualization.box_plot(df=retrieved_df, column_name1="clarity", column_name2="price")
    #print(box_plotting)# execute encoding

#Before fitting and training we need to split the data

# # train is now 80% of the entire data set
#     y_true = retrieved_df['price'].values
#     y_true = y_true.astype(int) #needed to add as wasn't turning into an object
#     x = retrieved_df[["carat","cut","color", "clarity", "depth","table","x","y","z"]].values
    
#     x_train, x, y_train, y_true = train_test_split(x, y_true, random_state=0, test_size=0.8)

# # # test is now 10% of the initial data set
# # # validation is now 10% of the initial data set
#     x_val, x, y_val, y_true = train_test_split(x, y_true, random_state=0, test_size=.5)
    #print(len(x))

# # Testing classifier.py

# NOTES for presentation: for Classifier we want the label to be clarity of the diamonds.

#     score_dict = {}
#     matrix_dict = {}

# #  #testing logisctical regression classifier   
# # NOTES: for presentation - use scalar and max iter 1000 for result of approx .55-.57
#     logistic_regression_classifier = Classifier.CustomLogisticRegression(params={'solver':'lbfgs', 'C':1.0,'max_iter':1000}, random_state=0) 
#     logistic_regression_classifier.fit(x_train, y_train)
#     score_dict["Logistic Regression"] = logistic_regression_classifier.score(y_true=y_true, x=x)
#     matrix_dict["Logistic Regression"] = logistic_regression_classifier.conf_matrix(y_true=y_true, x=x)
#     #logistic_regression_classifier.plt_conf_matrix(y_true=y_true, x=x)

# # testing Knn Classifier
# # NOTES: for presentation - Appears best n_neighbor is always 1 and consistently scores .55
#     #first find best n_neighbor
        
#     scores = []
#     nums = range(1,25)
#     best_knn = []
#     best_score_i = -1000

#     for i in nums:
#         knn = Classifier.KNeighborsClassifier(n_neighbors = i)
#         knn.fit(x_train, y_train)
#         score_i = knn.score(x, y_true)
#         scores.append(score_i)
        
#         if score_i > best_score_i:
#             best_score_i = score_i
#             best_knn = i

#     #print(best_knn)

#     knn = Classifier.CustomKNN_Classifier(n_neighbors=best_knn, params={})
#     knn.fit(x_train, y_train)
#     score_dict["KNN"] = knn.score(y_true=y_true, x=x)
#     matrix_dict["KNN"] = knn.conf_matrix(y_true=y_true, x=x)

#  #testing Decistion Tree classifier   
#  # NOTES: for presentation - used criterion Gini and scores approx .64-.65
#     decision_tree_classifier = Classifier.CustomDecisionTree(params={'criterion':'gini'})
#     decision_tree_classifier.fit(x_train, y_train)
#     score_dict["Decision Tree"] = decision_tree_classifier.score(y_true=y_true, x=x)
#     matrix_dict["Decision Tree"] = decision_tree_classifier.conf_matrix(y_true=y_true, x=x)

#  #testing Random Forest classifier   
# # NOTES: for presentation - used criterion Gini and scores approx .69-.70  
#     random_forest_classifier = Classifier.CustomRandomForest(n_estimators=100, random_state=0,params={'criterion':'gini'})
#     random_forest_classifier.fit(x_train, y_train)
#     score_dict["Random Forest"] = random_forest_classifier.score(y_true=y_true, x=x)
#     matrix_dict["Random Forest"] = random_forest_classifier.conf_matrix(y_true=y_true, x=x)

# #testing SVC Classifier
# # NOTES: for presentation - used defaults and scores at .56
#     svc_classifier = Classifier.CustomSVC(random_state=0, params={})
#     svc_classifier.fit(x_train, y_train)
#     score_dict["SVC"] = svc_classifier.score(y_true=y_true, x=x)
#     matrix_dict["SVC"] = svc_classifier.conf_matrix(y_true=y_true, x=x)

# #testing ANN Classifier
# # NOTES: for presentation - did some hypertuning here to get .69
#     ann_classifier = Classifier.CustomANN_Classifier(params={"hidden_layer_sizes":(100, 100, 100),'activation':'relu', 'solver':'adam', 'max_iter': 1000}, random_state=0)
#     ann_classifier.fit(x_train, y_train)
#     score_dict["ANN"] = ann_classifier.score(y_true=y_true, x=x)
#     matrix_dict["ANN"] = ann_classifier.conf_matrix(y_true=y_true, x=x)

# # Printing summary of scores and confusion matrices    
#     print(score_dict)
#     #print(matrix_dict)

# # #Plot of end results
#     plt.figure(figsize=(12,8))
#     plt.ylim(.5, 1)
#     sns.barplot(score_dict).set_title('Comparison of Classification models with target Clarity')
#     plt.show()


# # # Testing Regressor.py

# # NOTES for presentation: for Regressor we want to switch price and clarity again for label as the problem we are trying to solve with regression is the pricing of the diamonds based on their features

# #testing Linear Regressor  
#     linear_regression = Regressor.CustomLinearRegression(params={}, random_state=0)
#     linear_regression.fit(x_train, y_train)
#     linear_regression.score(y_true=y_true, x=x)
 

# # testing Knn Regressor

# #KNN

#     scores = []
#     nums = range(1,25)
#     best_knn = []
#     best_score_i = -1000

#     for i in nums:
#         knn_reg = Regressor.KNeighborsRegressor(n_neighbors = i)
#         knn_reg.fit(x_train, y_train)
#         score_i = knn_reg.score(x, y_true)
#         scores.append(score_i)
        
#         if score_i > best_score_i:
#             best_score_i = score_i
#             best_knn = i

#     print(best_knn)

#     knn_reg = Regressor.CustomKNN_Regressor(n_neighbors=best_knn, params={})
#     knn_reg.fit(x_train, y_train)
#     knn_reg.score(y_true=y_true, x=x)


#  #testing Decision Tree Regressor   
#     decision_tree_reg = Regressor.CustomDecisionTreeReg(params={})
#     decision_tree_reg.fit(x_train, y_train)
#     decision_tree_reg.score(y_true=y_true, x=x)


#  #testing Random Forest Regressor   
    
#     random_forest_regressor = Regressor.CustomRandomForestReg(n_estimators=100, random_state=0,params={'max_leaf_nodes': 100})
#     random_forest_regressor.fit(x_train, y_train)
#     random_forest_regressor.score(y_true=y_true, x=x)


# #testing SVR Regressor
#     svr_regressor = Regressor.CustomSVR(params={}, random_state=0)
#     svr_regressor.fit(x_train, y_train)
#     svr_regressor.score(y_true=y_true, x=x)

# #testing ANN Regressor
#     ann_regressor = Regressor.CustomANN_Regressor(params={"hidden_layer_sizes":(100, 100, 100),'activation':'relu', 'solver':'adam', 'max_iter': 1000}, random_state=0)
#     ann_regressor.fit(x_train, y_train)
#     ann_regressor.score(y_true=y_true, x=x)

#Testing Clustering

# # training data is different for clustering
    x_columns = 10
    x = retrieved_df.iloc[:, 0:x_columns].values


#testing K-Means Clustering

    # Elbow method
    
    kmean_cluster = Clustering.CustomKMeans(n_clusters = 3, random_state=0, params={})

    kmeanscores = []
    
    for i in range(1, 21):
        model = KMeans(n_clusters=i, random_state=0)
        model.fit(x)
        kmeanscores.append(model.inertia_)

    import matplotlib.pyplot as plt
    plt.plot(range(1, 21), kmeanscores, marker='.', markersize=10)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE') # Model Inertia
    plt.show()

    kmeans_label = kmean_cluster.fit(x)
    #print(kmeans_label)
    kmeans_predict = kmean_cluster.predict(x)
    print(kmeans_predict)
#    centres = kmeans_label.cluster_centers_ #  having trouble with this line to get it working
 #   print(centres)

main()