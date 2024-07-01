from capstone_project import Analyzer, Classifier

from sklearn.model_selection import train_test_split

def main(): 
#read data
    #absolute_path= 'E:/Repos/capstone_project/capstone_project/diamonds.csv'
    absolute_path= 'capstone_project/diamonds.csv'
    df = Analyzer.read_dataset(dataset_path=absolute_path)

#analyzer
#data exploration and preprocessing
    my_data_manipulation = Analyzer.DataManipulation(df=df)

#describe dataset
    #Analyzer.describe(dataset=df)

# Drop Missing Data
    #data_df = my_data_manipulation.drop_missing_data(df)
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
    #scaled_df = my_data_manipulation.standardize(df=df)
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


#testing DataVisualization class
   #my_data_visualization = Analyzer.DataVisualization(df=df)

# Plot correlation matrix
    #corr_matrix = my_data_visualization.plot_correlationMatrix(df=retrieved_df)
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
    #box_plotting = my_data_visualization.box_plot(df=retrieved_df, column_name1="clarity", column_name2="cut")
    #print(box_plotting)# execute encoding

#Before fitting and training we need to split the data

# train is now 80% of the entire data set
    y_true = retrieved_df['price'].values
    x = retrieved_df[["carat","cut","color","clarity","depth","table","x","y","z"]].values
    
    x_train, x, y_train, y_true = train_test_split(x, y_true, random_state=0, test_size=0.2)

# test is now 10% of the initial data set
# validation is now 10% of the initial data set
    x_val, x, y_val, y_true = train_test_split(x, y_true, random_state=0, test_size=.5)
    #print(y_true)   


    score_dict = {}

 #testing logisctical regression classifier   
    logistic_regression_classifier = Classifier.CustomLogiscticRegression(params={'solver':'lbfgs', 'tol': 0.0001, 'max_iter': 1000}, random_state=0)
    logistic_regression_classifier.fit(x_train, y_train)
    log_reg_predict = logistic_regression_classifier.predict(x)
    score_dict["Logistic Regression"] = logistic_regression_classifier.score(y_true, log_reg_predict)
    print(score_dict)
  


# testing Knn Classifier

    #KNN

    #neighbour = KNeighborsRegressor(n_neighbors = best_knn)
    #scores = []
    #nums = range(1,25)
    #best_knn = []
    #best_score_i = -1000

    #for i in nums:
    #    knn = Classifier.KNeighborsClassifier(n_neighbors = i)
    #    knn.fit(x_train, y_train)
    #    score_i = knn.score(x, y_true)
     #   scores.append(score_i)
        
    #    if score_i > best_score_i:
    #        best_score_i = score_i
    #        best_knn = i

    #print(best_knn)

    #knn = Classifier.CustomKNN_Classifier(n_neighbours=best_knn, params={})
    #knn.fit(x_train, y_train)
    #log_reg_predict = knn.predict(x)
    #score_dict["KNN"] = knn.score(y_true, log_reg_predict)
    #print(score_dict)

 #testing Decistion Tree classifier   
   # decision_tree_classifier = Classifier.CustomDecisionTree(params={'criterion':'gini'})
   # decision_tree_classifier.fit(x_train, y_train)
    #log_reg_predict = decision_tree_classifier.predict(x)
   # score_dict["Decision Tree"] = decision_tree_classifier.score(y_true, log_reg_predict)
   # print(score_dict)

 #testing Random Forest classifier   
    
    #random_forest_classifier = Classifier.CustomRandomForest(n_estimators=100, random_state=0,params={'criterion':'gini', 'max_leaf_nodes': 100})
    #random_forest_classifier.fit(x_train, y_train)
    #log_reg_predict = random_forest_classifier.predict(x)
    #score_dict["Random Forest"] = random_forest_classifier.score(y_true, log_reg_predict)
    #print(score_dict)

#testing SVC Classifier
    #svc_classifier = Classifier.CustomSVC(random_state=0, params={'kernel':'rbf', 'max_iter': 3, 'verbose':True})
    #svc_classifier.fit(x_train, y_train)
    #log_reg_predict = svc_classifier.predict(x)
    #score_dict["SVC"] = svc_classifier.score(y_true, log_reg_predict)
    #print(score_dict)

#testing ANN Classifier
    #ann_classifier = Classifier.CustomANN_Classifier(params={"hidden_layer_sizes":(100, 100, 100),'activation':'relu', 'solver':'adam', 'max_iter': 1000}, random_state=0)
    #ann_classifier.fit(x_train, y_train)
    #log_reg_predict = ann_classifier.predict(x)
    #score_dict["ANN"] = ann_classifier.score(y_true, log_reg_predict)
    #print(score_dict)

# #Plot of end results
# #plt.figure(figsize=(12,8))
# #plt.ylim(.5, 1)
# #sns.barplot(x=estimator, y= accuracy_score)
main()