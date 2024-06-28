from capstone_project import Analyzer, Classifier

from sklearn.model_selection import train_test_split

def main(): 
#read data
    absolute_path= 'E:/Repos/capstone_project/capstone_project/diamonds.csv'
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


#testing DataVisualization class
    my_data_visualization = Analyzer.DataVisualization(df=df)

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
    box_plotting = my_data_visualization.box_plot(df=retrieved_df, column_name1="clarity", column_name2="cut")
    print(box_plotting)# execute encoding

main()
#Before fitting and training we need to split the data

# train is now 80% of the entire data set
   # y_true = df['price'].values
  # x = df[["carat","cut","color","clarity","depth","table","x","y","z"]].values
    
    #x_train, x, y_train, y_true = train_test_split(x, y_true, random_state=0, test_size=0.2)

# test is now 10% of the initial data set
# validation is now 10% of the initial data set
   # x_val, x, y_val, y_true = train_test_split(x, y_true, random_state=0, test_size=.5)



    #score_dict[]


    #logistic_regression_classifier = classifier.LogisticRegression()
    #logistic_regression_classifier.fit(x_train, y_train)
    #score_dict["Logistic Regression"] = logistic_regression_classifier.score(x_test)


                                                                             
#     ann_classifier = classifier.ANN()                                                                         )
# #classifier



# if __name__ == "__main__":
#    # df = analyzer.read_dataset()





#     absolute_path= 'E:/Repos/capstone_project/capstone_project/diamonds.csv'
#     #absolute_path= 'C:/Users/hyppi/Repos/capstone_project/capstone_project/diamonds.csv'
#     df = Analyzer.read_dataset(dataset_path=absolute_path)

#     #def score(y_true: np.array, y_predicted: np.array) -> float: # non member function version
# #      accuracy_score = accuracy_score(y_true, y_predicted)
# #      return accuracy_score


#    #testing Logistic Regression
#    params = {
#       "criterion": "lbgs"
#       "solver": 100
#    }

#     logistic_regression = LogisticRegression(params=params)

#    #testing KNN Classifier
#    params = {
#       "criterion": "lbgs"
#       "solver": 100
#    }

#     knn_classifier = KNeighborsClassifier(params=params)

#     #testing Decision Tree
#     params = {
#     "gamma":"auto"
#    }

#     decision_tree = DecisionTreeClassifier(params=params)

#     #testing Random Forest
#     params = {
#     "gamma":"auto"
#    }

#     random_forest = RandomForestClassifier(params=params)

#    #testing SVC
#     params = {
#     "gamma":"auto"
#    }

#     svc = SVC(params=params)

#    #testing ANN
#     params = {
#     "max_iter": 1
#    }
#     ann_classifier = MLPClassifier(params=params)

# #Plot of end results
# #plt.figure(figsize=(12,8))
# #plt.ylim(.5, 1)
# #sns.barplot(x=estimator, y= accuracy_score)