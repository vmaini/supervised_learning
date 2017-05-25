import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from patsy import dmatrices
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import pydot

'''
PROCESS
1) visualize/explore the data, impute null values for key features w simple method (mean or mode)
2) separate into X and y for modeling
3) do a logistic regression, interpret coefficients
4) try a decision tree for intuitions about splits/what matters
5) try a random forest to improve performance
'''

train_data = pd.read_csv('../data/churn_train.csv')
test_data = pd.read_csv('../data/churn_test.csv')

def prepare_data(df):

    '''
    Input: churn data (Pandas dataframe)
    Returns:
    X with features normalized & missing values imputed
    y (1 if user churned, 0 if still active)
    '''

    #impute null values as mean
    df['avg_rating_by_driver'] = df['avg_rating_by_driver'].apply(lambda x: df['avg_rating_by_driver'].mean() if x == 'nan' or np.isnan(x) else x)

    df['avg_rating_of_driver'] = df['avg_rating_of_driver'].apply(lambda x: df['avg_rating_of_driver'].mean() if x == 'nan' or np.isnan(x) else x)

    # create col for churned or not

    df['churned'] = (df['last_trip_date'] < '2014-06-01') * 1

    # create dataframes with an intercept column and dummy variables

    y, X = dmatrices('churned ~ avg_rating_by_driver + avg_rating_of_driver + phone + trips_in_first_30_days + weekday_pct', df, return_type="dataframe")

    X = X.rename(columns = {'phone[T.iPhone]':'iPhone'})
    # flatten y into a 1-D array
    y = np.ravel(y)

    # normalize the features
    cols = X.columns
    min_max_scaler = preprocessing.MinMaxScaler()
    X = pd.DataFrame(min_max_scaler.fit_transform(X))
    X.columns = cols

    return X, y

X_train, y_train = prepare_data(train_data)

def logistic_reg(X,y):
    model = LogisticRegression()
    model = model.fit(X_train,y_train)
    print "Logistic regression score: ",model.score(X, y)

    # examine model coefficients
    print "Logistic regression coeffs: \n",pd.DataFrame(zip(X.columns, np.transpose(model.coef_)))

    # no p-values available here. lesson learned: use statsmodel for regression

def decision_tree(X,y,max_depth=5):
    # Let's try a decision tree
    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt = dt.fit(X_train,y_train)
    print "Decision tree score: ",dt.score(X,y)

    # visualize decision tree
    dot_data = export_graphviz(dt, out_file='tree.dot')
    # graph = pydot.graph_from_dot_data(dot_data)
    # graph.write_pdf("tree.pdf")

def random_forest(X,y):
    rf = RandomForestClassifier(n_estimators=500,max_depth=3)
    rf.fit(X_train,y_train)
    print "Random forest score: ", rf.score(X,y)

print "Performance on test data: "
X_test, y_test = prepare_data(test_data)
logistic_reg(X_test, y_test)
decision_tree(X_test,y_test)
random_forest(X_test,y_test)
