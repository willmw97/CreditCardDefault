
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import sklearn
import pydot


import pickle

#Sklearn is a machine learning library
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.externals.six import StringIO

from subprocess import call
from IPython.display import Image
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib



import warnings
warnings.filterwarnings('ignore')

#reads in and sets up data set
dataset = pd.read_csv('clients6_data.csv')
#dataset.drop(dataset.columns[dataset.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
#dataset = dataset.iloc[:,1:45]
#dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]
print(dataset)
#dataset.drop('ID',axis=1,inplace=True)
#dataset.drop('',axis=1,inplace=True)

#added features
# dataset['AgeCategory'] = 0
# dataset.loc[((dataset['AGE'] > 20) & (dataset['AGE'] < 30)) , 'AgeCategory'] = 1
# dataset.loc[((dataset['AGE'] >= 30) & (dataset['AGE'] < 40)) , 'AgeCategory'] = 2
# dataset.loc[((dataset['AGE'] >= 40) & (dataset['AGE'] < 50)) , 'AgeCategory'] = 3
# dataset.loc[((dataset['AGE'] >= 50) & (dataset['AGE'] < 60)) , 'AgeCategory'] = 4
# dataset.loc[((dataset['AGE'] >= 60) & (dataset['AGE'] < 70)) , 'AgeCategory'] = 5
# dataset.loc[((dataset['AGE'] >= 70) & (dataset['AGE'] < 80)) , 'AgeCategory'] = 6

#col_to_norm = ['LIMIT_BAL','BILL_AV_AMT', 'PAY_AMT_AV', 'AVAILABLE_CRED_PERCENT']
#dataset[col_to_norm] = dataset[col_to_norm].apply(lambda x : (x-np.mean(x))/np.std(x))



#used sample of 2000 so far
#dataset = dataset.sample(n=2000,replace = False,random_state=1)

dataset.info()

dataset.head()


#Set the features we want to look at
# features = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE' ,'PAY_MAX_SCORE', 'BILL_AV_AMT','PAY_AMT_AV',
#             'AVAILABLE_CRED_PERCENT']
features = [ 'LIMIT_BAL' , 'SEX' , 'EDUCATION' , 'MARRIAGE','AGE','PAY_MAX_SCORE','PREVIOUS_BAL','LAST_PAYMENT','CURRENT_BAL','CRED_UTILIZATION_PERCENT' ]

#Target assigned
y = dataset['default_payment_next_month'].copy()
#Features assigned
X = dataset[features].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)
X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.50, random_state=42)



X_train.shape
X_test.shape


dct = DecisionTreeClassifier(random_state = 0, max_leaf_nodes=9)
dct.fit(X_train,y_train)
treeModel = dct.fit(X_train,y_train)
y_pred = dct.predict(X_test)

filename = open("final_tree_model.sav","wb")
pickle.dump(treeModel, filename)

#testScores
#scores = cross_val_score(dct, X, y)
# print("Score Decision Tree")
# print(scores)

#dct_param_grid = {'splitter':['best','random'],'criterion': ['entropy', 'gini'],'class_weight' : ['balanced',None]}
roc=roc_auc_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model = pd.DataFrame([['Decision Tree Classifier', acc,prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])

print(model)

#
# #optimization using grid search
#
# # set up parameters
#
# parameters = {'max_depth':[2,4,6,8,10],'min_samples_leaf':[2,4,6,8,10],
#               'min_samples_split':[2,4,6,8,10]}
# grid_search_dt = GridSearchCV(estimator=dct,param_grid=parameters,scoring = 'accuracy',cv=5,n_jobs=-1)
# grid_search_dt = grid_search_dt.fit(X_train,y_train)
#
# best_accuracy_1 = grid_search_dt.best_score_
# #best_model stores the best chosen model
# best_model = grid_search_dt.best_estimator_
# best_parameters_2 = grid_search_dt.best_params_
#
# y_pred_dct = grid_search_dt.predict(X_test)
#
#
#
# roc=roc_auc_score(y_test, y_pred_dct)
# acc = accuracy_score(y_test, y_pred_dct)
# prec = precision_score(y_test, y_pred_dct)
# rec = recall_score(y_test, y_pred_dct)
# f1 = f1_score(y_test, y_pred_dct)
#
# model =  pd.DataFrame([['Decision Tree Tuned', acc,prec,rec, f1,roc]],
#                columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
#
# print(model)

#Output decision tree model

# Create tree
######REALLY IMPORTANT#######

#need to run the following command in terminal to work
#conda install python-graphviz
############################

#tree1 and dtree1 are just using the chosen tree parameters
tree.export_graphviz(treeModel, out_file='tree1.dot', feature_names=X.columns,
                filled=True, rounded=True,
                special_characters=True)

(graph,) = pydot.graph_from_dot_file('tree1.dot')
graph.write_png('dtree1.png')
#
# #tree2 and dtree2 are using grid search to find the best possible model
# tree.export_graphviz(best_model, out_file='tree2.dot', feature_names= X.columns,
#                 filled=True, rounded=True,
#                 special_characters=True)
#
# (graph,) = pydot.graph_from_dot_file('tree2.dot')
# graph.write_png('dtree2.png')


