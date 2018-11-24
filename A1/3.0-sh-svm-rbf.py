import datetime
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.model_selection import GridSearchCV
from sklearn import svm, metrics
import pickle
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

start = datetime.datetime.now()

# environment settings
cwd = os.getcwd()
data = os.path.join(cwd, 'data', '1.0-ag-data-exploration.csv')

# read in data
df = pd.read_csv(data)

# drop not needed columns
df = df.drop('Data Year - Fiscal', axis = 1)
df = df.drop('CompanyID', axis = 1)
df = df.drop('Return on Equity', axis = 1)

min = 0
max = len(df)

X = np.array(df.iloc[min:, df.columns != "BK"])
y = df.iloc[min:, df.columns == "BK"].values.reshape(-1,)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

print('After OverSampling, the shape of X_train: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of y_train: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))

parameter_candidates = [
    {'C': [1, 10, 100, 1000], 'gamma': [0.1, 0.001, 0.0001], 'kernel': ['rbf']}
]

# Create a classifier object with the classifier and parameter candidates
clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1, cv=10)

clf.fit(X_train_res, y_train_res)

print('Best params: ', clf.best_params_)

# save the trained classifier
with open(os.path.join(cwd, 'classifiers', '3.0-sh-svm.pkl'), 'wb') as f:
    pickle.dump(clf, f)

stop = datetime.datetime.now()

diff = stop - start

seconds = diff.seconds
minutes = (seconds % 3600) // 60

print('Total time: ', minutes)
