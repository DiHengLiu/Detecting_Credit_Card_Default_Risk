#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 17:36:51 2018

@author: dihengliu
"""

import math
import pandas
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from numpy.random import permutation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.cluster import KMeans
#from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
#mport sklearn
#import sklearn.datasets
#import sklearn.linear_model
#import matplotlib
from sklearn.neural_network import MLPClassifier
#from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
#from sklearn.datasets import make_classification

#credit = pandas.read_excel('/Users/dihengliu/Desktop/DM2 project 01/default of credit card clients.xls')
credit = pandas.read_excel('/Users/dihengliu/Desktop/DM2 project 01/Dataset.xls')
credit.head(10)

'''Pre-processing'''
columns = list(credit)
X = credit.iloc[:,0:-1]
y = credit.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=52)

random_indices = permutation(credit.index)
test_cutoff = math.floor(len(credit)/3)

test = credit.loc[random_indices[1:test_cutoff]]
train = credit.loc[random_indices[test_cutoff:]]

'''K-mean Plot(PCA)'''
kmeans_model = KMeans(n_clusters=2, random_state=1)
good_columns = credit._get_numeric_data().dropna(axis=1)
kmeans_model.fit(good_columns)
labels = kmeans_model.labels_
print(labels)

for i in range(0,22):
    pca_2 = PCA(23)
    plot_columns = pca_2.fit_transform(good_columns)
    plt.scatter(x=plot_columns[:,i], y=plot_columns[:,i+1], c=labels)
    plt.show()

'''Elbow Method'''
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
 
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

'''KNN'''
for j in range(1,11):
    knn = neighbors.KNeighborsClassifier(n_neighbors=j)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    accl = accuracy_score(predictions, y_test)
    mse = (((predictions - y_test) ** 2).sum()) / len(predictions)
    print("Number of neighbors:", j)
    scores = cross_val_score(knn, X, y, cv=10)
    print('The scroes mean:',scores.mean())
    print('The accuracy of KNN:', accl)
    print('The mean square error:', mse)
    
    precision = precision_score(y_test, predictions, average='weighted')
    print('The precision of KNN:', precision)
    recall = recall_score(y_test, predictions, average='weighted') 
    print('The recall-score of KNN:', recall)
    print()

'''SVM'''
cd = svm.SVC(kernel = 'rbf')
cd.fit(X_train, y_train)

y_hat_train = cd.predict(X_train)
y_hat_test = cd.predict(X_test)
train_acc = accuracy_score(y_hat_train, y_train)
test_acc = accuracy_score(y_hat_test, y_test)

scores = cross_val_score(cd, X, y, cv=3)
print('The scores mean:',scores.mean())
print('The train accuracy of SVM:',train_acc)
print('The test accuracy of SVM:',test_acc)

precision = precision_score(y_test, y_hat_test, average='weighted')
print('The precision of SVM:', precision)
recall = recall_score(y_test, y_hat_test, average='weighted') 
print('The recall-score of SVM:', recall)
print()

'''Neural Network''' 
clf = MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(10,), random_state=1, activation = 'logistic')
clf.fit(X_train, y_train)
y_hat_train = clf.predict(X_train)
y_hat_test = clf.predict(X_test)
train_acc = accuracy_score(y_hat_train, y_train)
test_acc = accuracy_score(y_hat_test, y_test)

scores = cross_val_score(clf, X, y, cv=10)
print('The scores mean:',scores.mean())
print('The train_acc accuracy of Neural Network:', train_acc )
print('The test_acc accuracy of Neural Network:', test_acc)

precision = precision_score(y_test, y_hat_test, average='weighted')
print('The precision of Neural Network:', precision)
recall = recall_score(y_test, y_hat_test, average='weighted') 
print('The recall-score of Neural Network:', recall)
print()

'''Feature_Selection'''
clf_rf_5 = RandomForestClassifier()      
clr_rf_5 = clf_rf_5.fit(X_train,y_train)

importances = clr_rf_5.feature_importances_
std = np.std([tree.feature_importances_ for tree in clr_rf_5.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

plt.figure(1, figsize=(14, 13))
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="g", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]))
plt.xlim([-1, X_train.shape[1]])
plt.show()
