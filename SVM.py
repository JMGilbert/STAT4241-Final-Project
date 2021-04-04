import pandas as pd
import numpy as np
import math
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from operator import itemgetter

# Fitting the SVM for white wine

whitedat = pd.read_csv("./winequality-white.csv", sep = ";")
X = whitedat[whitedat.columns[0:11]]
y = whitedat[whitedat.columns[11]]

clfknn = KNeighborsClassifier(n_neighbors=3)
clfknn.fit(X,y)

y_pred = clfknn.predict(X)

sigma2 = (y - y_pred).var() * 1.5
epsilon = sigma2/math.sqrt(len(y))

clf = SVR(kernel = 'rbf', epsilon = epsilon, C = 3)


unchanged = 0
droplist = []
gamma_search = [2**x for x in range(-15,3)]

wdcopy = whitedat.copy()
train_set = wdcopy.sample(frac = 0.67, random_state = 0)
test_set = wdcopy.drop(train_set.index)
X_train = train_set[train_set.columns[0:11]]
y_train = train_set[train_set.columns[11]]

X_test = test_set[test_set.columns[0:11]]
y_test = test_set[test_set.columns[11]]


clf.set_params(gamma = 0.5)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

L2 = mean_squared_error(y_test, y_pred)
print(L2)

while True:
    # First, search for optimal gamma
    wdcopy = whitedat.copy()
    train_set = wdcopy.sample(frac = 0.67, random_state = 0)
    test_set = wdcopy.drop(train_set.index)

    X_train = train_set[train_set.columns[0:11]]
    y_train = train_set[train_set.columns[11]]

    X_test = test_set[test_set.columns[0:11]]
    y_test = test_set[test_set.columns[11]]

    X_train = X_train.drop(columns = droplist)
    X_test = X_test.drop(columns = droplist)

    GS_L2 = [None] * len(gamma_search)
    for i in range(0,len(gamma_search)):
        clf.set_params(gamma = gamma_search[i])
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        GS_L2[i] = mean_squared_error(y_test, y_pred)

    gamma = gamma_search[min(enumerate(GS_L2), key=itemgetter(1))[0]]
    print(gamma)
    clf.set_params(gamma = gamma)
    clf.fit(X_train,y_train)

    means = X_train[X_train.columns].mean()
    Va = [None] * len(X_train.columns)

    for i in range(0,len(X_train.columns)):
        vartemp = X_train.copy()
        vartemp.iloc[:,0:len(vartemp.columns)] = means[0:len(vartemp.columns)]
        vartemp.iloc[:,i] = X_train.iloc[:,i]
        yi_pred = clf.predict(vartemp)
        Va[i] = yi_pred.var()

    Ra = Va/sum(Va)

    drop_var = min(enumerate(Ra), key=itemgetter(1))[0]
    droplist.append(X_train.columns[drop_var])

    X_train = X_train.drop(columns = X_train.columns[drop_var])
    X_test = X_test.drop(columns = X_test.columns[drop_var])

    # retrain:

    clf.fit(X_train,y_train)
    y_true = y_test.copy()
    y_pred = clf.predict(X_test)

    L2_prev = L2
    L2 = mean_squared_error(y_true, y_pred)
    print(L2)
    if L2 > L2_prev:
        droplist.pop()
        print(droplist)
        unchanged = unchanged + 1
        L2 = L2_prev
    else:
        unchanged = 0

    if unchanged == 2 or len(droplist) == 10:
        wdcopy = whitedat.copy()
        train_set = wdcopy.sample(frac = 0.67, random_state = 0)
        test_set = wdcopy.drop(train_set.index)

        X_train = train_set[train_set.columns[0:11]]
        y_train = train_set[train_set.columns[11]]

        X_test = test_set[test_set.columns[0:11]]
        y_test = test_set[test_set.columns[11]]

        X_train = X_train.drop(columns = droplist)
        X_test = X_test.drop(columns = droplist)
        clf.fit(X_train,y_train)
        y_true = y_test.copy()
        y_pred = clf.predict(X_test)
        print("L2 white")
        print(mean_squared_error(y_true, y_pred))
        print("Accuracy T1 -- white")
        print(sum((y_true - y_pred).abs() < 1)/len(y_test))
        print("Accuracy T.5 -- white")
        print(sum((y_true - y_pred).abs() < .50)/len(y_test))
        print("Accuracy T.25 -- white")
        print(sum((y_true - y_pred).abs() < .25)/len(y_test))
        print(X_train.columns)
        break