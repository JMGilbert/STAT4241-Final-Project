import pandas as pd
import numpy as np
import math
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix

# Fitting the SVM for red wine


reddat = pd.read_csv("./winequality-red.csv", sep = ";")
X = reddat[reddat.columns[0:11]]
y = reddat[reddat.columns[11]]



sigma2 = y.var() * 1.5
epsilon = sigma2/math.sqrt(len(y))

clf = SVR(kernel = 'rbf', epsilon = epsilon, C = 3)
clf.fit(X,y)

y_true = y.copy()
y_pred = clf.predict(X)

print("MAD red")
print(mean_absolute_error(y_true, y_pred))
print("Accuracy T1 -- red")
print(sum((y_true - y_pred).abs() < 1)/len(y))
print("Accuracy T.5 -- red")
print(sum((y_true - y_pred).abs() < .50)/len(y))
print("Accuracy T.25 -- red")
print(sum((y_true - y_pred).abs() < .25)/len(y))

y_pred = np.rint(y_pred)
print(confusion_matrix(y_true,y_pred))

# Fitting the SVM for white wine

whitedat = pd.read_csv("./winequality-white.csv", sep = ";")
X = whitedat[whitedat.columns[0:11]]
y = whitedat[whitedat.columns[11]]


sigma2 = y.var() * 1.5
epsilon = sigma2/math.sqrt(len(y))

clf = SVR(kernel = 'rbf', epsilon = epsilon, C = 3)
clf.fit(X,y)

y_true = y.copy()
y_pred = clf.predict(X)

print("MAD white")
print(mean_absolute_error(y_true, y_pred))
print("Accuracy T1 -- white")
print(sum((y_true - y_pred).abs() < 1)/len(y))
print("Accuracy T.5 -- white")
print(sum((y_true - y_pred).abs() < .50)/len(y))
print("Accuracy T.25 -- white")
print(sum((y_true - y_pred).abs() < .25)/len(y))


y_pred = np.rint(y_pred)
print(confusion_matrix(y_true,y_pred))