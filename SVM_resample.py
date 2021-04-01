import pandas as pd
import numpy as np
import math 
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter
import statistics

# Fitting the SVM for white wine

whitedat = pd.read_csv("./winequality-white.csv", sep = ";")
X = whitedat[whitedat.columns[0:11]]
y = whitedat[whitedat.columns[11]]

white_class=pd.unique(whitedat['quality'])
white_class.sort()


clfknn = KNeighborsClassifier(n_neighbors=3)
clfknn.fit(X,y)

y_pred = clfknn.predict(X)

sigma2 = (y - y_pred).var() * 1.5
epsilon = sigma2/math.sqrt(len(y))

clf = SVR(kernel = 'rbf', epsilon = epsilon, C = 3, gamma=2**0.5)

wdcopy = whitedat.copy()
train_set = wdcopy.sample(frac = 0.67, random_state = 0)
test_set = wdcopy.drop(train_set.index)
X_train = train_set[train_set.columns[0:11]]
y_train = train_set[train_set.columns[11]]

X_test = test_set[test_set.columns[0:11]]
y_test = test_set[test_set.columns[11]]

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred = np.rint(y_pred)

confusion_matrix(y_test,y_pred)

# Oversample underrepresented classes

counter = Counter(y_train)
print(counter)
sm = SMOTE(sampling_strategy="not majority",random_state=24,k_neighbors=2)
Augmented_X_white,Augmented_Y_white=sm.fit_resample(X_train, y_train)
counter = Counter(Augmented_Y_white)
print(counter)

clf.fit(Augmented_X_white, Augmented_Y_white)
y_pred = clf.predict(X_test)



white_precision=pd.DataFrame(index=white_class,columns=["T=0.5 (%)","T=1.0 (%)"])
for i,c in enumerate(white_class):
    label=y_test[y_test==c]
    predictor=y_pred[y_test==c]
    pre1=100*sum((label - predictor).abs() < 0.5)/len(label)
    pre2=100*sum((label - predictor).abs() < 1)/len(label)
    white_precision.at[c, "T=0.5 (%)"] = pre1
    white_precision.at[c, "T=1.0 (%)"] = pre2

overall=pd.DataFrame(index=["Overall"],columns=["T=0.5 (%)","T=1.0 (%)"])
pre1=100 * sum((y_test - y_pred).abs() < 0.5)/len(y_test)
pre2=100 * sum((y_test - y_pred).abs() < 1)/len(y_test)
overall.at["Overall", "T=0.5 (%)"] = pre1
overall.at["Overall", "T=1.0 (%)"] = pre2

white_precision = white_precision.append(overall)
print(white_precision)


y_pred = np.rint(y_pred)

print(confusion_matrix(y_test,y_pred))

# Median sampling

counter = Counter(y_train)
print(counter)
median=int(statistics.median(counter.values()))
sm = SMOTE(sampling_strategy={3:median,9:median},random_state=24,k_neighbors=2)
Augmented_X_white,Augmented_Y_white=sm.fit_resample(X_train, y_train)
counter = Counter(Augmented_Y_white)
print(counter)

clf.fit(Augmented_X_white, Augmented_Y_white)
y_pred = clf.predict(X_test)
white_precision=pd.DataFrame(index=white_class,columns=["T=0.5 (%)","T=1.0 (%)"])
for i,c in enumerate(white_class):
    label=y_test[y_test==c]
    predictor=y_pred[y_test==c]
    pre1=100*sum((label - predictor).abs() < 0.5)/len(label)
    pre2=100*sum((label - predictor).abs() < 1)/len(label)
    white_precision.at[c, "T=0.5 (%)"] = pre1
    white_precision.at[c, "T=1.0 (%)"] = pre2

overall=pd.DataFrame(index=["Overall"],columns=["T=0.5 (%)","T=1.0 (%)"])
pre1=100 * sum((y_test - y_pred).abs() < 0.5)/len(y_test)
pre2=100 * sum((y_test - y_pred).abs() < 1)/len(y_test)
overall.at["Overall", "T=0.5 (%)"] = pre1
overall.at["Overall", "T=1.0 (%)"] = pre2

white_precision = white_precision.append(overall)
print(white_precision)


y_pred = np.rint(y_pred)

print(confusion_matrix(y_test,y_pred))

# All but majority

counter = Counter(y_train)
print(counter)
sm = SMOTE(sampling_strategy="not majority",random_state=24,k_neighbors=2)
Augmented_X_white,Augmented_Y_white=sm.fit_resample(X_train, y_train)
counter = Counter(Augmented_Y_white)
print(counter)

clf.fit(Augmented_X_white, Augmented_Y_white)
y_pred = clf.predict(X_test)
white_precision=pd.DataFrame(index=white_class,columns=["T=0.5 (%)","T=1.0 (%)"])
for i,c in enumerate(white_class):
    label=y_test[y_test==c]
    predictor=y_pred[y_test==c]
    pre1=100*sum((label - predictor).abs() < 0.5)/len(label)
    pre2=100*sum((label - predictor).abs() < 1)/len(label)
    white_precision.at[c, "T=0.5 (%)"] = pre1
    white_precision.at[c, "T=1.0 (%)"] = pre2

overall=pd.DataFrame(index=["Overall"],columns=["T=0.5 (%)","T=1.0 (%)"])
pre1=100 * sum((y_test - y_pred).abs() < 0.5)/len(y_test)
pre2=100 * sum((y_test - y_pred).abs() < 1)/len(y_test)
overall.at["Overall", "T=0.5 (%)"] = pre1
overall.at["Overall", "T=1.0 (%)"] = pre2

white_precision = white_precision.append(overall)
print(white_precision)


y_pred = np.rint(y_pred)

print(confusion_matrix(y_test,y_pred))