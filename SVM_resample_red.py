import pandas as pd
import numpy as np
import math 
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
import statistics

# Fitting the SVM for red wine

f = open("redres_SVM.txt", "wb")
reddat = pd.read_csv("./winequality-red.csv", sep = ";")
X_red = reddat[reddat.columns[0:11]]
y_red = reddat[reddat.columns[11]]

red_class=pd.unique(reddat['quality'])
red_class.sort()


clfknn = KNeighborsClassifier(n_neighbors=3)
clfknn.fit(X_red,y_red)

y_pred = clfknn.predict(X_red)

sigma2 = (y_red - y_pred).var() * 1.5
epsilon = sigma2/math.sqrt(len(y_red))

clf = SVR(kernel = 'rbf', epsilon = epsilon, C = 3, gamma=2**-7)


X_train, X_test, y_train, y_test = train_test_split(X_red, y_red, test_size=0.33, random_state=42,stratify=y_red)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

red_precision=pd.DataFrame(index=red_class,columns=["T=0.5 (%)","T=1.0 (%)"])
for i,c in enumerate(red_class):
    label=y_test[y_test==c]
    predictor=y_pred[y_test==c]
    pre1=100*sum((label - predictor).abs() < 0.5)/len(label)
    pre2=100*sum((label - predictor).abs() < 1)/len(label)
    red_precision.at[c, "T=0.5 (%)"] = pre1
    red_precision.at[c, "T=1.0 (%)"] = pre2

overall=pd.DataFrame(index=["Overall"],columns=["T=0.5 (%)","T=1.0 (%)"])
pre1=100 * sum((y_test - y_pred).abs() < 0.5)/len(y_test)
pre2=100 * sum((y_test - y_pred).abs() < 1)/len(y_test)
overall.at["Overall", "T=0.5 (%)"] = pre1
overall.at["Overall", "T=1.0 (%)"] = pre2

red_precision = red_precision.append(overall)
f.write(b"================================\n")
f.write(b"no resampling\n")
f.write(b"================================\n")
f.write(bytes(red_precision.to_string(), "utf-8"))

y_pred = np.rint(y_pred).astype(int)
f.write(b"\n")
rang = range(min(min(y_pred), min(y_test)),max(max(y_pred), max(y_test)) + 1)
CM = pd.DataFrame(confusion_matrix(y_test,y_pred), index = rang, columns = rang)
f.write(bytes(CM.to_string(), "utf-8"))
f.write(b"\n")
f.write(b"================================\n")
f.write(b"Minority resampling\n")
f.write(b"================================\n")
# Oversample underrepresented classes

counter = str(Counter(y_train))
f.write(bytes(counter, "utf-8"))
f.write(b"\n")
sm = SMOTE(sampling_strategy={3:456,8:456},random_state=24,k_neighbors=4)
Augmented_X_red,Augmented_Y_red=sm.fit_resample(X_train, y_train)
counter = str(Counter(Augmented_Y_red))
f.write(bytes(counter, "utf-8"))
f.write(b"\n")
clf.fit(Augmented_X_red, Augmented_Y_red)
y_pred = clf.predict(X_test)


red_precision=pd.DataFrame(index=red_class,columns=["T=0.5 (%)","T=1.0 (%)"])
for i,c in enumerate(red_class):
    label=y_test[y_test==c]
    predictor=y_pred[y_test==c]
    pre1=100*sum((label - predictor).abs() < 0.5)/len(label)
    pre2=100*sum((label - predictor).abs() < 1)/len(label)
    red_precision.at[c, "T=0.5 (%)"] = pre1
    red_precision.at[c, "T=1.0 (%)"] = pre2

overall=pd.DataFrame(index=["Overall"],columns=["T=0.5 (%)","T=1.0 (%)"])
pre1=100 * sum((y_test - y_pred).abs() < 0.5)/len(y_test)
pre2=100 * sum((y_test - y_pred).abs() < 1)/len(y_test)
overall.at["Overall", "T=0.5 (%)"] = pre1
overall.at["Overall", "T=1.0 (%)"] = pre2

red_precision = red_precision.append(overall)

f.write(bytes(red_precision.to_string(), "utf-8"))


y_pred = np.rint(y_pred).astype(int)
f.write(b"\n")
rang = range(min(min(y_pred), min(y_test)),max(max(y_pred), max(y_test)) + 1)
CM = pd.DataFrame(confusion_matrix(y_test,y_pred), index = rang, columns = rang)
f.write(bytes(CM.to_string(), "utf-8"))
f.write(b"\n")
f.write(b"================================\n")
f.write(b"Median sampling\n")
f.write(b"================================\n")

counter = str(Counter(y_train))
f.write(bytes(counter, "utf-8"))
f.write(b"\n")
median=int(statistics.median(Counter(y_train).values()))
sm = SMOTE(sampling_strategy={min(y_test):median,max(y_test):median},random_state=24,k_neighbors=4)
Augmented_X_red,Augmented_Y_red=sm.fit_resample(X_train, y_train)
counter = str(Counter(Augmented_Y_red))
f.write(bytes(counter, "utf-8"))
f.write(b"\n")

clf.fit(Augmented_X_red, Augmented_Y_red)
y_pred = clf.predict(X_test)
red_precision=pd.DataFrame(index=red_class,columns=["T=0.5 (%)","T=1.0 (%)"])
for i,c in enumerate(red_class):
    label=y_test[y_test==c]
    predictor=y_pred[y_test==c]
    pre1=100*sum((label - predictor).abs() < 0.5)/len(label)
    pre2=100*sum((label - predictor).abs() < 1)/len(label)
    red_precision.at[c, "T=0.5 (%)"] = pre1
    red_precision.at[c, "T=1.0 (%)"] = pre2

overall=pd.DataFrame(index=["Overall"],columns=["T=0.5 (%)","T=1.0 (%)"])
pre1=100 * sum((y_test - y_pred).abs() < 0.5)/len(y_test)
pre2=100 * sum((y_test - y_pred).abs() < 1)/len(y_test)
overall.at["Overall", "T=0.5 (%)"] = pre1
overall.at["Overall", "T=1.0 (%)"] = pre2

red_precision = red_precision.append(overall)
f.write(bytes(red_precision.to_string(), "utf-8"))


y_pred = np.rint(y_pred).astype(int)
f.write(b"\n")
rang = range(min(min(y_pred), min(y_test)),max(max(y_pred), max(y_test)) + 1)
CM = pd.DataFrame(confusion_matrix(y_test,y_pred), index = rang, columns = rang)
f.write(bytes(CM.to_string(), "utf-8"))
f.write(b"\n")
f.write(b"================================\n")
f.write(b"All but majority\n")
f.write(b"================================\n")

counter = str(Counter(y_train))
f.write(bytes(counter, "utf-8"))
sm = SMOTE(sampling_strategy="not majority",random_state=24,k_neighbors=4)
Augmented_X_red,Augmented_Y_red=sm.fit_resample(X_train, y_train)
counter = str(Counter(Augmented_Y_red))
f.write(bytes(counter, "utf-8"))
f.write(b"\n")

clf.fit(Augmented_X_red, Augmented_Y_red)
y_pred = clf.predict(X_test)
red_precision=pd.DataFrame(index=red_class,columns=["T=0.5 (%)","T=1.0 (%)"])
for i,c in enumerate(red_class):
    label=y_test[y_test==c]
    predictor=y_pred[y_test==c]
    pre1=100*sum((label - predictor).abs() < 0.5)/len(label)
    pre2=100*sum((label - predictor).abs() < 1)/len(label)
    red_precision.at[c, "T=0.5 (%)"] = pre1
    red_precision.at[c, "T=1.0 (%)"] = pre2

overall=pd.DataFrame(index=["Overall"],columns=["T=0.5 (%)","T=1.0 (%)"])
pre1=100 * sum((y_test - y_pred).abs() < 0.5)/len(y_test)
pre2=100 * sum((y_test - y_pred).abs() < 1)/len(y_test)
overall.at["Overall", "T=0.5 (%)"] = pre1
overall.at["Overall", "T=1.0 (%)"] = pre2

red_precision = red_precision.append(overall)
f.write(bytes(red_precision.to_string(), "utf-8"))


y_pred = np.rint(y_pred).astype(int)
f.write(b"\n")
rang = range(min(min(y_pred), min(y_test)),max(max(y_pred), max(y_test)) + 1)
CM = pd.DataFrame(confusion_matrix(y_test,y_pred), index = rang, columns = rang)
f.write(bytes(CM.to_string(), "utf-8"))

f.close()