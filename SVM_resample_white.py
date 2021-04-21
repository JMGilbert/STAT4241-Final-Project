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

# Fitting the SVM for white wine

f = open("whiteres_SVM.txt", "wb")
whitedat = pd.read_csv("./winequality-white.csv", sep = ";")
X_white = whitedat[whitedat.columns[0:11]]
y_white = whitedat[whitedat.columns[11]]

white_class=pd.unique(whitedat['quality'])
white_class.sort()


clfknn = KNeighborsClassifier(n_neighbors=3)
clfknn.fit(X_white,y_white)

y_pred = clfknn.predict(X_white)

sigma2 = (y_white - y_pred).var() * 1.5
epsilon = sigma2/math.sqrt(len(y_white))

clf = SVR(kernel = 'rbf', epsilon = epsilon, C = 3, gamma=2**-2)

wdcopy = whitedat.copy()
X_train, X_test, y_train, y_test = train_test_split(X_white, y_white, test_size=0.33, random_state=42,stratify=y_white)

clf.fit(X_train, y_train)
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
f.write(b"================================\n")
f.write(b"no resampling\n")
f.write(b"================================\n")
f.write(bytes(white_precision.to_string(), "utf-8"))

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
sm = SMOTE(sampling_strategy={3:1455,9:1455},random_state=24,k_neighbors=2)
Augmented_X_white,Augmented_Y_white=sm.fit_resample(X_train, y_train)
counter = str(Counter(Augmented_Y_white))
f.write(bytes(counter, "utf-8"))
f.write(b"\n")
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

f.write(bytes(white_precision.to_string(), "utf-8"))


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
sm = SMOTE(sampling_strategy={min(y_test):median,max(y_test):median},random_state=24,k_neighbors=2)
Augmented_X_white,Augmented_Y_white=sm.fit_resample(X_train, y_train)
counter = str(Counter(Augmented_Y_white))
f.write(bytes(counter, "utf-8"))
f.write(b"\n")

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
f.write(bytes(white_precision.to_string(), "utf-8"))


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
f.write(b"\n")
sm = SMOTE(sampling_strategy="not majority",random_state=24,k_neighbors=2)
Augmented_X_white,Augmented_Y_white=sm.fit_resample(X_train, y_train)
counter = str(Counter(Augmented_Y_white))
f.write(bytes(counter, "utf-8"))
f.write(b"\n")

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
f.write(bytes(white_precision.to_string(), "utf-8"))


y_pred = np.rint(y_pred).astype(int)
f.write(b"\n")
rang = range(min(min(y_pred), min(y_test)),max(max(y_pred), max(y_test)) + 1)
CM = pd.DataFrame(confusion_matrix(y_test,y_pred), index = rang, columns = rang)
f.write(bytes(CM.to_string(), "utf-8"))

f.close()