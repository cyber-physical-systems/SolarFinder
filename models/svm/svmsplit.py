import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
%matplotlib inline

col_names =  ['id', 'location','image', 'size','pole','mean','stddev','square','ratiowh','ratioarea','approxlen','numangle','numangle90','prediction','prediction_class','label']
# load dataset
data = pd.read_csv("./lr.csv", header=None, names=col_names)
data = data.dropna()
# feature_cols = ['pole','mean','stddev','square','ratiowh','ratioarea','approxlen','numangle','numangle90','prediction']
feature_cols = ['pole','mean','stddev','square','ratiowh','ratioarea','approxlen','numangle','numangle90','prediction']
X = data[feature_cols]

scaler = StandardScaler()
X = scaler.fit_transform(X)# Features

y = data.label # Target variable
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
X_train = X
y_train = y
from sklearn.svm import SVC
svclassifier = SVC(kernel='poly',class_weight='balanced', degree=8, random_state=0)
svclassifier.fit(X_train, y_train)
# y_pred = svclassifier.predict(X_test)
# from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))

col_names =  ['id', 'location','image', 'size','pole','mean','stddev','square','ratiowh','ratioarea','approxlen','numangle','numangle90','prediction','prediction_class','label','lrpredict']
# load dataset
data = pd.read_csv("./location810/lr.csv", header=None, names=col_names)
data = data.dropna()
# feature_cols = ['pole','mean','stddev','square','ratiowh','ratioarea','approxlen','numangle','numangle90','prediction']
feature_cols = ['pole','mean','stddev','square','ratiowh','ratioarea','approxlen','numangle','numangle90','prediction']
X1 = data[feature_cols]

scaler = StandardScaler()
X1 = scaler.fit_transform(X1)# Features

y1 = data.label # Target variable
y_pred1 = svclassifier.predict(X1)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y1,y_pred1))
print(classification_report(y1,y_pred1))


