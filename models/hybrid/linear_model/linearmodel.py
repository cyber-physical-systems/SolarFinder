mport pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report, confusion_matrix
col_names =  ['id', 'image', 'size','pole','mean','square','ratiowh','ratioarea','approxlen','numangle','numangle90','numangle70','prediction','prediction_class','label']
# load dataset
data = pd.read_csv("./train/vgg_predict.csv", header=None, names=col_names)
data = data.dropna()
# feature_cols = ['pole','mean','stddev','square','ratiowh','ratioarea','approxlen','numangle','numangle90','prediction']
feature_cols = ['pole','mean','square','ratiowh','ratioarea','approxlen','numangle','numangle90','numangle70','prediction']
# feature_cols = ['pole','prediction']
X = data[feature_cols]

scaler = StandardScaler()
X = scaler.fit_transform(X)# Features

y = data.label # Target variable

X_train = X
y_train = y
from sklearn import linear_model

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_features=4, random_state=0)
model =PassiveAggressiveClassifier(max_iter=1000, random_state=0,tol=1e-3,class_weight = 'balanced')


# fit the model with data
model.fit(X_train, y_train)

col_names =  ['id', 'image', 'size','pole','mean','square','ratiowh','ratioarea','approxlen','numangle','numangle90','numangle70','prediction','prediction_class','label','lrpredict','svmpredict']
# load dataset
data = pd.read_csv("./vgg_predict.csv", header=None, names=col_names)

data = data.dropna()
# feature_cols = ['pole','mean','stddev','square','ratiowh','ratioarea','approxlen','numangle','numangle90','prediction']
feature_cols = ['pole','mean','square','ratiowh','ratioarea','approxlen','numangle','numangle90','numangle70','prediction']

X1 = data[feature_cols]

scaler = StandardScaler()
X1 = scaler.fit_transform(X1)# Features

y1 = data.label # Target variable


y_pred1 = model.predict(X1)

    

print(confusion_matrix(y1,y_pred1 ))
print(classification_report(y1,y_pred1 ))

