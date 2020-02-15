import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
%matplotlib inline

col_names =  ['id', 'image', 'size','pole','mean','square','ratiowh','ratioarea','approxlen','numangle','numangle90','numangle70','prediction','prediction_class','label']
# load dataset
data = pd.read_csv(".vgg_predict.csv", header=None, names=col_names)
data = data.dropna()
# feature_cols = ['pole','mean','stddev','square','ratiowh','ratioarea','approxlen','numangle','numangle90','prediction']
feature_cols = ['pole','mean','square','ratiowh','ratioarea','approxlen','numangle','numangle90','numangle70']
# feature_cols = ['pole','prediction']
X = data[feature_cols]

scaler = StandardScaler()
X = scaler.fit_transform(X)# Features

y = data.label # Target variable

X_train = X
y_train = y
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0,class_weight='balanced')

model = clf.fit(X_train, y_train)
# # y_pred = svclassifier.predict(X_test)
# # from sklearn.metrics import classification_report, confusion_matrix
# # print(confusion_matrix(y_test,y_pred))
# # print(classification_report(y_test,y_pred))

col_names =  ['id', 'image', 'size','pole','mean','square','ratiowh','ratioarea','approxlen','numangle','numangle90','numangle70','prediction','prediction_class','label','lrpredict','svmpredict']
# load dataset
data = pd.read_csv("./vgg_predict.csv", header=None, names=col_names)

data = data.dropna()
# # feature_cols = ['pole','mean','stddev','square','ratiowh','ratioarea','approxlen','numangle','numangle90','prediction']
feature_cols = ['pole','mean','square','ratiowh','ratioarea','approxlen','numangle','numangle90','numangle70']

X1 = data[feature_cols]

scaler = StandardScaler()
X1 = scaler.fit_transform(X1)# Features

y1 = data.label # Target variable
y_pred1 = model.predict(X1)


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
print(confusion_matrix(y1,y_pred1))
print(classification_report(y1,y_pred1))
print(accuracy_score(y1,y_pred1))
print(balanced_accuracy_score(y1,y_pred1))
print(metrics.precision_score(y1,y_pred1))
print(metrics.recall_score(y1,y_pred1))
print(metrics.f1_score(y1,y_pred1))
print(matthews_corrcoef(y1,y_pred1))
print(roc_auc_score(y1,y_pred1))