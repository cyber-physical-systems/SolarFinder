import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
col_names =  ['id', 'image', 'size','pole','mean','square','ratiowh','ratioarea','approxlen','numangle','numangle90','numangle70','prediction','prediction_class','label']
# load dataset
data = pd.read_csv("./vgg_predict.csv", header=None, names=col_names)
data = data.dropna()
# feature_cols = ['pole','mean','stddev','square','ratiowh','ratioarea','approxlen','numangle','numangle90','prediction']
feature_cols = ['pole','mean','square','ratiowh','ratioarea','approxlen','numangle','numangle90','numangle70']
# feature_cols = ['pole','prediction']
X = data[feature_cols]
y = data.label 
scaler = StandardScaler()
X = scaler.fit_transform(X)# Features
from sklearn.decomposition import PCA

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
pca = PCA(n_components=6)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


from sklearn.svm import SVC
svclassifier = SVC(kernel='poly',degree = 7,class_weight='balanced', random_state=0)
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

