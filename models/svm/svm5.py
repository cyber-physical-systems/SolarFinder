import pandas
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import csv
import time
start_time = time.time()



data = pd.read_csv("./feature_17_all.csv")
data = data.dropna()
feature_cols = ['size','pole','mean','stddev','b_mean','g_mean','r_mean','b_stddev','g_stddev','r_stddev','square','ratiowh','ratioarea','approxlen','numangle','numangle90','numangle70']
X = data[feature_cols]

scaler = StandardScaler()
X = scaler.fit_transform(X)# Features

y = data.label # Target variable

X_train = X
y_train = y

from sklearn.svm import SVC
svclassifier = SVC(kernel='poly',class_weight='balanced', degree=5, random_state=0,probability=True)
model = svclassifier.fit(X_train, y_train)


from sklearn import metrics

# instantiate the model (using the default parameters)


# fit the model with data
model.fit(X_train, y_train)



datatest = pd.read_csv("./feature_810_all.csv")
datatest = datatest.dropna()
feature_cols = ['size','pole','mean','stddev','b_mean','g_mean','r_mean','b_stddev','g_stddev','r_stddev','square','ratiowh','ratioarea','approxlen','numangle','numangle90','numangle70']
Xtest = datatest[feature_cols]
scaler = StandardScaler()
Xtest = scaler.fit_transform(Xtest)# Features
ytest = datatest.label # Target variable

y_predict= model.predict(Xtest)
y_pro = model.predict_proba(Xtest)[:,1]



df = pd.DataFrame(datatest) 
df.insert(23, "svm_poly5_class", y_predict, True) 
df.insert(24, "svm_poly5_pro", y_predict, True) 
export_csv = df.to_csv ('./svm_poly5.csv', index = None)
print(confusion_matrix(ytest, y_predict))
tn, fp, fn, tp = confusion_matrix(ytest, y_predict, labels=[0,1]).ravel()
print(tn,fp,fn,tp)
with open('./result.csv', 'a') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['svm_poly5',tn,fp,fn,tp])
csvfile.close()
time = time.time() - start_time
with open('./time.csv', 'a') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['svm_poly5',time])
csvfile.close()








