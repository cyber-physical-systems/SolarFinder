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



data = pd.read_csv("./svmrbftrainprobility.csv")
data = data.dropna()
feature_cols = ['vgg_pro','vgg_class','svmrbf_class','svmrbfpro']
X = data[feature_cols]

scaler = StandardScaler()
X = scaler.fit_transform(X)# Features

y = data.label # Target variable

# from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
X_train = X
y_train = y

from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf',class_weight='balanced')
model = svclassifier.fit(X_train, y_train)


# instantiate the model (using the default parameters)

# fit the model with data
model.fit(X_train, y_train)
from sklearn.externals import joblib
from joblib import dump, load
dump(model, 'svmrbfhybrid.joblib') 
# model = load('svmrbfhybrid.joblib') 

from sklearn import metrics




datatest = pd.read_csv("./svmrbftestpro.csv")
datatest = datatest.dropna()
feature_cols = ['vgg_pro','vgg_class','svmrbf_class','svmrbfpro']
Xtest = datatest[feature_cols]
scaler = StandardScaler()
Xtest = scaler.fit_transform(Xtest)# Features
ytest = datatest.label # Target variable

y_predict= model.predict(Xtest)


df = pd.DataFrame(datatest) 
df.insert(25, "hybrid", y_predict, True) 

export_csv = df.to_csv ('./svmrbftestprohybrid.csv', index = None)
print(confusion_matrix(ytest, y_predict))
tn, fp, fn, tp = confusion_matrix(ytest, y_predict, labels=[0,1]).ravel()
print(tn,fp,fn,tp)
with open('./result.csv', 'a') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['hybrid',tn,fp,fn,tp])
csvfile.close()
time = time.time() - start_time
with open('./time.csv', 'a') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['hybrid',time])
csvfile.close()









