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
%matplotlib inline



data = pd.read_csv("./svmrbftrainprobility.csv")
data = data.dropna()
# feature_cols = ['vgg_pro','vgg_class','svmrbf_class','svmrbfpro']
feature_cols = ['vgg_pro','svmrbfpro']
X = data[feature_cols]

scaler = StandardScaler()
X = scaler.fit_transform(X)# Features

y = data.label # Target variable


X_train = X
y_train = y




#  use linear regression 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight = 'balanced')

# instantiate the model (using the default parameters)

# fit the model with data
model.fit(X_train, y_train)
# from sklearn.externals import joblib
# from joblib import dump, load
# dump(model, 'svmrbfhybrid.joblib') 
# model = load('svmrbfhybrid.joblib') 
print(model.coef_ )
print(model.intercept_ )
from sklearn import metrics




datatest = pd.read_csv("./svmrbftestpro.csv")
datatest = datatest.dropna()
# feature_cols = ['vgg_pro','vgg_class','svmrbf_class','svmrbfpro']
feature_cols = ['vgg_pro','svmrbfpro']
Xtest = datatest[feature_cols]
scaler = StandardScaler()
Xtest = scaler.fit_transform(Xtest)# Features
ytest = datatest.label # Target variable
y_predict_vgg = datatest.vgg_pro
y_predict_svm = datatest.svmrbfpro



y_predict= model.predict(Xtest)
y_predict_pro = model.predict_proba(Xtest)
y_predict_pro = y_predict_pro[:, 1]



df = pd.DataFrame(datatest) 
df.insert(25, "svm_nosplit_pro", y_predict_pro, True) 
df.insert(26, "svm_nosplit_class", y_predict, True) 

export_csv = df.to_csv ('./vggsvmlogicalregression2features.csv', index = None)
print(confusion_matrix(ytest, y_predict))
tn, fp, fn, tp = confusion_matrix(ytest, y_predict, labels=[0,1]).ravel()
print(tn,fp,fn,tp)
with open('./result.csv', 'a') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['vggsvmlogicalregression2features.csv',tn,fp,fn,tp])
csvfile.close()
time = time.time() - start_time
with open('./time.csv', 'a') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['vggsvmlogicalregression2features.csv',time])
csvfile.close()



from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot
print(confusion_matrix(ytest, y_predict))
print(classification_report(ytest, y_predict))
print(accuracy_score(ytest, y_predict))
print(balanced_accuracy_score(ytest, y_predict))
print(metrics.precision_score(ytest, y_predict))
print(metrics.recall_score(ytest, y_predict))
print(metrics.f1_score(ytest, y_predict))
print(matthews_corrcoef(ytest, y_predict))
print(roc_auc_score(ytest, y_predict))
print(roc_auc_score(ytest, y_predict_vgg ))
print(roc_auc_score(ytest, y_predict))
lr_fpr, lr_tpr, _ = roc_curve(ytest, y_predict_pro)
lr_fpr_vgg, lr_tpr_vgg, _ = roc_curve(ytest, y_predict_vgg )
lr_fpr_svm, lr_tpr_svm, _ = roc_curve(ytest, y_predict_svm)
pyplot.plot(lr_fpr, lr_tpr, marker='x', label='Logistic')
pyplot.plot(lr_fpr_vgg, lr_tpr_vgg, marker='o', label='vgg')
pyplot.plot(lr_fpr_svm, lr_tpr_svm, marker='v', label='svm kernel=rbf')
pyplot.xlabel('False Positive Rate',{'size': 14})
pyplot.ylabel('True Positive Rate',{'size': 14})
# show the legend
pyplot.legend()
pyplot.tight_layout()
pyplot.savefig('./split_roc.png')
# show the plot
pyplot.show()








