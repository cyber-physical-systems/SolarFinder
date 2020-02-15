import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
%matplotlib inline

data = pd.read_csv("./vgg_predict.csv")
data = data.dropna()


df = pd.DataFrame(data)
y1 = df.iloc[:,14].astype(int)
print(y1)
y_pred1 = df.iloc[:,16].astype(int)


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