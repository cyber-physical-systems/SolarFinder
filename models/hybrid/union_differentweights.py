import pandas
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
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
dataset1 = pd.read_csv("./non_split_test_result.csv")
dataset1 = dataset1.dropna()
df = pd.DataFrame(dataset1) 

# def f(x,y):
# #     print(x,y)
#     return round(0.5*x + 0.5*y)
    
ytest1 = dataset1.label


y_predict1=dataset1.hard_pred_label
print(confusion_matrix(ytest1, y_predict1))
tn, fp, fn, tp = confusion_matrix(ytest1, y_predict1, labels=[0,1]).ravel()
print(tn,fp,fn,tp)


