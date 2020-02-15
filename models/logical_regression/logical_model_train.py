import pandas
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
col_names =  ['id', 'location', 'image', 'size','pole','mean','stddev','square','ratiowh','ratioarea','approxlen','numangle','numangle90','prediction','prediction_class','label']
# load dataset
data = pd.read_csv("./location810/lr.csv", header=None, names=col_names)
data = data.dropna()
feature_cols = ['pole','mean','stddev','square','ratiowh','ratioarea','approxlen','numangle','numangle90','prediction']
X = data[feature_cols]

scaler = StandardScaler()
X = scaler.fit_transform(X)# Features

y = data.label # Target variable

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# instantiate the model (using the default parameters)
model = LogisticRegression(class_weight = 'balanced')

# fit the model with data
model.fit(X_train, y_train)
print(model.coef_ )
print(model.intercept_ )
filename = 'RLmodel.sav'
pickle.dump(model, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)
y_predict= model.predict(X_test)
print("Y predict/hat ", y_predict)
print(metrics.confusion_matrix(y_test, y_predict))









import pandas
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
col_names =  ['id','image', 'size','pole','mean','square','ratiowh','ratioarea','approxlen','numangle','numangle90','numangle70','prediction','prediction_class','label']
# load dataset
data = pd.read_csv("./vgg_predict.csv", header=None, names=col_names)
data = data.dropna()
feature_cols = ['pole','mean','square','ratiowh','ratioarea','approxlen','numangle','numangle90','numangle70','prediction']
X = data[feature_cols]

scaler = StandardScaler()
X = scaler.fit_transform(X)# Features

y = data.label # Target variable


filename ='./nosplit/' + 'RLmodel.sav'
# pickle.dump(model, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X, y)
print(result)
y_predict= loaded_model.predict(X)
print("Y predict/hat ", y_predict)
print(metrics.confusion_matrix(y, y_predict))

y_predict= loaded_model.predict(X)
print(y_predict)

  
# Convert the dictionary into DataFrame 
df = pd.DataFrame(data) 
  
# Using DataFrame.insert() to add a column 
df.insert(15, "predict", y_predict, True) 
  
# Observe the result 

export_csv = df.to_csv ('./vgg_predict.csv', index = None, header=False)




