import numpy as np
import pandas as pd

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