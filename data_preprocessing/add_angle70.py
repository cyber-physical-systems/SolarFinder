import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import csv


data = pd.read_csv(".csv")
df = pd.DataFrame(data)
data1 =pd.read_csv("/contour_all.csv")
df1 = pd.DataFrame(data1)

angle70 = df1.iloc[:,13]
df.insert(13, "numangle70", angle70, True)

export_csv = df.to_csv ('/location810/angle70.csv'，index=None)



