# %matplotlib inline
import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

<<<<<<< HEAD
# col_names = ['id','image','size','pole','mean','square','ratiowh','ratioarea','approxlen','numangle','numangle90','numangle70','label']
# # load dataset
# data = pd.read_csv("./mean.csv", names=col_names)
# data = data.dropna()

# df = pd.DataFrame(data, columns=["size",'label'])
# sns.jointplot(x="size", y="label", data=df)
# plt.savefig("out.png")
# mean, cov = [0, 1], [(1, .5), (.5, 1)]
# data = np.random.multivariate_normal(mean, cov, 200)
# df = pd.DataFrame(data, columns=["x", "y"])
# sns.barplot(x="x", y="y", data=df);

import seaborn as sns
sns.set(style="darkgrid")
titanic = pd.read_csv("./mean.csv")
=======
import seaborn as sns
sns.set(style="darkgrid")
titanic = pd.read_csv("")
>>>>>>> 39db66de7b321f1d8347e674b5c8fa5f34ff3b62
ax = sns.countplot(x="mean", data=titanic)