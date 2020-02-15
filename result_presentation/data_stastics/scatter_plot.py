import math
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# seaborn.pairplot API
# https://seaborn.pydata.org/generated/seaborn.pairplot.html

col_names = ['id', 'image', 'size', 'pole', 'mean', 'stddev', 'square', 'ratiowh', 'ratioarea', 'approxlen', 'numangle', 'numangle90', 'numangle70', 'label']

data = pd.read_csv("./solarpanel/data/Training_set/location_1_7_all.csv",
	names=col_names)
data = data.dropna()

g_plot_outputDir = './solarpanel/output/location1-7/scatter/'

analysis_features = ['size', 'mean', 'stddev', 'square', 'ratiowh', 'ratioarea', 'approxlen', 'numangle', 'numangle90', 'numangle70']
# analysis_features = ['size', 'mean']

palette = sns.color_palette(["#e69138", "#3d85c6"])

sns.set(font_scale = 1.5)

sns_pairplot = sns.pairplot(data, vars=analysis_features,
	hue='label', markers=["o", "s"], palette=palette,
	diag_kind='kde')

sns_pairplot.savefig(g_plot_outputDir + 'scatter_plot' + '.png')

plt.show()
