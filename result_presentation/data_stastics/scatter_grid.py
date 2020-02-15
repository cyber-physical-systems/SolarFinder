import math
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# seaborn.PairGrid API
# https://seaborn.pydata.org/generated/seaborn.PairGrid.html#seaborn.PairGrid

col_names = ['id', 'image', 'size', 'pole', 'mean', 'stddev', 'square', 'ratiowh', 'ratioarea', 'approxlen', 'numangle', 'numangle90', 'numangle70', 'label']

data = pd.read_csv("./solarpanel/data/Training_set/location_1_7_all.csv",
	names=col_names)
data = data.dropna()

g_plot_outputDir = './solarpanel/output/location1-7/scatter/'

analysis_features = ['size', 'mean', 'stddev', 'square', 'ratiowh', 'ratioarea', 'approxlen', 'numangle', 'numangle90', 'numangle70']
# analysis_features = ['size', 'mean']

palette = sns.color_palette(["#e69138", "#3d85c6"])

# sns.set(font_scale = 1.5)
sns.set_context(rc={'axes.labelsize': 25.0, 'xtick.labelsize': 'small', 'ytick.labelsize': 'small', 'axes.linewidth': 0, 'ytick.major.size': 0, 'xtick.major.size': 0})
# print(sns.plotting_context())

sns_pairplot = sns.PairGrid(data, vars=analysis_features,
	hue='label', hue_kws={"marker": ["o", "s"]}, palette=palette)
sns_pairplot = sns_pairplot.map(plt.scatter, linewidths=1, edgecolor="w", s=40)
# sns_pairplot = sns_pairplot.add_legend()

plt.subplots_adjust(hspace = 0.01, wspace = 0.01)
sns_pairplot.savefig(g_plot_outputDir + 'scatter_grid' + '.png')
plt.show()