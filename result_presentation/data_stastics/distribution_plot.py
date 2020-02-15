import math
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# seaborn.distplot API
# http://seaborn.pydata.org/generated/seaborn.distplot.html

col_names = ['id', 'image', 'size', 'pole', 'mean', 'stddev', 'square', 'ratiowh', 'ratioarea', 'approxlen', 'numangle', 'numangle90', 'numangle70', 'label']

data = pd.read_csv("./data/Training_set/location_1_7_all.csv",
	names=col_names)
data = data.dropna()

# print(data[:5])
# print(data.shape)

analysis_features = ['size', 'mean', 'stddev', 'square', 'ratiowh', 'ratioarea', 'approxlen', 'numangle', 'numangle90', 'numangle70']

g_plot_outputDir = './solarpanel/output/location1-7/distributions/'

positive_sample_set = data[data['label'] == 1.0]
negative_sample_set = data[data['label'] == 0.0]

for analysis_feature in analysis_features:
	
	N = max(data[analysis_feature])
	binsize = np.arange(0,N+1,math.ceil(N/100))
	if analysis_feature == 'square' or analysis_feature == 'ratiowh' or analysis_feature == 'ratioarea':
		binsize = None

	distplot_labels=['ALL', 'positive_sample_set', 'negative_sample_set']

	distplot_ked = False
	# Generate distplot
	# sns_distplot = sns.distplot(data[analysis_feature], kde=distplot_ked, label=distplot_labels[0], bins=binsize);
	sns_distplot = sns.distplot(positive_sample_set[analysis_feature], kde=distplot_ked, label=distplot_labels[1], bins=binsize)
	sns_distplot = sns.distplot(negative_sample_set[analysis_feature], kde=distplot_ked, label=distplot_labels[2], bins=binsize)
	sns_distplot.legend()

	sns_distplot.set_title(analysis_feature+'_distribution', fontsize=30)
	fig = sns_distplot.get_figure()
	fig.savefig(g_plot_outputDir + analysis_feature + '.png')
	plt.show()
	
	'''
	# Generate distplot for positive_sample_set
	sns_distplot = sns.distplot(positive_sample_set[analysis_feature], kde=distplot_ked)#, bins=binsize)

	sns_distplot.set_title(analysis_feature+'_positive_set_distribution')
	fig = sns_distplot.get_figure()
	fig.savefig(g_plot_outputDir + analysis_feature + '_positive_set_distribution.png')
	plt.show()
	'''

# pd_hist = data.groupby('label')[analysis_feature].hist(alpha=0.4)
# pd_hist = positive_sample_set.hist(column=analysis_features)
# pd_hist = negative_sample_set.hist(column=analysis_features)

# axis=0 for index, axis=1 for column
# features_only_data = data.drop(['id', 'image'], axis=1)

# sns_pairplot = sns.pairplot(features_only_data, diag_kind='kde')

# sns_pairplot.savefig(g_plot_outputDir + 'scatter' + '.png')
