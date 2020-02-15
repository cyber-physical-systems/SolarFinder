import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

filepath = './vgg_predict.csv' #your path here
data = np.genfromtxt(filepath, delimiter=',', dtype='float64')

scaler = MinMaxScaler(feature_range=[0, 1])
data_rescaled = scaler.fit_transform(data[1:, 3:13])
#Fitting the PCA algorithm with our Data
pca = PCA().fit(data_rescaled)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Pulsar Dataset Explained Variance')
plt.savefig('pca.png')
plt.show()

