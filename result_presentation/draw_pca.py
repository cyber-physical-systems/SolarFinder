import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
%matplotlib inline
filepath = './feature_test_all.csv' #your path here
data = np.genfromtxt(filepath, delimiter=',', dtype='float64')

scaler = MinMaxScaler(feature_range=[0, 1])
data_rescaled = scaler.fit_transform(data[1:, 3:19])
#Fitting the PCA algorithm with our Data
pca = PCA().fit(data_rescaled)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_), linewidth ='3')
plt.xlabel('Number of Components',{'size': 14})
plt.ylabel('Variance',{'size': 14}) #for each component
# plt.title('Pulsar Dataset Explained Variance')
plt.tight_layout()
plt.savefig('./finaltest/data/pca.png')
plt.show()