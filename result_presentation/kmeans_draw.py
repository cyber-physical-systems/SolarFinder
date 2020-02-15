import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import pandas as pd
import seaborn as sns
import math
from sklearn.datasets.samples_generator import (make_blobs,
                                                make_circles,
                                                make_moons)
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt  
from matplotlib import style 
from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs 

%matplotlib inline
import numpy as np




<<<<<<< HEAD
img = imread('./finaltest/data/roof_images/28.png')
=======
img = imread('')
>>>>>>> 39db66de7b321f1d8347e674b5c8fa5f34ff3b62
img_size = img.shape

print(img_size)
# Reshape it to be 2-dimension
X = img.reshape(img_size[0] * img_size[1], img_size[2])
print(X.shape)

cost =[] 
for i in range(1, 11): 
    KM = KMeans(n_clusters = i, max_iter = 100) 
    KM.fit(X) 
      
    # calculates squared error 
    # for the clustered points 
    cost.append(KM.inertia_)      
  
# plot the cost against K values 
plt.plot(range(1, 11), cost, color ='g', linewidth ='3')
# plt.rcParams.update({'font.size': 22})
plt.xlabel("Value of K", {'size': 14}) 
plt.ylabel("Sqaured Error (Cost)", {'size': 14}) 
plt.tight_layout()
<<<<<<< HEAD
plt.savefig("./data/roof_images/square_error28.png")
=======
plt.savefig("")
>>>>>>> 39db66de7b321f1d8347e674b5c8fa5f34ff3b62
plt.show() # clear the plot 