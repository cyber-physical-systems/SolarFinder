from keras.models import Sequential
from keras.layers import Dense
import numpy
import os
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt('./lr_train.csv', delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,2:13]
Y = dataset[:,14]
# create model
model = Sequential()
model.add(Dense(12, input_dim=11, init='uniform', activation='relu')) 
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Fit the model
model.fit(X, Y, nb_epoch=5, batch_size=128)
datasettest = numpy.loadtxt('./vggtest.csv', delimiter=",")
# split into input (X) and output (Y) variables
Xtest = datasettest[:,2:13]
Ytest = datasettest[:,14]


# evaluate the model
scores = model.evaluate(Xtest, Ytest)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))





