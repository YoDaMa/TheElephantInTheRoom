# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 21:10:42 2017

@author: gqycl
"""

from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

import pandas as pd
# Load the data
fer_data = pd.read_csv('fer2013.csv')
#fer_data.info()

uv = fer_data[['Usage']].values.ravel()
em = fer_data[['Usage']].values.ravel()
X = []
Y = []
for i in range(1,35887):
    if uv[i] == "Training": 
        x_data = fer_data['pixels'][i]
        x_data= numpy.asarray(x_data.split(' '))
        x_data = x_data.astype(int)
        X.append(x_data)
        Y.append(em[i])


"""
dim_data = 2304 #48x48 pixel shapes
# create model
model = Sequential()
model.add(Dense(12, input_dim=dim_data, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, nb_epoch=150, batch_size=10)    

# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)
"""