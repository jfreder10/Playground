!pip3 install --upgrade tensorflow
#alternatively in the terminal: 
#python3 -m pip install --upgrade tensorflow
import tensorflow as tf
tf.__version__
#'2.6.0'

#Create some nonlinear toy data.
import matplotlib.pyplot as plt
import numpy as np
ct = np.ones(20) 
X1 = np.random.normal(size=20) #variable, 20 rows
X2 = np.random.normal(size=20) #variable, 20 rows
X = np.array(np.column_stack((X1,X2)))
y = ct*2.2222 + X1*5.4675 + + X2*10.1115 - 3*X1**2
y = np.where(y > 0,1,0) #make it a binary problem


###Implementing a neural regression network with the functional tf.keras API

#Specify architecture
inputs = tf.keras.layers.Input(shape=(X.shape[1],), name='input') #Note: shape is a tuple and does not includes records. For a two dimensional input dataset, use (Nbrvariables,). We would use the position after the comma, if it would be a 3-dimensional tensor (e.g., images). Note that (something,) does not create a second dimension. It is just Python's way of generating a tuple (which is required by the Input layer).
hidden1 = tf.keras.layers.Dense(units=2, activation="sigmoid", name = 'hidden1')(inputs)
hidden2 = tf.keras.layers.Dense(units=2, activation="sigmoid", name= 'hidden2')(hidden1)
output = tf.keras.layers.Dense(units=1, activation = "sigmoid", name= 'output')(hidden2) #change activation function

#Create model 
model = tf.keras.Model(inputs = inputs, outputs = output)

#Compile model
model.compile(loss = 'binary_crossentropy', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001)) #change loss function

#Fit model
model.fit(x=X,y=y, batch_size=1, epochs=10) #this can be run any number of times and it will start from the last version of the weights. To reset the weights, rerun the specification to trigger the random initialization.

#making a prediction (all records)
yhat = model.predict(x=X)


