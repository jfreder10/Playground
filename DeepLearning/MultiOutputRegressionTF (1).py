#Note:  
#here we are going to create two outputs using the most flexible way. It allows us to use a separate loss function and weight per loss.
#in the multilabel classification example, we will see another way to use multiple outputs with less flexibility by simply using two units in the output layer.

#!pip3 install --upgrade tensorflow
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
y1 = ct*2.2222 + X1*5.4675 + X2*10.1115 - 3*X1**2
y2 = ct*3.3332 + X1*4.4766 + X2*10.4572 - 6*X1**2


###Implementing a neural regression network with the functional tf.keras API

#Specify architecture
inputs = tf.keras.layers.Input(shape=(X.shape[1],), name='input') #Note: shape is a tuple and does not includes records. For a two dimensional input dataset, use (Nbrvariables,). We would use the position after the comma, if it would be a 3-dimensional tensor (e.g., images). Note that (something,) does not create a second dimension. It is just Python's way of generating a tuple (which is required by the Input layer).
hidden1 = tf.keras.layers.Dense(units=2, activation="sigmoid", name = 'hidden1')(inputs)
hidden2 = tf.keras.layers.Dense(units=2, activation="sigmoid", name= 'hidden2')(hidden1)
output1 = tf.keras.layers.Dense(units=1, activation = "linear", name= 'output1')(hidden2)
output2 = tf.keras.layers.Dense(units=1, activation = "linear", name= 'output2')(hidden2)

#Create model 
model = tf.keras.Model(inputs = inputs, outputs = [output1,output2])

#Compile model
model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001))
#same loss function is used for both outputs
#We can do this explicitly as follows:
model.compile(loss = ['mse','mse'], optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001))
#TensorFlow will compute the two losses and simply take the average of them to get the final loss used for training
#This is the same as:
model.compile(loss = ['mse','mse'], loss_weights = [0.5,0.5], optimizer = tf.keras.optimizers.SGD(lr = 0.001))
#This means we can give a different weight to each loss when they are averaged.

#Fit model
model.fit(x=X,y=[y1,y2], batch_size=1, epochs=10) #this can be run any number of times and it will start from the last version of the weights. To reset the weights, rerun the specification to trigger the random initialization.

#making a prediction (first record)
yhat1, yhat2 = model.predict(x=X[0:1])

#getting the average loss, and individual losses
model.evaluate(X,[y1,y2])

#summarizing the model
model.summary()

