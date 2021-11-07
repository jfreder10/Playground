#!pip3 install --upgrade tensorflow
#alternatively in the terminal: 
#python3 -m pip install --upgrade tensorflow
import tensorflow as tf
tf.__version__

#############################################
#Create some nonlinear toy data.
import matplotlib.pyplot as plt
import numpy as np
ct = np.ones(20) 
X1 = np.random.normal(size=20) #variable, 20 rows
X2 = np.random.normal(size=20) #variable, 20 rows
X = np.array(np.column_stack((X1,X2)))
y = ct*2.2222 + X1*5.4675 + X2*10.1115 - 3*X1**2

ycat = []
for i in y:
    if i <= np.quantile(y,0.25):
        ycat.append(['cat 0'])
    elif i <= np.quantile(y,0.35):
        ycat.append(['cat 1'])
    elif i <= np.quantile(y,0.50):
        ycat.append(['cat 2'])        
    elif i <= np.quantile(y,0.75):        
        ycat.append(['cat 3'])        
    elif i <= np.max(y):        
        ycat.append(['cat 4'])

ycat

#predictor variables are X
#categorical response variable is ycat




####################################################
####################################################
####################################################
# Dense implementation
####################################################
####################################################
####################################################

#############################################
#preprocess the data

#step 1: get unique categories
#flatten nested list
y_cat_flat = [item for subcat in ycat for item in subcat]
unique_categories = np.unique(np.array(y_cat_flat))
unique_categories

#step 2: make lookup table with index value for each category
indices = np.array(range(len(unique_categories)), dtype = np.int64)
indices
lookuptable = np.column_stack([unique_categories,indices])

#step 3: apply lookup table to data
main_result = []
for i in range(len(ycat)):
    res = []
    for ii in range(len(ycat[i])):
        res.append(int(lookuptable[lookuptable[:,0]==ycat[i][ii],1][0]))
    main_result.append(res)

main_result

#step 4: create dummy encoded data

y_final = np.array([list(np.zeros(len(unique_categories))) for i in range(len(ycat))])
for i in range(len(ycat)):
    for ii in range(len(main_result[i])):
        y_final[i,main_result[i][ii]] = 1        

y_final





###Implementing a neural network with the functional tf.keras API

#Specify architecture
inputs = tf.keras.layers.Input(shape=(X.shape[1],), name='input') #Note: shape is a tuple and does not includes records. For a two dimensional input dataset, use (Nbrvariables,). We would use the position after the comma, if it would be a 3-dimensional tensor (e.g., images). Note that (something,) does not create a second dimension. It is just Python's way of generating a tuple (which is required by the Input layer).
hidden1 = tf.keras.layers.Dense(units=2, activation="sigmoid", name = 'hidden1')(inputs)
hidden2 = tf.keras.layers.Dense(units=2, activation="sigmoid", name= 'hidden2')(hidden1)
outputs = tf.keras.layers.Dense(units=y_final.shape[1], activation = "softmax", name= 'output')(hidden2) #changed activation to softmax


#Create model 
model = tf.keras.Model(inputs = inputs, outputs = outputs)

#Compile model
model.compile(loss = 'categorical_crossentropy', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001)) #loss function changed to categorical_crossentropy

#Fit model
model.fit(x=X,y=y_final, batch_size=1, epochs=10) #this can be run any number of times and it will start from the last version of the weights. To reset the weights, rerun the specification to trigger the random initialization.

#making a prediction (all records)
yhat = model.predict(x=X)

model.evaluate(x=X,y=y_final)



####################################################
####################################################
####################################################
# Sparse implementation
####################################################
####################################################
####################################################


#step 1: get unique categories
#flatten nested list
y_cat_flat = [item for subcat in ycat for item in subcat]
unique_categories = np.unique(np.array(y_cat_flat))
unique_categories

#step 2: make lookup table with index value for each category
indices = np.array(range(len(unique_categories)), dtype = np.int64)
indices
lookuptable = np.column_stack([unique_categories,indices])

#step 3: apply lookup table to data
main_result = []
for i in range(len(ycat)):
    res = []
    for ii in range(len(ycat[i])):
        res.append(int(lookuptable[lookuptable[:,0]==ycat[i][ii],1][0]))
    main_result.append(res)

main_result = np.array(main_result)

main_result



###Implementing a neural regression network with the functional tf.keras API

#Specify architecture
inputs = tf.keras.layers.Input(shape=(X.shape[1],), name='input') #Note: shape is a tuple and does not includes records. For a two dimensional input dataset, use (Nbrvariables,). We would use the position after the comma, if it would be a 3-dimensional tensor (e.g., images). Note that (something,) does not create a second dimension. It is just Python's way of generating a tuple (which is required by the Input layer).
hidden1 = tf.keras.layers.Dense(units=2, activation="sigmoid", name = 'hidden1')(inputs)
hidden2 = tf.keras.layers.Dense(units=2, activation="sigmoid", name= 'hidden2')(hidden1)
outputs = tf.keras.layers.Dense(units=y_final.shape[1], activation = "softmax", name= 'output')(hidden2) #changed activation to softmax


#Create model 
model = tf.keras.Model(inputs = inputs, outputs = outputs)

#Compile model
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001)) #loss function changed to sparse_categorical_crossentropy

#Fit model
model.fit(x=X,y=main_result, batch_size=1, epochs=10) #this can be run any number of times and it will start from the last version of the weights. To reset the weights, rerun the specification to trigger the random initialization.

#making a prediction (all records)
yhat = model.predict(x=X)

model.evaluate(x=X,y=main_result)
