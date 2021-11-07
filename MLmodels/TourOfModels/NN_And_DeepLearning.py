
import pandas as pd  
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras_visualizer import visualizer 
from keras import layers 
from keras import models 
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from ann_visualizer.visualize import ann_viz


###################################################################################
##
#keras
##
#predicting the type of wine 
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')
heart=pd.read_csv(r"C:\Users\fredr\Documents\StatTool\BZAN540\Homework\HW6\HeartDisease.csv")  

heart
#get the prob of heart disease for discritinzed groups of age 

white.head()
red.head()
white['type']=0 
red['type']=1
wine=pd.concat([white,red],axis=0)
len(x_train.columns)

heart.head() 
heart.drop(['Case'],axis=1,inplace=True)
train,test=train_test_split(heart, test_size=0.20, random_state=42)
x_train=train.drop(['HeartDisease'],axis=1) 
x_train.head()
y_train=train['HeartDisease']
x_test=test.drop(['HeartDisease'],axis=1) 
y_test=test['HeartDisease']


#define model, add layers with node size and function 
model = Sequential()
model.add(Dense(12, input_dim=len(x_train.columns), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(np.array(x_train), np.array(y_train), epochs=500, batch_size=10)
# evaluate the keras model
accuracy = model.evaluate(np.array(x_test), np.array(y_test))
print('Accuracy: %.2f' % (accuracy*100))

#plot network 
#save as .json 


ann_viz(model, view=True, filename="network.gv")
###################################################################################
##
#Pytorch
##



###################################################################################
##
#TensorFlow 
##