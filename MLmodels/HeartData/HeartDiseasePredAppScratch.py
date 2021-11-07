import sklearn
import pandas as pd 

import numpy as np
from sklearn import datasets, ensemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import statistics as st
import itertools
from sklearn.model_selection import cross_val_score
from sklearn.experimental import enable_hist_gradient_boosting  
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from statsmodels  import regression as reg
import statsmodels.api as regMods 
from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import expon
from scipy.stats import poisson
from scipy.stats import binom
from scipy.stats import t 
import plotly.express as px
import plotly.figure_factory as ff

heart=pd.read_csv(r"C:\Users\fredr\Documents\StatTool\BZAN540\Homework\HW6\HeartDisease.csv")  
heart.columns

train, test = train_test_split(heart[['x1', 'x2', 'x3', 'x4', 'x5','HeartDisease']], test_size=0.2)
y_train=train['HeartDisease']
x_train=train[['x1', 'x2', 'x3', 'x4', 'x5']]

x_test=test[['x1', 'x2', 'x3', 'x4', 'x5']]
y_test=test['HeartDisease']

#boosting to predict heart disease 

#make expand grid function to get all combos of the parameters 
def expandgrid(*itrs):
   product = list(itertools.product(*itrs))
   return {'Var{}'.format(i+1):[x[i] for x in product] for i in range(len(itrs))}

#set the range for the parameter values:
n_estimators=np.arange(300, 450, 50) #the number of trees to fit 
max_depth=np.arange(3, 5, 1)
min_samples_split=np.arange(3,4,1)
learning_rate=np.arange(0.001,0.004,0.001)
a=expandgrid(n_estimators,max_depth, min_samples_split,learning_rate)
params=pd.DataFrame.from_dict(a)
len(params)

#time the code ??? 
#looping through the possible parameters for the model and store the estimated validation rmse
ValAcc=list(range(0,len(params)))
for i in range(0,len(params)):
    scores = cross_val_score(HistGradientBoostingClassifier(min_samples_leaf=params['Var3'].iloc[i],
    max_depth=params['Var2'].iloc[i],
    learning_rate=params['Var4'].iloc[i],max_iter=params['Var1'].iloc[i]).fit(x_train, y_train), 
    x_train, y_train, cv=4,scoring='accuracy')
    acc=st.mean(scores)
    ValAcc[i]=acc

ValAcc
max(ValAcc)
pars=list(params.iloc[ValAcc==max(ValAcc)].iloc[0])
pars.append(max(ValAcc))
pars
bestPos=np.array(np.where(np.array(ValAcc)==max(ValAcc))).tolist()[0][0]
#fit the best model on Train then predict on Test if mean acc close to val then fit on entire data 
bestPos

bestMod=HistGradientBoostingClassifier(min_samples_leaf=params['Var3'].iloc[bestPos],
    max_depth=params['Var2'].iloc[bestPos],
    learning_rate=params['Var4'].iloc[bestPos],max_iter=params['Var1'].iloc[bestPos]).fit(x_train, y_train)

#gets the predicted values on the test data 
bestMod.predict(x_test)
len(y_test[bestMod.predict(x_test)==y_test])/len(y_test) #67% acc on test 
#create a dataset with one row and each col is a ind var from model fit above, then input data per var to fill df then predict y on the values in this df 
df_i=pd.DataFrame({'x1':np.mean(heart['x1']), 'x2':np.mean(heart['x2']),'x3':np.mean(heart['x3']),'x4':np.mean(heart['x4']),'x5':np.mean(heart['x5'])},index=[0])

if(bestMod.predict(df_i)==0):
    print('Predicted: No Heart Disease')
else:
    print('Predicted: Has Heart Disease')


#plot two densities centered on the mean of the var and the selected value of the var for all vars 
#start with treating each var as a normal distro then plot a density curve where the 
#mean is the mean of the var and another curve on same plot where the mean is the selected value from the input of a normal distro set sd to the sd of the var 
#for both in the plots, generate random vars the size of the data, except for history heart disease treat as beta with p=actuap prob for var and p=random value that is
#greater than .5 

#generates random values from a normal distro with mean=loc and sd=scale 
norm.rvs(size=10000,loc=3,scale=8)
#x1:
x1=190
mean=np.mean(heart['x1'])
sd=np.std(heart['x1'])
meanx1_2=x1
xActual=norm.rvs(size=len(heart),loc=mean,scale=sd)
xInput=norm.rvs(size=len(heart),loc=meanx1_2,scale=sd)

group_labels = ['actual','center_selected']
hist_data=[xActual,xInput]
fig = ff.create_distplot(hist_data,group_labels)
fig.show()