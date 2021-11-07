import pandas as pd
import numpy as np
import statsmodels as stm #loads statsmodels for linear regresssion 
import matplotlib.pyplot as plt
from statsmodels  import regression as reg
import statsmodels.api as sm
import seaborn as sns
import itertools
import scipy.stats as spst #package for probability modeling 
import statistics as st
import sklearn as skl #imports sklearn 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt 
import plotly.express as px 
from math import comb
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

heart=pd.read_csv(r'C:\Users\fredr\Documents\StatTool\PythonCode\FunctionsAndCoolStuff\DATA\heart.csv')
heart['intercept']=1

#####################################################################################
##
#set vars for xs and y as arrays
#
x1=np.array(heart[['age','chol','thalach','intercept']])

y1=np.array(heart['target'])



#####################################################################################
##
#fit model and get summary 
#

modSum=sm.Logit(y1,x1)
result = modSum.fit(method='newton')
result.summary()
#####################################################################################
##
#get confusion matrix from model  
#
logreg = LogisticRegression()
model=logreg.fit(x1,y1)
model.get_params()
y_pred=logreg.predict(x1)
cm = metrics.confusion_matrix(y1, y_pred)
cm

#test the deviance if model is adequate 
modSum=sm.Logit(y1,x1)
result = modSum.fit(method='newton')
result.summary()
#get divance divide by the deg.Freedom  if <1 then log model is an adequate fit  






#########################################################################################
##
#import binomial data
##
#get prob for each obs given num of successes and num of trials per obs then get odds then log the odds and fit a linera reg model to data predicting the log odds 
playoff=pd.read_csv(r'C:\Users\fredr\Documents\StatTool\PythonCode\FunctionsAndCoolStuff\Regression\playoffs.txt',delimiter = "\t")
playoff.head()
playoff.info()
playoff.columns 
playoff['LogOdds']=np.log(playoff['Proportion']/(1-playoff['Proportion']))
playoff['LogOdds']
playoff['LogOdds'][playoff['Proportion']==0]=np.log((.001/(1-.001)))
playoff['LogOdds'][playoff['Proportion']==1]=np.log((.999/(1-.999)))
playoff['intercept']=1
#fit linear reg model predicting the log odds 
import statsmodels.api as sm
playoff
est=sm.OLS(endog=playoff['LogOdds'], exog=playoff[['Population','intercept']], missing='drop').fit()
est.summary()
#get residuals 
est.resid
fig=px.scatter(x=playoff['Population'],y=est)
fig.show() #non constant variance 

#fit logistic reg model predicting probability
y1=playoff['Proportion'] 
x1=['Population']
modSum=sm.Logit(y1,x1)
result = modSum.fit(method='newton')
result.summary()







