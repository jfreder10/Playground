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
import statsmodels.api as regMods
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt 
import plotly.express as px 
from math import comb

heart=pd.read_csv(r'C:\Users\fredr\Documents\StatTool\PythonCode\FunctionsAndCoolStuff\DATA\heart.csv')
heart['intercept']=1

#A.number of possible models is choose numx k 
#B.number of 2 way interactions is choose k 2
#C.number of m 2 way interaction vars is choose num2wayInts m 
#A.*C.

#for each ith possible var combo loop through the j interactions the size k out of the number of selected vars for the ith combo 
#then need to loop through the interactions selected to create the interaction vars 

#define all possible x vars to use 
xvars=np.array(['age','sex','cp','chol','oldpeak','thalach'],dtype=str)
#get all combinations of the x variables of size k 
k=4
predictors=[i for i in itertools.combinations(xvars, k)] 
predictors[0]
#get all combos of 2 way interactions (of 2 variables from the ith predictors list)
#in outter loop right before inner 
twoways=[i for i in itertools.combinations(predictors[0], 2)]
len(twoways)
twoways 
#select the number of 2 way interactions to use in a model from the possible 2 way interactions 
m=2 #m is the number of possible 2 way interactions to include in the models  
twoTwoWays=[i for i in itertools.combinations(np.array(twoways),m)] # 
twoTwoWays
twoTwoWays[0] 
#make 2 new vars that are age*sex and age*cp do this automatically  

interactions_j=np.array(twoTwoWays[0])
heart[interactions_j[0]]

[i for i in interactions_j ]

#make function that will take as input the combo of 2 way interactions and return m columns 


for i in range(0,len(interactions_j)):
    df1=heart[interactions_j[i]]
    heart['int'+str(i)]=df1.iloc[0:len(df1),0]*df1.iloc[0:len(df1),1]
heart

def intCreate(ints):
    for i in range(0,len(ints)):
        df1=heart[ints[i]]
        if(i==0):
            dfInts=pd.DataFrame(df1.iloc[0:len(df1),0]*df1.iloc[0:len(df1),1])
            dfInts.columns=['-'.join([j for j in df1.columns])]
        else:
            dfHold=pd.DataFrame(df1.iloc[0:len(df1),0]*df1.iloc[0:len(df1),1])
            dfHold.columns=['-'.join([j for j in df1.columns])]
            dfInts=pd.concat([dfInts,dfHold],axis=1)
    return dfInts




#this is the interaction columns so append this to the data needed for the ith model 
intCreate(interactions_j)

dfBuild=heart[np.array(predictors[0])]
dfBuild=pd.concat([dfBuild,intCreate(interactions_j)],axis=1)
dfBuild.head()
y=np.array(heart['target'])
x=np.array(dfBuild)
model=sm.Logit(y, x)
result = model.fit(method='newton')
result.summary()
result.llf
dir(result)
#loop through each predictor combo as outer then loop through all combos of ints per predictor combo 
#make 2 lists to store models and to store R^2 adj 
models=list()
R2adjS=list() 
y=np.array(heart['target'])
del interactions_j 
xvars=np.array(['age','sex','cp','chol','oldpeak','thalach'],dtype=str)
#get all combinations of the x variables of size k 
k=4
predictors=[i for i in itertools.combinations(xvars, k)] 
for i in range(0,len(predictors)):
    twoways=[i for i in itertools.combinations(predictors[i], 2)]
    twoTwoWays=[i for i in itertools.combinations(np.array(twoways),m)]
    for j in range(0,len(twoTwoWays)):
        interactions_j=np.array(twoTwoWays[j])
        dfBuild=heart[np.array(predictors[i])]
        dfBuild=pd.concat([dfBuild,intCreate(interactions_j)],axis=1)
        x=np.array(dfBuild)
        model=sm.Logit(y, x)
        result = model.fit(method='newton')
        models.append(result.summary())
        R2adjS.append(result.llf)

max(R2adjS)

#make into a function: 
#inputs are the data as a pandas data frame, the poddible predictors, and weather or not you want logistic or linear reg 
del interactions_j 
del twoTwoWays 
del twoways
del predictors
def regIntCombos(df,y,predictors,m):
    models=list()
    R2adjS=list() 
    y=np.array(df[y])
    for i in range(0,len(predictors)):
        twoways=[i for i in itertools.combinations(predictors[i], 2)]
        mTwoWays=[i for i in itertools.combinations(np.array(twoways),m)]
        for j in range(0,len(mTwoWays)):
            interactions_j1=np.array(mTwoWays[j])
            dfBuild=heart[np.array(predictors[i])]
            dfBuild=pd.concat([dfBuild,intCreate(ints=interactions_j1)],axis=1)
            x=np.array(dfBuild)
            model=sm.Logit(y, x)
            result = model.fit(method='newton')
            models.append(result.summary())
            R2adjS.append(result.llf)
    return models 


xvars=np.array(['age','sex','cp','chol','oldpeak','thalach'],dtype=str)
#get all combinations of the x variables of size k 
k=4
preds1=[i for i in itertools.combinations(xvars, k)] 
regIntCombos(heart,'target',preds1,m=2)
        
       
    





