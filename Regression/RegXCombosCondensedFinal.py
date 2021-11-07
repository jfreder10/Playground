from math import comb
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


gdb=pd.read_csv(r'C:\Users\fredr\Documents\StatTool\BZAN540\Homework\HW1\GoodBellyData.csv')
gdb=gdb[gdb.columns[0:12]]
gdb.head()
gdb.columns
gdb['intercept']=1
type(gdb)
gdb[gdb.columns[[0,7]]].iloc[0:] #random but useful subsetting 
gdb.columns

xvars=['Average Retail Price','Demo', 'Store','Demo1-3', 'Demo4-5','Sales Rep', 'Endcap']

finalvars=list(itertools.combinations(xvars, 3))
#convert finalvars
type(finalvars[0])
list(finalvars[0])
finalvars[0]=list(finalvars[0])
type(finalvars[0])
for i in range(0,len(finalvars)):
    finalvars[i]=list(finalvars[i])
finalvars
#add intercept to each 
for i in range(0, len(finalvars)):
    finalvars[i].append('intercept')
finalvars
len(finalvars)

rsquareds=list(range(0,len(finalvars)))
allregs=list()
for i in range(0,len(finalvars)):
    if(np.array(gdb[finalvars[i]].dtypes=='O').sum()>=1):
        a=pd.DataFrame(gdb[list(np.array(finalvars[i])[np.array(list(gdb[finalvars[i]].dtypes!='O'))])])
        b=pd.DataFrame(pd.get_dummies(gdb[list(np.array(finalvars[i])[np.array(list(gdb[finalvars[i]].dtypes=='O'))])]))
        est=sm.OLS(endog=gdb['Units Sold'], exog=pd.concat([a,b], axis=1), missing='drop').fit()
        allregs.append(est.summary())
        rsquareds[i]=est.rsquared_adj
    else:
        est=sm.OLS(endog=gdb['Units Sold'], exog=gdb[finalvars[i]], missing='drop').fit()
        allregs.append(est.summary())
        rsquareds[i]=est.rsquared_adj
rsquareds=np.array(rsquareds)
len(rsquareds)
max(rsquareds)
np.where(rsquareds==max(rsquareds))
allregs[22]

#make this into a function then can loop through many possible k (number of vars to use out of the total n predictor vars ava)
#the data, the x vars as a list, the response var, and the k number of vars to use out of all xvars will all be inputs 


#df, yvar, xvars, and the k choices out of combos will be the inputs 
#make sure you make a variable intercept in df that is 1 for each row 
del allregs
def allRegCombos(df,yvar,xvars,k,metric="all"):
    finalvars=list(itertools.combinations(xvars, k))
    for i in range(0,len(finalvars)):
        finalvars[i]=list(finalvars[i])
    for i in range(0, len(finalvars)):
        finalvars[i].append('intercept')
        finalvars
    rsquareds=list(range(0,len(finalvars)))
    allregs=list(range(0,len(finalvars)))
    for i in range(0,len(finalvars)):
        if(np.array(df[finalvars[i]].dtypes=='O').sum()>=1):
            a=pd.DataFrame(df[list(np.array(finalvars[i])[np.array(list(df[finalvars[i]].dtypes!='O'))])])
            b=pd.DataFrame(pd.get_dummies(df[list(np.array(finalvars[i])[np.array(list(df[finalvars[i]].dtypes=='O'))])]))
            est=sm.OLS(endog=df[yvar], exog=pd.concat([a,b], axis=1), missing='drop').fit()
            allregs[i]=est.summary()
            rsquareds[i]=est.rsquared_adj
        else:
            est=sm.OLS(endog=df[yvar], exog=df[finalvars[i]], missing='drop').fit()
            allregs[i]=est.summary()
            rsquareds[i]=est.rsquared_adj
    if(metric=='rsquareds'):
        return rsquareds
    else:
        return allregs
            


x=['Average Retail Price','Demo', 'Store','Demo1-3', 'Demo4-5','Sales Rep', 'Endcap'] 
len(x)
y='Units Sold'
regs=allRegCombos(gdb,y,x,3,metric='all')
len(regs) 
regs[0] 
comb(7,3)





#redo this to be a lot simpler: ex below is for logistic reg so generalize it to linear reg 
xvars=np.array(['age','sex','cp','chol'],dtype=str)
predictors=[i for i in itertools.combinations(xvars, 2)]
len(predictors)
preds_i=np.array(predictors[0])
preds_i=np.append(preds_i,'intercept')
x=np.array(heart[np.array(preds_i)])
y=np.array(heart['target'])

model = sm.Logit(y, x)
result = model.fit(method='newton') #dtore this as the ith element in kmods list 
result.params
result.predict(x)

kmods=list()
for i in range(0,len(predictors)):
    preds_i=np.array(predictors[i])
    preds_i=np.append(preds_i,'intercept')
    x=np.array(heart[np.array(preds_i)])
    model = sm.Logit(y, x)
    result = model.fit(method='newton')
    kmods.append(result)
