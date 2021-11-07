import sklearn
import pandas as pd 
import matplotlib.pyplot as plt
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

def expandgrid(*itrs):
   product = list(itertools.product(*itrs))
   return {'Var{}'.format(i+1):[x[i] for x in product] for i in range(len(itrs))}

#import the data (heart disease default):

heart=pd.read_csv(r"C:\Users\fredr\Documents\StatTool\PythonCode\heart.csv")  


#data engineering, get possible vars from the data mean num vars per level of x vars, etc.. 
heart.describe()
len(heart.columns)
heart.columns
#find linear relationships among vars create a k by k pd df filled with 0s and loop through ith row and jth col to get the cor between 
#independent var i and independent var j (diagnol of matrix will be 0)
indvars=['age','sex','cp','trestbps','chol']
k=len(indvars)
xvarcors=pd.DataFrame(np.zeros((k,k)))
#xvarcors.iloc[0,0]=1 sets an ith of jth value in a df to some value 
#xvarcors.iloc[0,0]
#get the cor between var 1 and var 1 (so will be 1)
heart[indvars[0]]
regss=regMods.OLS(endog=heart[indvars[0]], exog=heart[indvars[1]], missing='drop').fit()
regss.rsquared_adj
#make for loop that fills the i j values of the xcors matrix 
for i in range(0,len(indvars)):
    for j in range(0,len(indvars)):
        regss1=regMods.OLS(endog=heart[indvars[i]], exog=heart[indvars[j]], missing='drop').fit()
        xvarcors.iloc[i,j]=regss1.rsquared_adj
xvarcors
#do something with the vars that are more related with each other than the average correlation 
xvarcors.describe()
type(xvarcors.describe())
st.mean(xvarcors.describe().iloc[1]) #this is the mean of all jth cols means 
#make a list that stores the i j indeices if the i j cor >=.7 and <1 (becuse 1 is cor for itself)
len(xvarcors.columns)
intvars=list()
#join the sorted objects so can be done in the loop 


#loop getting the combos of vars that are sig to make interactions with 
for i in range(0,len(xvarcors)):
    for j in range(0,len(xvarcors.columns)):
        if(xvarcors.iloc[i,j]>=.70 and xvarcors.iloc[i,j]<1):
            #put the sort/join command around this some how 
            intvars.append(indvars[i]+"_"+indvars[j])
intvars
#sort this then re join and set to the finalInt vars to use 
np.unique(intvars)[0].split('_') 
'_'.join(sorted(np.unique(intvars)[0].split('_')))
sortintvars=list()
for i in range(0,len(intvars)):
    sortintvars.append('_'.join(sorted(np.unique(intvars)[i].split('_'))))
#combine the two loops above into a function 
#can use this to remove interaction combos that are redundant 
sortintvars
#remove the repeat values, set it to the unique values 
np.unique(np.array(sortintvars))
#resplit the strings of the list above
#make interactions with some function for each of the ith element of the code below 
finalintvars=np.unique(np.array(sortintvars))[0].split('_') #loop through this 
#split these again and then perfor interaction computatins on them, for each possible ith interaction on the jth combo get the R^2 (or if 
# class as response, do something else) between the interaction and the response var
"_".join(finalintvars)+'_interaction'

for i in range(0,len(np.unique(np.array(sortintvars)))):
    finalintvars=np.unique(np.array(sortintvars))[i].split('_')
    heart["_".join(finalintvars)+'_interaction']=heart[finalintvars].iloc[0:,0]*heart[finalintvars].iloc[0:,1]
    
heart.head()







#split the train and test data 
train, test = train_test_split(heart[['age','sex','cp','trestbps','chol']], test_size=0.2)
y_train=train['chol']
x_train=train[['age','sex','cp','trestbps']]



#boosting to predict chol 
#set the range for the parameter values:
n_estimators=np.arange(100, 350, 50) #the number of trees to fit 
max_depth=np.arange(1, 3, 1)
min_samples_split=np.arange(3,4,1)
learning_rate=np.arange(0.002,0.005,0.001)

a=expandgrid(n_estimators,max_depth, min_samples_split,learning_rate)
params=pd.DataFrame.from_dict(a)
len(params)
#name the parameters
val_rmses=list(params.index)
len(val_rmses)
#time the code
#looping through the possible parameters for the model and store the estimated validation rmse
for i in range(0,len(params)):
    scores = cross_val_score(HistGradientBoostingRegressor(min_samples_leaf=params['Var3'].iloc[i],
    max_depth=params['Var2'].iloc[i],
    learning_rate=params['Var4'].iloc[i],max_iter=params['Var1'].iloc[i]).fit(X_train, y_train), 
    X_train, y_train, cv=4,scoring='neg_mean_squared_error')
    rmse=st.mean((scores*-1) ** (1/2))
    val_rmses[i]=rmse

#gets the row (param settings) that has the lowest val error 
min(val_rmses) 
list(params.iloc[val_rmses==min(val_rmses)].iloc[0])
pars=list(params.iloc[val_rmses==min(val_rmses)].iloc[0])
pars.append(min(val_rmses))
pars

tpprmrn.loc[len(tpprmrn)]=pars
tpprmrn


#make a df that stores the params and rmse for that param (the best param for the ith run and the rmse) so the ncols=1+the params
#make a df with ncols= to the ncols f params+1 (1 extra for the rmse col)
#make another df whwer the range of the params considered for each ith iteration 

#only run this once 
parnames=['n_estimators','max_depth','min_samples_split','learning_rate','rmse']
np.arange(len(parnames))
tpprmrn=pd.DataFrame(index=np.arange(1), columns=np.arange(len(parnames)))
tpprmrn.iloc[0]=pars

#iteration1:53.495
#iteration2:
#make a df that stores the params and rmse for that param (the best param for the ith run and the rmse) so the ncols=1+the params


#classifer to predict if ith person has heart disease 
heart.columns
train, test = train_test_split(heart[['age','sex','cp','trestbps','chol','target','age_chol_interaction','age_trestbps_interaction',
'chol_trestbps_interaction']], test_size=0.2)
y_train=train['target']
X_train=train[['age','sex','cp','trestbps','chol','age_chol_interaction','age_trestbps_interaction',
'chol_trestbps_interaction']]
y_test=test['target']
X_test=test[['age','sex','cp','trestbps','chol','age_chol_interaction','age_trestbps_interaction',
'chol_trestbps_interaction']]



n_estimators=np.arange(900, 950, 50) #the number of trees to fit 
max_depth=np.arange(3, 4, 1)
min_samples_split=np.arange(3,4,1)
learning_rate=np.arange(0.0003, 0.0004, 0.0001)

a=expandgrid(n_estimators,max_depth, min_samples_split,learning_rate)
params=pd.DataFrame.from_dict(a)
val_acc=list(params.index)
len(params)
mods=list(params.index)
for i in range(0,len(params)):
    scores = cross_val_score(HistGradientBoostingClassifier(min_samples_leaf=params['Var3'].iloc[i],
    max_depth=params['Var2'].iloc[i],
    learning_rate=params['Var4'].iloc[i],max_iter=params['Var1'].iloc[i]).fit(X_train, y_train), 
    X_train, y_train, cv=4,scoring='accuracy')
    accuracy=st.mean(scores)
    val_acc[i]=accuracy
    

max(val_acc)
len(params)
len(val_acc)
pars=list(params.iloc[val_acc==max(val_acc)].iloc[0])
pars
pars.append(max(val_acc))
tpprmrn.loc[len(tpprmrn)]=pars
tpprmrn

#figure out how to predict on the test data, use the best model then .fit(Xtest,ytest)
np.where(val_acc==max(val_acc)) #gets the index value for the row in params with the best accuarcy on from train 
kmod=1
bestFit=HistGradientBoostingClassifier(min_samples_leaf=params['Var3'].iloc[kmod],
    max_depth=params['Var2'].iloc[kmod],
    learning_rate=params['Var4'].iloc[kmod],max_iter=params['Var1'].iloc[kmod]).fit(X_test,y_test)
bestFit.predict(X_test)
bestFit.score(X_test,y_test) #accuarcy on test data is 85% 








#only run one time 
# pars=list(params.iloc[val_acc==max(val_acc)].iloc[0])
# pars.append(min(val_acc))
# parnames=['n_estimators','max_depth','min_samples_split','learning_rate','accuracy']
# np.arange(len(parnames))
# tpprmrn=pd.DataFrame(index=np.arange(1), columns=np.arange(len(parnames)))
# tpprmrn.iloc[0]=pars


#svm



#adaBoost




#rf MajorietyClass



#regLogReg



#

