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



#################################################### make into a function where the inputs are: x (array from np), y (one var array from np), arg to choose if linear or logistic reg 
#need to put an intercept column in the df before this is done 

def LogRegKinteractions(df,yvar,kvars,n2WayInts):
    y=df[yvar]
    xvars=df.columns[~df.columns.isin(['intercept',yvar])]
    predictors=[i for i in itertools.combinations(xvars, kvars)] 
    allmods=list()
    for i in range(0,len(predictors)):
        preds_i=np.array(predictors[i])
        twoways=[i for i in itertools.combinations(predictors[i], 2)]
        twoTwoWays=[i for i in itertools.combinations(np.array(twoways),n2WayInts)]
        for j in range(0,len(twoTwoWays)):
            interactions_j=np.array(twoTwoWays[j])
            intcols=list()
            for m in range(0,len(interactions_j)):
                df[interactions_j[m][0]+'_'+'Interaction_'+interactions_j[m][1]]=df[interactions_j[0]].iloc[0:len(
                    df[interactions_j[m]]),0]*df[interactions_j[m]].iloc[0:len(df[interactions_j[m]]),1]
                intcols.append(interactions_j[m][0]+'_'+'Interaction_'+interactions_j[m][1])
                
            
            x=df[np.array(preds_i)]
            x[intcols]=df[intcols]
            #need to add the interaction vars and  
            try:
                model = sm.Logit(y, x)
                result = model.fit(method='newton')
                allmods.append(result)
                df=df.drop([interactions_j[m][0]+'_Interaction_'+interactions_j[m][1]],axis=1)
            except:
                df=df.drop([interactions_j[m][0]+'_Interaction_'+interactions_j[m][1]],axis=1)
                continue
    return allmods

heart=pd.read_csv(r'C:\Users\fredr\Documents\StatTool\PythonCode\FunctionsAndCoolStuff\DATA\heart.csv')

heart['intercept']=1
vars=['age','sex','cp','chol','oldpeak','thalach','target']

heart[vars]
outputTest=LogRegKinteractions(heart[vars],'target',kvars=4,n2WayInts=2)
heart.columns
outputTest[0].summary()
len(outputTest) #225 possible models but not all will be fit because singular matrix error 
comb(6,4) #number of possible models no interactions 
comb(4,2) #number of possible one 2 way interactions=6 per model if only wanted this then comb(6,1)
comb(6,2) #number of possible two 2 way interactions per model 
comb(6,4)*comb(6,2) # =225
comb(4,2) #the number of possible 2 way interactions 
comb(6,2) #the number of possible combos of two 2 way interactions in each model 
comb(6,4)*comb(comb(4,2),2)


########################## make another function but for linear reg 
cars=pd.read_csv(r'C:\Users\fredr\Documents\StatTool\PythonCode\FunctionsAndCoolStuff\DATA\MTCARS.csv')

def LinearRegKinteractions(df1,yvar,kvars,n2WayInts):
    df=df1
    y=df[yvar]
    xvars=df.columns[~df.columns.isin(['intercept',yvar])]
    predictors=[i for i in itertools.combinations(xvars, kvars)] 
    allmods=list()
    for i in range(0,len(predictors)):
        preds_i=np.array(predictors[i])
        twoways=[i for i in itertools.combinations(predictors[i], 2)]
        twoTwoWays=[i for i in itertools.combinations(np.array(twoways),n2WayInts)]
        for j in range(0,len(twoTwoWays)):
            interactions_j=np.array(twoTwoWays[j])
            intcols=list()
            for m in range(0,len(interactions_j)):
                df[interactions_j[m][0]+'_'+'Interaction_'+interactions_j[m][1]]=df[interactions_j[0]].iloc[0:len(
                    df[interactions_j[m]]),0]*df[interactions_j[m]].iloc[0:len(df[interactions_j[m]]),1]
                intcols.append(interactions_j[m][0]+'_'+'Interaction_'+interactions_j[m][1])
                
            
            x=df[np.array(preds_i)]
            x[intcols]=df[intcols]
            #need to add the interaction vars and  
            try:
                model = sm.OLS(y, x)
                result = model.fit()
                allmods.append(result)
                df=df.drop([intcols[0],intcols[1]],axis=1) 
            except:
                df=df.drop([intcols[0],intcols[1]],axis=1) 
                continue
    return allmods

cars['intercept']=1
cars.drop(['Unnamed: 0'],axis=1,inplace=True)
cars.columns #why is running the function changing the input object to the function 

outputTestLin=LinearRegKinteractions(cars,'mpg',kvars=4,n2WayInts=2)
len(outputTestLin)
outputTestLin[len(outputTestLin)-1].summary()






#############################scratch work used to build function above 

############# clean code up, walk through and make sure makes sense and add code for linear reg under the logistic reg 


# heart=pd.read_csv(r'C:\Users\fredr\Documents\StatTool\PythonCode\FunctionsAndCoolStuff\DATA\heart.csv')
# heart
# heart['intercept']=1
# heart.info() #target is neumeric 
# x=np.array(heart[['thalach','age']]).reshape(-1,3)
# x
# y=np.array(heart['target'])
# y
# #with stats mods

# x=sm.add_constant(x) #adds intercept to x matrix 
# x[0:len(x),1]
# model = sm.Logit(heart['target'], heart[['thalach','age']])
# result = model.fit(method='newton')
# result.params
# result.predict(x)
# min(result.predict(x))
# result.summary()
# dir(result)
# result.llf



#plot the output 0 or one on y axis an x as x axis then plot the predicted probs as a line through the plot 
# #plots the logistic reg preds as y to x 
# fig=px.scatter(x=x[0:len(x),1],y=result.predict(x))
# fig.show()
# np.mean(heart['age'][heart['target']==0])
# np.mean(heart['age'][heart['target']==1]) 
# #fit the log odds to x and perform model diagonstics 
# logodds=np.log(result.predict(x)/(1-result.predict(x)))
# logodds
# fig=px.scatter(x=x[0:len(x),1],y=logodds)
# fig.show()
# #get the satuarated model for each unique value of x 
# #get the prop of the values in target that ==1 for each unique value of x 
# heart['thalach'].unique()
# len(heart[heart['thalach']==heart['thalach'].unique()[0]][heart['target'][heart['thalach']==heart['thalach'].unique()[0]]==1])/len(heart[heart['thalach']==heart['thalach'].unique()[0]])
# saturated=np.zeros(len(heart['thalach'].unique()))
# saturated[0]
# for i in range(0,len(heart['thalach'].unique())):
#     saturated[i]=len(heart[heart['thalach']==heart['thalach'].unique()[i]][heart['target'][heart['thalach']==heart['thalach'].unique()[i]]==1])/len(heart[heart['thalach']==heart['thalach'].unique()[i]])

# saturated
# #plot the saturate probs given each value of x to the fitted probs 
# #fit mod on the unique values of x to compare to saturated model 
# xunique=np.array(heart['thalach'].unique())
# x=sm.add_constant(xunique)
# len(result.predict(x))
# len(saturated)
# df1=pd.DataFrame({'x':heart['thalach'].unique(),'fitted':result.predict(x),'model':'log'})
# df2=pd.DataFrame({'x':heart['thalach'].unique(),'fitted':saturated,'model':'saturated'})
# dfBoth=pd.concat([df1,df2])
# #not a good fit the models fitted values do not match up with the saturated probs given values of x 
# fig=px.scatter(x=dfBoth['x'],y=dfBoth['fitted'],color=dfBoth['model'])
# fig.show()


# #EX if 50 possible x vars but only want 10 in model then choose(50,10) arn number of possible unique variable combos possible, then if want 
# # combos of all 3 two way interactions among all possible predictors of 10 get the ith predictor subset and do choose(10,2) these would be the number of possible 
# #2 way interactions, then of this list select 3 at a time to fit in model with other 10 predictors of ith combo subset, 
# #so total number of vars would be: choose(50,10)*choose(len(choose(10,2)),3)
# #  


# #figure out how to select all possible 2 way interactions among variables given a vector of possible x vars
# #choose len(x), 2 1st possible 2 way interactions  
# #(len(x)-1, 2)-1 2nd possible interactions
# heart.columns
# xvars=np.array(['age','sex','cp','chol'],dtype=str)
# comb(4,2) #6 possible combos of a 2 way interaction for the first possible interaction 
# itertools.combinations(xvars, 2)
# #gets all possible 2 way interactions for the first interaction variable 
# twoways=[i for i in itertools.combinations(xvars, 2)]
# #fit a model with all x vars predicting target and each possible pair of the 2 way interactions (2 2way interactions) choose 6,2 
# #gets all possible combos of 2 two way interaction vars for a model 
# twoTwoWays=[i for i in itertools.combinations(np.array(twoways),2)]
# len(twoTwoWays)
# heart[twoTwoWays[0][0]] #gets the columns for the first interaction in the first combo of all possible 2 interactions 

# #loop through combos of all possible x vars with size equal to k as outer loop 
# xvars=np.array(['age','sex','cp','chol','oldpeak','thalach'],dtype=str)
# predictors=[i for i in itertools.combinations(xvars, 4)]
# len(predictors)
# preds_i=np.array(predictors[0])
# preds_i=np.append(preds_i,'intercept')
# x=np.array(heart[np.array(preds_i)])
# y=np.array(heart['target'])

# model = sm.Logit(y, x)
# result = model.fit(method='newton') #dtore this as the ith element in kmods list 
# result.params
# result.predict(x)

# kmods=list()
# for i in range(0,len(predictors)):
#     preds_i=np.array(predictors[i])
#     preds_i=np.append(preds_i,'intercept')
#     x=np.array(heart[np.array(preds_i)])
#     model = sm.Logit(y, x)
#     result = model.fit(method='newton')
#     kmods.append(result)

# len(kmods)
# kmods[0].summary()


#for each ith possible var combo loop through the j interactions the size k out of the number of selected vars for the ith combo 
#then need to loop through the interactions selected to create the interaction vars 
# df=heart 
# xvars=df.columns
# predictors=[i for i in itertools.combinations(xvars, 4)]
# predictors[0] 
# #in outter loop right before inner 
# twoways=[i for i in itertools.combinations(predictors[0], 2)]
 
# twoTwoWays=[i for i in itertools.combinations(np.array(twoways),2)] # set this to k
# interactions_j=np.array(twoTwoWays[0])


# heart[interactions_j[0]] #get product of these 2 cols as the names pasted together 
#  #this initilizes the interaction column
# intcols=list()
# for m in range(0,len(interactions_j)):
#     heart[interactions_j[m][0]+'_'+interactions_j[m][1]]=heart[interactions_j[0]].iloc[0:len(heart[interactions_j[m]]),0]*heart[interactions_j[m]].iloc[0:len(heart[interactions_j[m]]),1]
#     intcols.append(pd.DataFrame(heart[interactions_j[m][0]+'_'+interactions_j[m][1]]).columns)
# heart
# np.array(intcols)

# preds_i=np.array(predictors[0])
# predictors
# preds_i=np.append(preds_i,'intercept')
# np.append(preds_i,np.array(intcols)) #put 
# x=np.array(heart[np.array(preds_i)])
# ith_interaction_mod=list()



# ################################################## make into a function with the df,yvar, number of xvar out of all, and the number of 2 way interactions 
# df=heart 
# xvars=df.columns
# predictors=[i for i in itertools.combinations(xvars, 4)]
 
# #in outter loop right before inner 
# twoways=[i for i in itertools.combinations(predictors[0], 2)]
 
# twoTwoWays=[i for i in itertools.combinations(np.array(twoways),2)]


# for j in range(0,len(twoTwoWays)):
#     interactions_j=np.array(twoTwoWays[j])
#     intcols=list()
#     for m in range(0,len(interactions_j)):
#         heart[interactions_j[m][0]+'_'+interactions_j[m][1]]=heart[interactions_j[0]].iloc[0:len(heart[interactions_j[m]]),0]*heart[interactions_j[m]].iloc[0:len(heart[interactions_j[m]]),1]
#         intcols.append(pd.DataFrame(heart[interactions_j[m][0]+'_'+interactions_j[m][1]]).columns)
        
#     np.append(preds_i,np.array(intcols))
#     x=np.array(heart[np.array(preds_i)])
#     model = sm.Logit(y, x)
#     heart=heart.drop([interactions_j[m][0]+'_'+interactions_j[m][1]],axis=1) 
#     result = model.fit(method='newton')
#     ith_interaction_mod.append(result)
# heart
# len(ith_interaction_mod) #15 because we only went through one possible combo 

# #wrap code in outer loop for selecting the ith subset of predictors size = k 
# #gets all models for a subset of xvars = k and for all possible 2 two way interactions 
# xvars
# xvars=np.array(['age','sex','cp','chol','oldpeak','thalach'],dtype=str)
# predictors=[i for i in itertools.combinations(xvars, 4)] #loop through as outer loop 
# predictors
# xvars
# twoways=[i for i in itertools.combinations(predictors[0], 2)] #set this to get the number of m 2 way interaction combos 
# twoways
# twoTwoWays=[i for i in itertools.combinations(np.array(twoways),2)] #loop through in inner with the predictors set to i for all j 
# twoTwoWays

# #innner loop of the inner loop creates vars so can append to current x vars from outer loop 
# ############################################# function 



# xvars=np.array(['age','sex','cp','chol','oldpeak','thalach'],dtype=str)
# predictors=[i for i in itertools.combinations(xvars, 4)] #loop through as outer loop 
# predictors
# xvars
# twoways=[i for i in itertools.combinations(predictors[0], 2)] #set this to get the number of m 2 way interaction combos 
# twoways
# twoTwoWays=[i for i in itertools.combinations(np.array(twoways),2)] #loop through in inner with the predictors set to i for all j 
# twoTwoWays
# allmods=list()
# y=heart['target']
# for i in range(0,len(predictors)):
#     preds_i=np.array(predictors[i])
#     twoways=[i for i in itertools.combinations(predictors[i], 2)]
#     twoTwoWays=[i for i in itertools.combinations(np.array(twoways),2)]
#     for j in range(0,len(twoTwoWays)):
#         interactions_j=np.array(twoTwoWays[j])
#         intcols=list()
#         for m in range(0,len(interactions_j)):
#             heart[interactions_j[m][0]+'_Interaction_'+interactions_j[m][1]]=heart[interactions_j[0]].iloc[0:len(
#                 heart[interactions_j[m]]),0]*heart[interactions_j[m]].iloc[0:len(heart[interactions_j[m]]),1]

#             intcols.append(pd.DataFrame(heart[interactions_j[m][0]+'_Interaction_'+interactions_j[m][1]]).columns)
            
#         x=heart[np.append(preds_i,np.array(intcols))]
        
#         model = sm.Logit(heart['target'],heart[np.append(preds_i,np.array(intcols))])
#         #heart=heart.drop([interactions_j[m][0]+'_Interaction_'+interactions_j[m][1]],axis=1) 
#         try: 
#             result = model.fit(method='newton') 
#             result.summary()
#             allmods.append(result)
#         except:
#             continue


# x.head()
# np.unique(y)
# heart.columns
# len(allmods) #225 possible models but not all will be fit because singular matrix error 
# comb(6,4) #number of possible models no interactions 
# comb(4,2) #number of possible one 2 way interactions=6 per model if only wanted this then comb(6,1)
# comb(6,2) #number of possible two 2 way interactions per model 
# comb(6,4)*comb(6,2) # =225
# comb(4,2) #the number of possible 2 way interactions 
# comb(6,2) #the number of possible combos of two 2 way interactions in each model 
# comb(6,4)*comb(comb(4,2),2)

# allmods[5].summary()
