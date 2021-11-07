import pandas as pd 
import numpy as np 
import plotly.express as px
import statsmodels as stm 
import statsmodels.api as sm 
import statistics as st
from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import expon
from scipy.stats import poisson
from scipy.stats import binom
import itertools

net=pd.read_csv(r'C:\Users\fredr\Documents\StatTool\PythonCode\FunctionsAndCoolStuff\netflix_titles.csv')
net.head()
netshows=net[net['type']=='TV Show']
netshows.head()
netshows.columns
#get the prob of duration per rating and country then use bays to classify a duration per spec values of rating and country 
ratingvalues=netshows['rating'].unique()
seasons=netshows['duration'].unique()
netshows['duration'].describe()
len(seasons)
len(netshows[netshows['duration']==seasons[0]])
shows=list(range(0,len(seasons)))
for i in range(0,len(seasons)):
    shows[i]=len(netshows[netshows['duration']==seasons[i]])
shows

#loop through these unique values and get the prob of season size where season is the outer loop 
cond1=netshows['rating']==ratingvalues[0]
cond2=netshows['duration']==seasons[0]
netshows[cond1 & cond2]
#divide by the ones that == the ith rating value (for bays therom)
len(netshows[cond1 & cond2])/len(netshows[cond2])
pd.DataFrame(index=range(0,len(netshows['duration'].unique())), columns=range(0,len(netshows['rating'].unique())))
probs=pd.DataFrame(index=range(0,len(netshows['duration'].unique())), columns=range(0,len(netshows['rating'].unique())))
probs=pd.DataFrame(probs, columns=ratingvalues)
probs.head()
probs=probs.fillna(0)

for i in range(0,len(seasons)):
    for j in range(0,len(ratingvalues)):
        cond1=netshows['rating']==ratingvalues[j]
        cond2=netshows['duration']==seasons[i]
        a=len(netshows[cond1 & cond2])
        b=len(netshows[cond2])
        probs.iloc[i,j]=(a/b)
probs
len(ratingvalues)
len(seasons)
len(probs)
probs
#sanity check with code below, all so far have checked out make loop to go through all and make sure they all == each other 
cond1a=netshows['rating']==ratingvalues[0]
cond2a=netshows['duration']==seasons[0]
netshows[cond1a & cond2a]
len(netshows[cond1a & cond2a])/len(netshows[cond2a])
probs[4]
#to get bays for this loop through each col as outter loop and each row as inner if want all, if only need oned level of x then just choose that var and 
#loop through those rows 
classMA=list(range(0,len(probs)))
probs['TV-MA'][0]
#make into a function then can loop into a list for however many independent vars to predict response with bays 
def probBays(df,yvar, xvar):
    xlevels=df[xvar].unique()
    ylevels=df[yvar].unique()
    probs=pd.DataFrame(index=range(0,len(df[yvar].unique())), columns=range(0,len(df[xvar].unique())))
    probs=pd.DataFrame(probs, columns=xlevels)
    probs=probs.fillna(0)
    for i in range(0,len(ylevels)):
        for j in range(0,len(xlevels)):
            cond1=df[xvar]==xlevels[j]
            cond2=df[yvar]==ylevels[i]
            if(len(df[cond1 & cond2])==0):
                probs.iloc[i,j]=0
            else:
                a=len(df[cond1 & cond2])
                b=len(df[cond2])
                probs.iloc[i,j]=(a/b)
    return probs
prob1=probBays(netshows,'duration','rating')
prob1
#anity checks
cond1a=netshows['rating']==ratingvalues[2]
cond2a=netshows['duration']==seasons[2]
netshows[cond1a & cond2a]
len(netshows[cond1a & cond2a])/len(netshows[cond2a])

#now write loop to calculate bays for any rating to predict the duration 
len(netshows[netshows['duration']==netshows['duration'].unique()[0]])/len(netshows)*prob1['TV-MA'].iloc[0]
baysma=list(range(0,len(netshows['duration'].unique())))
for i in range(0,len(netshows['duration'].unique())):
    baysma[i]=len(netshows[netshows['duration']==netshows['duration'].unique()[i]])/len(netshows)*prob1['TV-MA'].iloc[i]
max(baysma) #second unique level
dfex=pd.read_csv(r'C:\Users\fredr\Documents\StatTool\BZAN542\Homework\HW4DataQ1.csv')
dfex
probBays(dfex,'Class','Movie Format')
probBays(dfex,'Class','Movie Category')
pC0=len(dfex[dfex['Class']=='C0'])/len(dfex)
pC1=len(dfex[dfex['Class']=='C1'])/len(dfex)
#bays for C0|Format=='DVD' and Cat=='Com'
pC0*.6*.7
#bays for C1|Format=='DVD' and Cat=='Com'
pC1*.2*.1
#classified as C0
#make a function that will take the x levels to condition on and output is the bays number for each class 
#could loop through all possible vars conditioning on as outter loop then loop through the levels of these as the inner 
#make expand grid to loop through the appended data (cbind) and get the ith levels that == the cols in the appended df then can multiply those values together 
#make a expand grid of all the probs per each class (so 2 grids if have 2 response levels) then could multiply the values in a row together, each row will represent the ith 
#combination of the levels of the k independent vars ex grid should work??? make k other cols to keep track of the names of the levels per each combo 
def expandgrid(*itrs):
   product = list(itertools.product(*itrs))
   return {'Var{}'.format(i+1):[x[i] for x in product] for i in range(len(itrs))}

a1=probBays(dfex,'Class','Movie Format')
b1=probBays(dfex,'Class','Movie Category')
a1.iloc[0,]
a1
b1
pd.DataFrame(expandgrid(a1.iloc[0,], b1.iloc[0,]))
pd.DataFrame(expandgrid(dfex['Movie Format'].unique(), dfex['Movie Category'].unique()))
pd.concat([pd.DataFrame(expandgrid(a1.iloc[0,], b1.iloc[0,])),pd.DataFrame(expandgrid(dfex['Movie Format'].unique(), dfex['Movie Category'].unique()))], axis=1)
#generalize this to work/ automated for many independent vars with many classes 
#could just loop through the expand grid of all combos of the levels and then get the probs in one loop 
#best way is to loop through the function for the number of x vars and store each level of interest per xvar  as a value in a list 
#use a df with more than 2 possible x vars as ex, store x vars in a list then loop through these to store the prob of interest for that level for each predicted class
#in a list or a df 

#loop through all possible x vars in netshows and use function for all print out df 
netshows.columns
xvars=['country','rating','release_year']
yvar='duration'
xlevels=['United States','TV-MA',2016]
a1=probBays(netshows,yvar,xvars[0])
#make a df with the num cols == the num of x vaars and the num rows == to the num unique duration values then store in each col the code below for the level of interest
#for the ith x var
cond_probs=pd.DataFrame(index=range(0,len(netshows[yvar].unique())), columns=range(0,len(xvars)))
cond_probs=pd.DataFrame(cond_probs, columns=xlevels)
cond_probs.head()
cond_probs=cond_probs.fillna(0)
a1.loc[0:len(a1),a1.columns[a1.columns==xlevels[0]]]
for i in range(0,len(xvars)):
    a1=probBays(netshows,yvar,xvars[i])
    cond_probs.iloc[0:len(cond_probs),i]=a1[xlevels[i]]
cond_probs
#gets prior prob for each level of the response 
list(netshows['duration'].value_counts()/len(netshows))
cond_probs['prior _prob']=list(netshows['duration'].value_counts()/len(netshows))
cond_probs
#now multiply across each row 
cond_probs.iloc[0].prod()
#make another col that is bays value and loop through all rows with code above 
np.repeat(.001,len(cond_probs))
cond_probs['bays_number']=np.repeat(.000,9)
for i in range(0,len(cond_probs)):
    cond_probs['bays_number'][i]=cond_probs.iloc[i].prod()
cond_probs
#now make all into function 
def bays_part2(df, xvars,yvar,xlevels):
    cond_probs=pd.DataFrame(index=range(0,len(df[yvar].unique())), columns=range(0,len(xvars)))
    cond_probs=pd.DataFrame(cond_probs, columns=xlevels)
    cond_probs=cond_probs.fillna(0)
    cond_probs['prior _prob']=list(df[yvar].value_counts()/len(df))
    for i in range(0,len(xvars)):
        a1=probBays(df,yvar,xvars[i])
        cond_probs.iloc[0:len(cond_probs),i]=a1[xlevels[i]]
    cond_probs['bays_number']=np.repeat(.000,len(cond_probs))
    for i in range(0,len(cond_probs)):
        cond_probs['bays_number'][i]=cond_probs.iloc[i,0:len(cond_probs.columns)-1].prod()
    return cond_probs

bays_part2(netshows,['country','rating','release_year'],'duration',['United States','TV-MA',2016])
#test on dfex from class 
dfex=pd.read_csv(r'C:\Users\fredr\Documents\StatTool\BZAN542\Homework\HW4DataQ1.csv')
dfex
bays_part2(dfex,['Movie Format','Movie Category'], 'Class', ['DVD','Com'])
bays_part2(dfex,['Movie Format','Movie Category'], 'Class', ['DVD','Ent'])





