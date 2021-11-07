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

dfex=pd.read_csv(r'C:\Users\fredr\Documents\StatTool\BZAN542\Homework\HW4DataQ1.csv')
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

probBays(dfex,'Class','Movie Format')
probBays(dfex,'Class','Movie Category')
pC0=len(dfex[dfex['Class']=='C0'])/len(dfex)
pC1=len(dfex[dfex['Class']=='C1'])/len(dfex)

#bays for C0|Format=='DVD' and Cat=='Com'
pC0*.6*.7
#bays for C1|Format=='DVD' and Cat=='Com'
pC1*.2*.1
#classified as C0

#second function (needs first one to work) to classify any amount of levels of a respense with bays for any amount of x vars but inputing the levels of the x vars as inputs
def bays_part2(df,xvars,yvar,xlevels):
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
#test on dfex from class as sanity check, it seems to work out 
dfex=pd.read_csv(r'C:\Users\fredr\Documents\StatTool\BZAN542\Homework\HW4DataQ1.csv')
dfex
bays_part2(dfex,['Movie Format','Movie Category'], 'Class', ['DVD','Com'])
bays_part2(dfex,['Movie Format','Movie Category'], 'Class', ['DVD','Ent'])

#loop through all combos with exp grid then for each combo make choose the level of y with th highest bays value for those x levels amonk the vars


