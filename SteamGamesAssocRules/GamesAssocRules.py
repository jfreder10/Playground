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
import plotly.graph_objects as go

steam=pd.read_csv(r'C:\Users\fredr\Documents\StatTool\PythonCode\FunctionsAndCoolStuff\SteamGamesAssocRules\steam-200k.csv',header=None)
#get the probability that a customer has bought 2 or more specfic games 
len(steam)
steam=steam.drop(4, axis=1)
steam.columns= ['custID', 'game', 'platVsOnlPur', 'HrPlay']
steam.head()
#make loop to iteratively do this 
#initilize the x df
#!!!!very important get the cust ids that have the ith game of interest then get the unique 
dfx=steam[steam['game'].isin([bethgames[0]])]
dfx=dfx['custID'].unique()
dfx=pd.DataFrame(dfx)
dfx.columns=['custID']
len(dfx)/len(steam['custID'].unique()) #same as R with the arules package for Skyrim and Fallout 4
#!!!
#2nd iteration 
#set dfx2 for the next iterative level
dfx2=steam[steam['game'].isin([bethgames[1]])]
dfx2=dfx2['custID'].unique()
dfx2=pd.DataFrame(dfx2)
dfx2.columns=['custID']
len(dfx2)/len(steam['custID'].unique())
#get the common custIDs among dfx and dfx2 and set these to dfx2
#then set dfx to the values in dffx that are in dfx2
dfx[dfx['custID'].isin(dfx2['custID'])] #107 rows
#put two loops below into a function that takes two lists as its arguments one for xlevels and ylevels
xlevels=['The Elder Scrolls V Skyrim','Fallout 4']
for i in range(0,len(xlevels)):
    if(i==0):
      dfx=steam[steam['game'].isin([xlevels[0]])]
      dfx=dfx['custID'].unique()
      dfx=pd.DataFrame(dfx)
      dfx.columns=['custID']
    else:
      dfx2=steam[steam['game'].isin([xlevels[i]])]
      dfx2=dfx2['custID'].unique()
      dfx2=pd.DataFrame(dfx2)
      dfx2.columns=['custID']
      dfx=dfx[dfx['custID'].isin(dfx2['custID'])]
len(dfx)
#if ylevels are more than 1 will have make a loop like above and do the same as the x vars
ylevels=['BioShock 2', 'Spore']
len(steam[steam['game'].isin(ylevels)])
for i in range(0,len(ylevels)):
    if(i==0):
      dfy=steam[steam['game'].isin([ylevels[0]])]
      dfy=dfy['custID'].unique()
      dfy=pd.DataFrame(dfy)
      dfy.columns=['custID']
    else:
      dfy2=steam[steam['game'].isin([ylevels[i]])]
      dfy2=dfy2['custID'].unique()
      dfy2=pd.DataFrame(dfy2)
      dfy2.columns=['custID']
      dfy=dfy[dfy['custID'].isin(dfy2['custID'])]
len(dfy)/len(steam['custID'].unique())
#get the ids in dfx that are also in dfy
#this is the cond prob that they have bought the games in y given we know they have bought the games in x 
len(dfx[dfx['custID'].isin(dfy['custID'])])/len(dfx)
#after make function go through all possible levels of x of a certain size and y of a certain size (nlevels choose k)
#gets the lift if the rule (the factor by which the prob of the y levels beiing purchased when we know that the x levels have been purchased)
(len(dfx[dfx['custID'].isin(dfy['custID'])])/len(dfx))/(len(dfy)/len(steam['custID'].unique()))
steam.head()
def associationRules(df,xlevels1,ylevels1):
  for i in range(0,len(xlevels1)):
    if(i==0):
      dfx1=steam[df['game'].isin([xlevels1[0]])]
      dfx1=dfx1['custID'].unique()
      dfx1=pd.DataFrame(dfx1)
      dfx1.columns=['custID']
    else:
      dfx21=df[df['game'].isin([xlevels1[i]])]
      dfx21=dfx21['custID'].unique()
      dfx21=pd.DataFrame(dfx21)
      dfx21.columns=['custID']
      dfx1=dfx1[dfx1['custID'].isin(dfx21['custID'])]
  for i in range(0,len(ylevels1)):
    if(i==0):
      dfy1=df[df['game'].isin([ylevels1[0]])]
      dfy1=dfy1['custID'].unique()
      dfy1=pd.DataFrame(dfy1)
      dfy1.columns=['custID']
    else:
      dfy21=df[df['game'].isin([ylevels1[i]])]
      dfy21=dfy21['custID'].unique()
      dfy21=pd.DataFrame(dfy21)
      dfy21.columns=['custID']
      dfy1=dfy1[dfy1['custID'].isin(dfy21['custID'])]
  priorprob=round(len(dfy1)/len(steam['custID'].unique()),4)
  if(len(dfx1)<1):
    condprob=0
  else:
    condprob=round(len(dfx1[dfx1['custID'].isin(dfy1['custID'])])/len(dfx1),4)
  
  lift=round(condprob/priorprob,4)
  stats=pd.DataFrame({'priorprobYLEVELS':[priorprob], 'condprobYGIVENXLEVELS': [condprob], 'lift':[lift]})
  return stats

steam['game'].unique()
xlevels=[['The Elder Scrolls V Skyrim','Fallout 4']]
xlevels[0]
ylevels=['Spore']
associationRules(steam,xlevels[0],ylevels)
len(steam[steam['game'].isin(['Spore','Fallout 4', 'BioShock 2'])])/len(steam[steam['game'].isin(['Spore','Fallout 4'])])

#sanity checks
#ex1 (cust IDs that have purchased the first game)
ex1=pd.DataFrame(steam['custID'][steam['game']=='Fallout 4'].unique())
ex1.columns=['custID']
#ex2 (cust IDs that have purchased the second game but are also present in the df of custs that have bought first game )
ex2=pd.DataFrame(steam['custID'][steam['game']=='Spore'].unique())
ex2.columns=['custID']
#ex3
ex3=pd.DataFrame(steam['custID'][steam['game']=='Fallout 4'].unique())
ex3.columns=['custID']
ex3
#get the numerator
exNum=ex1[ex1['custID'].isin(ex2['custID'])]
#check cond prob
len(exNum[exNum['custID'].isin(ex3['custID'])])/len(ex1[ex1['custID'].isin(ex2['custID'])])
#check prior prob
len(steam['custID'][steam['game']=='BioShock 2'].unique())/len(steam['custID'].unique())
#check lift
(len(exNum[exNum['custID'].isin(ex3['custID'])])/len(ex1[ex1['custID'].isin(ex2['custID'])]))/(len(steam['custID'][steam['game']=='BioShock 2'].unique())/len(steam['custID'].unique())) 


#now make a combo of the a nested list with several rules for game purchases in xlevels and ylevels
xlevelsMore=[['Spore','Fallout 4'],['Space Colony','Spore','Fallout 4'],['The Elder Scrolls V Skyrim','Life is Hard']]
xlevelsMore[0][0] #gets first element of the first nested list in the overall list 
steam['game'].unique()
ylevelsMore=[['Life is Hard'], ['The Elder Scrolls V Skyrim'],['Fallout 4']]
#now loop through all of these with the function and get the stats of all rule sets 
associationRules(steam,xlevelsMore[0],ylevelsMore[1]) #loop through these and store into a 
ass1
rules=pd.DataFrame({'priorprobYLEVELS':[0.0], 'condprobYGIVENXLEVELS':[0.0],'lift':[0.0]})
rules
pd.DataFrame(ass1)
#make an if else that will remove the game in the xlevels if it is in the jth ylevels 

#gets the position that are common among xlevels and ylevels

rm=np.where(pd.Series(xlevelsMore[0]).isin(pd.Series(ylevelsMore[0]))==True)

len(np.array(np.where(pd.Series(xlevelsMore[0]).isin(pd.Series(ylevelsMore[0]))==True)).tolist()[0])

len(np.where(pd.Series(xlevelsMore[0]).isin(pd.Series(ylevelsMore[0]))==True))
np.array(rm).tolist()[0]
#drops this position
list(pd.Series(xlevelsMore[0]).drop(np.array(rm).tolist()[0]))



#i think this works???
#make into function where inputs are the x and y levels the df and the var of the x and y levels and the var on how to track the count(userID custID basketID etc..)
for i in range(0,len(xlevelsMore)):
  for j in range(0,len(ylevelsMore)):
    if(len(np.array(np.where(pd.Series(xlevelsMore[i]).isin(pd.Series(ylevelsMore[j]))==True)).tolist()[0])>0):
      rm=np.where(pd.Series(xlevelsMore[i]).isin(pd.Series(ylevelsMore[j]))==True)
      rm=np.array(rm).tolist()[0]
      xS=list(pd.Series(xlevelsMore[i]).drop(np.array(rm).tolist()[0]))
      yS=ylevelsMore[j]
    else:
      xS=xlevelsMore[i]
      yS=ylevelsMore[j]
    
    ass1=associationRules(steam,xS,yS)
    print(ass1)

#make data table with go to display results better 
xlevels=[['The Elder Scrolls V Skyrim','Fallout 4']]
xlevels[0]
ylevels=['Spore']
df1=associationRules(steam,xlevels[0],ylevels)
df1
type(df1)
 fig = go.Figure(data=[go.Table(
        header=dict(values=list(df1.columns),
                fill_color='paleturquoise',
                align='left'),
        cells=dict(values=[df1.priorprobYLEVELS, df1.condprobYGIVENXLEVELS,df1.lift],
               fill_color='lavender',
               align='left'))
    ])

fig.show()