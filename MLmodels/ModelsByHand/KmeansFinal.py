import pandas as pd 
import numpy as np 
import statistics as st 

heart=pd.read_csv(r'C:\Users\fredr\Documents\StatTool\PythonCode\FunctionsAndCoolStuff\DATA\heart.csv')

#put all together 
#STEP 1: initilize and get total wcss of random centers 
clust1=heart.iloc[90]
clust2=heart.iloc[45]
#repeat the number of clusters so that you can select however many you want 

#store in numpy in future instead of lists 

clusters=[clust1,clust2]
totalwcss=list()
wcss=list()
#STEP 2: Assign membership to the random clusters 
member=list()
distObs=list(range(0,len(clusters)))
for i in range(0,len(heart)):
    for j in range(0,len(clusters)):
        distObs[j]=sum(((heart.iloc[i]-clusters[j]) **2))**.5
    member.append(np.array(np.where(np.array(distObs)==min(distObs))).tolist()[0][0])
member
#STEP 3 get wcss of the random clusters 
totalwcss=list()
for i in range(0,len(clusters)):
    wcss.append(sum(pd.DataFrame((heart.iloc[np.array(np.where(np.array(member)==i)).tolist()[0]]-clusters[i])**2).sum(axis=1)))
totalwcss.append(sum(wcss))
totalwcss

#STEP 1.2 update clusters from random cluster assignment  
for i in range(len(clusters)):
    clusters[i]=heart.iloc[np.where(np.array(member)==i)].describe().iloc[1]

#STEP 2.2 from updated clusters reassign membership of all observations 
member=list()
for i in range(0,len(heart)):
    for j in range(0,len(clusters)):
        distObs[j]=sum(((heart.iloc[i]-clusters[j]) **2))**.5
    member.append(np.array(np.where(np.array(distObs)==min(distObs))).tolist()[0][0])
#STEP 2.3 get wcss from updated cluster assignment 
wcss=list()
for i in range(0,len(clusters)):
    wcss.append(sum(pd.DataFrame((heart.iloc[np.array(np.where(np.array(member)==i)).tolist()[0]]-clusters[i])**2).sum(axis=1)))
totalwcss.append(sum(wcss))
totalwcss
#now put into while loop and repeat until wcss no longer decreases from the previous iteration to 2 previous iterations 
k=2
while(totalwcss[k-1]<totalwcss[k-2]):
    #1. update cluster centers 
    for i in range(len(clusters)):
        clusters[i]=heart.iloc[np.where(np.array(member)==i)].describe().iloc[1]
    #2. assign membership based on updated clusters 
    member=list()
    for i in range(0,len(heart)):
        for j in range(0,len(clusters)):
            distObs[j]=sum(((heart.iloc[i]-clusters[j]) **2))**.5
        member.append(np.array(np.where(np.array(distObs)==min(distObs))).tolist()[0][0])
    #3. get wcss for updated clusters 
    wcss=list()
    for i in range(0,len(clusters)):
        wcss.append(sum(pd.DataFrame((heart.iloc[np.array(np.where(np.array(member)==i)).tolist()[0]]-clusters[i])**2).sum(axis=1)))
    totalwcss.append(sum(wcss))
    k=k+1
len(totalwcss) #11 iterations 


#define a function to do the above 