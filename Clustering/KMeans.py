import pandas as pd 
import numpy as np 
import statistics as st 

#import a df with only numeric cols 
heart=pd.read_csv(r'C:\Users\fredr\Documents\StatTool\PythonCode\FunctionsAndCoolStuff\DATA\heart.csv')
heart.head() #all numeric 
#randomly assign 2 clusters then get the distance between the 1st row and each cluster then assign  it to the cluster it is most similar to this will be inner loop 
#of for loop 
clust1=heart.iloc[90]
clust1 
clust2=heart.iloc[45]
clust2
clusters=[clust1,clust2]
clusters[0]
distObs=list(range(0,len(clusters)))
member=list()
for j in range(0,len(clusters)):
    distObs[j]=sum(((heart.iloc[0]-clusters[j]) **2))**.5
#member[0] will be member [i] in the outer loop but out the inner loop 
member.append(np.array(np.where(np.array(distObs)==min(distObs))).tolist()[0][0]) #keep the [0][0] there 
member
#make a membership list that will store the assingment for the ith obs as the ith element of the list 
# figure out how to get the mean of all cols (with describe ) and assign this to the cluster 
#loop through all rows in the df and get the assigned cluster for each in the member list
#gets the member assigment for the initalized random clusters 
member=list()
for i in range(0,len(heart)):
    for j in range(0,len(clusters)):
        distObs[j]=sum(((heart.iloc[i]-clusters[j]) **2))**.5
    member.append(np.array(np.where(np.array(distObs)==min(distObs))).tolist()[0][0])
member
#get the mean for each col of each cluster assigment and make these the new clusters 
#update assigment 
#gets the means of all vars of the obs within the cluster 
heart.iloc[np.where(np.array(member)==0)].describe().iloc[1]

member=list()
for i in range(0,len(heart)):
    for j in range(0,len(clusters)):
        distObs[j]=sum(((heart.iloc[i]-clusters[j]) **2))**.5
    member.append(np.array(np.where(np.array(distObs)==min(distObs))).tolist()[0][0])
member
#get wcss 
np.array(np.where(np.array(member)==0)).tolist()[0]
(heart.iloc[np.array(np.where(np.array(member)==0)).tolist()[0]]-clusters[0])**2
#sum each col for each row
wcss=list()
sum(pd.DataFrame((heart.iloc[np.array(np.where(np.array(member)==0)).tolist()[0]]-clusters[0])**2).sum(axis=1))
for i in range(0,len(clusters)):
    wcss.append(sum(pd.DataFrame((heart.iloc[np.array(np.where(np.array(member)==i)).tolist()[0]]-clusters[i])**2).sum(axis=1)))
sum(wcss)


#put all together 
#STEP 1: initilize and get total wcss of random centers 
clust1=heart.iloc[90]
clust2=heart.iloc[45]
clusters=[clust1,clust2]
totalwcss=list()
wcss=list()
#STEP 2: Assign membership to the random clusters 
member=list()
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
    #update cluster centers 
    for i in range(len(clusters)):
        clusters[i]=heart.iloc[np.where(np.array(member)==i)].describe().iloc[1]
    #assign membership based on updated clusters 
    member=list()
    for i in range(0,len(heart)):
        for j in range(0,len(clusters)):
            distObs[j]=sum(((heart.iloc[i]-clusters[j]) **2))**.5
        member.append(np.array(np.where(np.array(distObs)==min(distObs))).tolist()[0][0])
#get wcss for updated clusters 
    wcss=list()
    for i in range(0,len(clusters)):
        wcss.append(sum(pd.DataFrame((heart.iloc[np.array(np.where(np.array(member)==i)).tolist()[0]]-clusters[i])**2).sum(axis=1)))
    totalwcss.append(sum(wcss))
    k=k+1
len(totalwcss) #11 iterations 









#just keep repeating until the assigments do not change or get wcss and if wcss[i-1]<wcss[i-2] then continue to update clusters 








