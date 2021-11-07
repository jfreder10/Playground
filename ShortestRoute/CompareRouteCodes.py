import random
import math
import TryAll
import pandas as pd
import itertools
import statistics as st 
import numpy as np

banks=pd.read_csv(r'C:\Users\fredr\Documents\StatTool\BZAN544\project\Hawaii_Banks_and_Credit_Unions.csv')
#function to get the selected island 
del dfFun
del island
def islandChoice(island1):
    dfFun=banks[['X','Y','name','location']][banks['island']==island1]
    dfFun.index=list(range(0,len(dfFun)))
    dfFun.head()
    dfFun=dfFun.dropna()
    dfFun.index=list(range(0,len(dfFun)))
    return dfFun

#Code from Dr.Day 
#funtion 
def route(dfExample):
    pointsList=list()
    for i in range(0,len(dfExample)):
        pointsList.append((dfExample['X'].iloc[i],dfExample['Y'].iloc[i]))
    allPaths = []
    allDists = []
# Let every point be the starting point for a path
    for i, startPtIndex in enumerate(pointsList):
    # initialize a pt holder, path, and holder for distance
        ptsLeft = pointsList.copy()
        aPath = []
        distance = 0
    # move the first point to the path
        aPath.append(ptsLeft.pop(i))
    # find the next point to add to path.
        while len(ptsLeft)>0:
            currPt = aPath[-1]
            distToCurrPt = [math.dist(currPt, pt) for pt in ptsLeft]
            minDistIndex = distToCurrPt.index(min(distToCurrPt))
            distance += distToCurrPt[minDistIndex]
            aPath.append(ptsLeft.pop(minDistIndex))
        allDists.append(distance)
        allPaths.append(aPath)
    bestDist = min(allDists)
    return allPaths[allDists.index(bestDist)]

#breakdown of function 
dfExample=islandChoice('Hawaii')



pointsList=list()
    for i in range(0,len(dfExample)):
        pointsList.append((dfExample['X'].iloc[i],dfExample['Y'].iloc[i]))

pointsList #this is a list of tuples 
#initilize allPaths and allDists lists 
allPaths = []
allDists = []

dir(enumerate(pointsList))
help(enumerate)
dir(enumerate)
for i, startPtIndex in enumerate(pointsList):
    ptsLeft = pointsList.copy()
    aPath = []
    distance = 0
    aPath.append(ptsLeft.pop(i))
    while len(ptsLeft)>0:
        currPt = aPath[-1] 
        distToCurrPt = [math.dist(currPt, pt) for pt in ptsLeft] 
        minDistIndex = distToCurrPt.index(min(distToCurrPt)) 
        distance += distToCurrPt[minDistIndex] 
        aPath.append(ptsLeft.pop(minDistIndex))
    
    allPaths.append(aPath)
    allDists.append(distance)


#code i made for the best route for a single starting point 
def bestithpath(a,index1):
    path=[index1]
    #make dists a np.array (np runs faster built from c )
    dists=list(range(0,len(a)))
    sumdists=0
    while(len(path)<len(a)):
        for i in range(0,len(a)):
            if(i in path):
                dists[i]=0
            else:
                dists[i]=sum(((a.iloc[path[len(path)-1]]-a.iloc[i])**2)**.5)
        path.append(np.array(np.where(np.array(dists)==min(np.array(dists)[np.array(np.where(np.array(dists)!=0)).tolist()[0]]))).tolist()[0][0])
        sumdists+=min(np.array(dists)[np.array(np.where(np.array(dists)!=0)).tolist()[0]])
    return path, sumdists

#make a df with several repearing points 

exDf=pd.DataFrame({'X':[1,1,1,5,6,8,9], 'Y':[2,2,2,6,8,10,13]})

bestithpath(exDf,0) #gives the error because of repeats fix so it doesent do this by makeing a paths left list that will be used to get the value of the next 
#path with the index of dists then we will remove it if it is the min distance 





#re code the ith path functon so that it will only have to loop through the positions not visited yet 




path=[0]
#make this all of the indexes in the df other than the one in path 
pathToGo=list(exDf.index[exDf.index!=0])

#get the dist from path[len(path)-1] or path[-1] row of df to each one in path then store these in dists, then select the min dist (if more than one then add all
# of these points to path and remove all from paths to go ) and add
dists=list()
math.dist(exDf.iloc[path[-1]], exDf.iloc[pathToGo[0]])

for i in range(0,len(pathToGo)):
    dists.append(math.dist(exDf.iloc[path[-1]], exDf.iloc[pathToGo[i]]))



#get the index of the min of the dists list
dists
nextpos=[i for i, n in enumerate(dists) if n == min(dists)]
nextpos
np.array(pathToGo)[nextpos].tolist()
#add these to path then remove these from paths to go 

#make if else if the len(np.array(pathToGo)[nextpos].tolist())>1 then use extend else use append 
if(len(np.array(pathToGo)[nextpos].tolist())>1):
    path.extend(np.array(pathToGo)[nextpos].tolist())
else:
   path.append(np.array(pathToGo)[nextpos].tolist()) 
path

#now remove these from pathsToGo 
pathToGo=[ elem for elem in pathToGo if elem  not in np.array(pathToGo)[nextpos].tolist() ]

#check path and pathTogo 
path
pathToGo
#get the distance of the move to the closest point as the sum of the distances min()




#now get dist of row path[-1] toeach of the rows in pathTogo and repeat, etc...



#put all in while loop: while(len(pathTogo)>0): do all the stuff abov somehow 
#create random points as exDf
def createPoints(num = 5):
    """ returns a dataframe with num rows and columns x, y"""
    thisData = [ [random.uniform(-10, 10), random.uniform(-10, 10)] for i in range(num)]
    return pd.DataFrame(data = thisData, columns = ['x', 'y'])


# I think this is working set exDf to dfexapmle
dfExample=islandChoice('Hawaii')
exDf=dfExample[['X','Y']]
#exDf=createPoints(7)
path=[0]
pathToGo=list(exDf.index[exDf.index!=path[0]])
distanceI=0
#make a distIthPath that keeps track of the sum of the distances for each move to the closest point 
while(len(pathToGo)>0):
    dists=list()
    for i in range(0,len(pathToGo)):
        dists.append(math.dist(exDf.iloc[path[-1]], exDf.iloc[pathToGo[i]]))
    nextpos=[i for i, n in enumerate(dists) if n == min(dists)]
    if(len(np.array(pathToGo)[nextpos].tolist())>1):
        path.extend(np.array(pathToGo)[nextpos].tolist())
    else:
        path.append(np.array(pathToGo)[nextpos].tolist()[0]) 
    
    distanceI+=min(dists)
    pathToGo=[ elem for elem in pathToGo if elem  not in np.array(pathToGo)[nextpos].tolist() ]

path
exDf
distanceI #the sum of the distances 
#put outer loop around this and loop through all possible starting points 

#this is def faster then the original but not as fast as Dr. Day's
#skeleton of function to get the best route given each point can be a possible starting point 
distsIthRoute=list()
for i in range(0,len(exDf)):
    path=[i]
    pathToGo=list(exDf.index[exDf.index!=path[0]])
    distanceI=0
    while(len(pathToGo)>0):
        dists=list()
        for i in range(0,len(pathToGo)):
            dists.append(math.dist(exDf.iloc[path[-1]], exDf.iloc[pathToGo[i]]))
        nextpos=[i for i, n in enumerate(dists) if n == min(dists)]
        if(len(np.array(pathToGo)[nextpos].tolist())>1):
            path.extend(np.array(pathToGo)[nextpos].tolist())
        else:
            path.append(np.array(pathToGo)[nextpos].tolist()[0]) 
    
        distanceI+=min(dists)
        pathToGo=[ elem for elem in pathToGo if elem  not in np.array(pathToGo)[nextpos].tolist() ]
    
    distsIthRoute.append(distanceI)

min(distsIthRoute)




































