import random
import math
import TryAll
import pandas as pd
import itertools
import statistics as st 
import numpy as np

banks=pd.read_csv(r'C:\Users\fredr\Documents\StatTool\BZAN544\project\Hawaii_Banks_and_Credit_Unions.csv')

#make the lists into np.arrays 
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