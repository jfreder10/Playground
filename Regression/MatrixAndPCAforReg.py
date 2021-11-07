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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px 

#use heart data to predict chol and use pca then visualize it then find how to do matrix operations or write functions to do them and solve a model by 'hand'
heart=pd.read_csv(r'C:\Users\fredr\Documents\StatTool\PythonCode\FunctionsAndCoolStuff\DATA\heart.csv')
heart.head()
xvars=['age','cp','thalach','oldpeak','slope','chol']
y=['target']
heartX=heart[xvars]

xDF = StandardScaler().fit_transform(heartX)
#have to set the number of components here
pca = PCA(n_components=5)
principalComponents = pca.fit_transform(xDF)
pcaDF=pd.DataFrame(principalComponents)
principalComponents.explained_variance_ratio_

#gets the variance explained by each component as a lead on how many compoents to use 
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
dir(pca)
#figure out what pca components is: Principal axes in feature space, representing the directions of maximum variance in the data. 
# The components are sorted by explained_variance_.
pca.components_

pcaFit=pca.fit(xDF)
pcaFit.inverse_transform(pcaFit)
xDF



#The singular values corresponding to each of the selected components. 
# The singular values are equal to the 2-norms of the n_components variables in the lower-dimensional space.
pca.singular_values_


#make plot that will show the percent captured as y and the pc.i as x for all pcs, (same as screeplot in R)


#figure out to set the cols to pca+a sequence the len of cols from 1 to the length 
pcaDF.columns=['pca1','pca2','pca3','pca4','pca5']
#visualize the class labels of target when plotting pca1 (as x) vs pca2 (as y) then can loop through all combos and view all 
heartPCA=pd.concat([heart,pcaDF],axis=1)
fig=px.scatter(data_frame=heartPCA,x='pca1',y='pca2',color='target') #kind of like support vector machine, it kinda seperated the classes 
fig.show()
dir(principalComponents)
principalComponents.var()
principalComponents
#fit log reg model predicting target from the components then compare this to the actual vars 

#use kfold cv to find best number of components in pcr to use to predict y, the number of compoennts are the parameters (if 10 components then 10 parameters, 
# first is component 1, 2nd is components 1 and 2, etc.)
#empty df with number of folds as the columns and the number of rows equal to the number of parameters, ith jth element is the rmse of the ith parameter for the jth 
#fold, then average the values of each row as one col as the mean rmse for that parameter or the est validation error 
