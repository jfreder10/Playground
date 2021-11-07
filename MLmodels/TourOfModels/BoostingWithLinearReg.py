import pandas as pd
import numpy as np
import itertools
import statistics as st
import statsmodels as stm
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols 
import statsmodels.api as regMods
#import some numeric data then fit m models to predict some variable succesively predict the errors of each iteration then update the predictions to get the 
#current residual to predict 
stuaid=pd.read_csv(r"C:\Users\fredr\Documents\StatTool\BZAN543\StudentAidAnalysis\mmcreal.csv") 
#fit an ols model predicting a numeric var from another series of numeric vars get the error then fit another model to predict this error, then multiply 
#this to the learning rate and add to previous prediction (prediction used to est the residuals), and repeat 
#i have a good template in R of how to do this 
stuaid.columns
stuaid=stuaid.rename(columns={"Aid Offered": "AidOffered"})
stuaid.columns
#store the possible x vars in a list 
xvars=['HSGPA','EFC']
model = regMods.OLS(endog=stuaid['AidOffered'], exog=stuaid[xvars[0]], missing='drop').fit()
model.pvalues[0:len(model.pvalues)]
