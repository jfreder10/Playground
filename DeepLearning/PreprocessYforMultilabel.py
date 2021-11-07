!pip3 install --upgrade tensorflow
#alternatively in the terminal: 
#python3 -m pip install --upgrade tensorflow
import tensorflow as tf
tf.__version__

#############################################
#Create some nonlinear toy data.
import matplotlib.pyplot as plt
import numpy as np
ct = np.ones(20) 
X1 = np.random.normal(size=20) #variable, 20 rows
X2 = np.random.normal(size=20) #variable, 20 rows
X = np.array(np.column_stack((X1,X2)))
y = ct*2.2222 + X1*5.4675 + X2*10.1115 - 3*X1**2

ycat = []
for i in y:
    if i <= np.min(y):
        ycat.append(['cat 0'])
    elif i <= np.quantile(y,0.25):
        ycat.append(['cat 1','cat 2'])
    elif i <= np.quantile(y,0.50):
        ycat.append(['cat 2'])        
    elif i <= np.quantile(y,0.75):        
        ycat.append(['cat 3'])        
    elif i <= np.max(y):        
        ycat.append(['cat 4'])

ycat

#predictor variables are X
#categorical response variable is ycat


#############################################
#preprocess the data



#step 1: get unique categories
#flatten nested list
y_cat_flat = [item for subcat in ycat for item in subcat]
unique_categories = np.unique(np.array(y_cat_flat))
unique_categories

#step 2: make lookup table with index value for each category
indices = np.array(range(len(unique_categories)), dtype = np.int64)
indices
lookuptable = np.column_stack([unique_categories,indices])

#step 3: apply lookup table to data
main_result = []
for i in range(len(ycat)):
    res = []
    for ii in range(len(ycat[i])):
        res.append(int(lookuptable[lookuptable[:,0]==ycat[i][ii],1][0]))
    main_result.append(res)

main_result

#step 4: create dummy encoded data

y_final = np.array([list(np.zeros(len(unique_categories))) for i in range(len(ycat))])
for i in range(len(ycat)):
    for ii in range(len(main_result[i])):
        y_final[i,main_result[i][ii]] = 1        

y_final






