import pandas as pd 
import numpy as np 
from scipy.stats import norm

#make an A df and a B df wher the ncols of A = the nrows of b 
index_a=range(0,3)
columns_a=range(0,7)
A_df = pd.DataFrame(index=index_a, columns=columns_a)
A_df.iloc[0]
A_df
#makes a df with 3 rows and 7 cols where each row is generated from a normal dist
for i in range(0,len(A_df)):
    A_df.iloc[i]=norm.rvs(size=len(A_df.columns),loc=3,scale=8)

#makes B df where the number of rows are equal to the number of cols in A_df 
index_b=range(0,7)
columns_b=range(0,2)
B_df = pd.DataFrame(index=index_b, columns=columns_b)

for i in range(0,len(B_df)):
    B_df.iloc[i]=norm.rvs(size=len(B_df.columns),loc=3,scale=8)

#for the ith row in a multiply it to each jth equal 
A_df
B_df
#multiply the values of the fisrt row to the values of the first col of B
A_df.iloc[0]
B_df.iloc[0:len(B_df),0]
A_df.iloc[0]*B_df.iloc[0:len(B_df),0]

#make a C matrix that will be the nrows of a and th ncols of b to store multiplied matrix 
index_c=range(0,len(A_df))
columns_c=range(0,len(B_df.columns))
C_df = pd.DataFrame(index=index_c, columns=columns_c)
C_df
#loop through the rows of A and multiply each to the cols of B then get sum of each (here rows A =3 and cols B= 2 so C will have 3 rows and 2 cols ) get sum of the 
#products and loop into C 
for i in range(0,len(A_df)):
    for j in range(0,len(B_df.columns)):
        C_df.iloc[i,j]=np.sum(A_df.iloc[i]*B_df.iloc[0:len(B_df),j])

C_df

#sanity checks 
np.sum(A_df.iloc[1]*B_df.iloc[0:len(B_df),1])

#write into function 
del A_df
del B_df
del C_df
def matrixMult(A_df,B_df):
    if(len(A_df.columns)==len(B_df)):
        index_c=range(0,len(A_df))
        columns_c=range(0,len(B_df.columns))
        C_df = pd.DataFrame(index=index_c, columns=columns_c)
        for i in range(0,len(A_df)):
            for j in range(0,len(B_df.columns)):
                C_df.iloc[i,j]=np.sum(A_df.iloc[i]*B_df.iloc[0:len(B_df),j])
        return C_df
    else:
        return print('ncol of matrix A needs to be the same as the nrow of matrix B')

index_a=range(0,3)
columns_a=range(0,7)
A1_df = pd.DataFrame(index=index_a, columns=columns_a)
A1_df.iloc[0]
for i in range(0,len(A1_df)):
    A1_df.iloc[i]=norm.rvs(size=len(A1_df.columns),loc=3,scale=8)

index_b=range(0,7)
columns_b=range(0,2)
B1_df = pd.DataFrame(index=index_b, columns=columns_b)
for i in range(0,len(B1_df)):
    B1_df.iloc[i]=norm.rvs(size=len(B1_df.columns),loc=3,scale=8)

A1_df
B1_df
#I think it worked, still need to sanity check below: 
matrixMult(A1_df,B1_df)

#get a X matrix simple reg for now so a col that is all 1's and a col that is the X values then make th inverse of X'X 









