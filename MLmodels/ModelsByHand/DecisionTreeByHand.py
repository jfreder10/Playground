import pandas as pd 
import statistics as st 
import numpy as np 
#import heart data then build regression tree predicting chol from only 2 xvars (at first):
heart=pd.read_csv(r"C:\Users\fredr\Documents\StatTool\PythonCode\heart.csv")  
heart.columns
hearttree=heart[['age','chol','thalach']]
#predict chol from age and thalach
hearttree.describe()
xvars=['age','thalach']
list(hearttree.columns[hearttree.columns==xvars[0]])[0]
hearttree[list(hearttree.columns[hearttree.columns==xvars[0]])[0]] #keep last 0 (0 most to the right)
hearttree[list(hearttree.columns[hearttree.columns==xvars[0]])[0]].unique()[0] #middle 0 stays 0 first is ith outer looping var, last 0 is jth looping var
hearttree[list(hearttree.columns[hearttree.columns==xvars[0]])[0]].unique()
#write loop to print the jth unique value per the vars 
for i in range(0,len(xvars)):
    for j in range(0,len(hearttree[list(hearttree.columns[hearttree.columns==xvars[i]])[0]].unique())):
        print(xvars[i]+str(hearttree[list(hearttree.columns[hearttree.columns==xvars[i]])[0]].unique()[j]))
#for an ith var and jth value get the mean of chol for these rows that are <= xij and the mean > it and get sse on all obs in the two subsets and average together
a=0
b=0
#treat a as the ith possible split x var and b as the value of the ith possible split var
hearttree[list(hearttree.columns[hearttree.columns==xvars[a]])[0]].unique()[b]

#partition data for child node 1 of this possible split (the onse that are <= this value of this var)
submean1=hearttree['chol'][hearttree[list(hearttree.columns[hearttree.columns==xvars[a]])[0]]<=hearttree[list(hearttree.columns[
    hearttree.columns==xvars[a]])[0]].unique()[b]].mean()
    #now subtract the mean above to all chol valuse that satisfy condition of the ith var above
subobs1=hearttree['chol'][hearttree[list(hearttree.columns[hearttree.columns==xvars[a]])[0]]<=hearttree[list(hearttree.columns[
    hearttree.columns==xvars[a]])[0]].unique()[b]]
submean1
st.mean((subobs1-submean1)**2)

#partition data for child node 2 of this possible split (the onse that are > this value of this var)
submean2=hearttree['chol'][hearttree[list(hearttree.columns[hearttree.columns==xvars[a]])[0]]>hearttree[list(hearttree.columns[
    hearttree.columns==xvars[a]])[0]].unique()[b]].mean()
    #now subtract the mean above to all chol valuse that satisfy condition of the ith var above
subobs2=hearttree['chol'][hearttree[list(hearttree.columns[hearttree.columns==xvars[a]])[0]]>hearttree[list(hearttree.columns[
    hearttree.columns==xvars[a]])[0]].unique()[b]]
submean2
st.mean((subobs2-submean2)**2)
#now get the mean of these mses as the mse for the split 
st.mean([st.mean((subobs1-submean1)**2),st.mean((subobs2-submean2)**2)])





#loop through this one var as example for every possible unique value of the var, loop through the lenth of the unique vaulues of this var
#store mses for possible splits in a list
#works but will give a warn if the max unique value bc none are > than this but it dosent matter because this split would just be the mean of y for all data 
a=0
msesplits=list()
for j in range(0,len(hearttree[list(hearttree.columns[hearttree.columns==xvars[a]])[0]].unique())):
    b=j
    submean1=hearttree['chol'][hearttree[list(hearttree.columns[hearttree.columns==xvars[a]])[0]]<=hearttree[list(hearttree.columns[
        hearttree.columns==xvars[a]])[0]].unique()[b]].mean()
    subobs1=hearttree['chol'][hearttree[list(hearttree.columns[hearttree.columns==xvars[a]])[0]]<=hearttree[list(hearttree.columns[
        hearttree.columns==xvars[a]])[0]].unique()[b]]
    submean2=hearttree['chol'][hearttree[list(hearttree.columns[hearttree.columns==xvars[a]])[0]]>hearttree[list(hearttree.columns[
        hearttree.columns==xvars[a]])[0]].unique()[b]].mean()
    subobs2=hearttree['chol'][hearttree[list(hearttree.columns[hearttree.columns==xvars[a]])[0]]>hearttree[list(hearttree.columns[
        hearttree.columns==xvars[a]])[0]].unique()[b]]
    msesplits.append(st.mean([st.mean((subobs1-submean1)**2),st.mean((subobs2-submean2)**2)]))
#put list in outer loop that will store the min mse for each ith var and store the best value to split on for each var 
#first element of list 1 wiold be the mean mse of the nodes if split for the best value of the first considered var
#first element of list 2 would be the value to split on to get the lowest mean mse of the nodes for all possible values in var 1 




len(msesplits)
type(msesplits)
len(hearttree[list(hearttree.columns[hearttree.columns==xvars[a]])[0]].unique())
bestval=np.where(np.array(msesplits)==min(msesplits))
msesplits[bestval[0].tolist()[0]]
#create a cond for the rule then for each added rule add it to the list of these (each successive split)


#do same as above for next var then combine the two together and make a function for any var size 

#write while loop that will have a for loop in it that will split the data into 2^k subgegions were k is the looping var in the while loop
#for loop will split into subregions
#out of for loop but in while loop extend the sub regions into the kth position of a list 
#element one of this list will have the first 2^k df's so 2 (k==1), the second wll have 2^k so 4 (k==2), etc... 
#Keep track of the splitting var and the split value of that var for each 2^k size of partitions 
#use the split df index and the jth split var/value to split given it is in that jth df 



#keep track ofe each of the child nodes per split of the tree?????
#each noed can only feed into 2*that node or 2*that node +1 (the child nodes )
#if at node i and <= the thres value then it will go to node i*2 else it will go to node i*2+1 and where ever it goes will be node i, etc so use for loop 
#make a df with the splits 

#gets how to predict an individual for a single var, generalize to many vars 
heart.describe()
splitdf=pd.DataFrame({'split':[1,2,3], 'thresh':[54.366337,47,61], 'var':['age']*3 })
#important need to reset the index in this df otherwise the first split will be referenced as 0 (0*2=0; 0*2+1=1; so this isnt right)
splitdf.index=[1,2,3]
x=heart['age'][1]
x
split=1
splitdf['split'][1]
x<=splitdf['thresh'][split]

if(x<=splitdf['thresh'][split]):
    print('go to node'+str(split*2))
else:
   print('go to node'+str(split*2+1)) 

split=2
if(x<=splitdf['thresh'][split]):
    split=split*2
else:
   split=split*2+1

split

split
split=1
#need to determine the number of choices need to make to get an obs to a terminal node and loop through these 
for k in range(0,2):
    if(x<=splitdf['thresh'][split]):
        split=split*2
    else:
        split=split*2+1
split #predct this individual to be the mean chol for all people age > 61 
#now figure out how to do this for more than one possible vars 

