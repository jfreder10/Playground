##################################################
#Create some toy data. Set b=2.2222 and X=5.4675
import numpy as np
ct = np.ones(20) 
X = np.random.normal(size=20) #one variable, 20 rows
ct_X = np.array(np.column_stack((ct,X))) #combine ct and X
y = ct*2.2222 + X*5.4675
import matplotlib.pyplot as plt
plt.scatter(X,y)
plt.show()
##################################################
#Which parameter values does vanilla linear regression find?
from sklearn import linear_model
# specify model
# we set fit_intercept to false because we manually added the 1 vector
reg = linear_model.LinearRegression(fit_intercept=False) 
#fit regression and extract coefficients
reg.fit(ct_X,y).coef_  
#array([2.2222, 5.4675]) #finds the right parameters


##################################################
#####incremental implementation of a linear model

eps = 0.0000000000001

#generate random initial weights
W = np.array(np.random.normal(size=2)).reshape(2,1)

#define error function
def error(weights,X,y): 
    #matrix multiply data with weights to obtain y_hat
    #np.matmul(ct_X, W)
    y_hat = np.matmul(X, weights).flatten()
    #compute error
    error = (y-y_hat)**2
    #return error
    return error

# Let's try out our function for the first record:
# error(W,ct_X[0],y[0])


#training loop using the learning rule to update the weights record by record
#we are going to loop 1000 times through all the instances
for j in range(1000): #j for epoch
    for i in range(len(ct_X)): #i for instances
        #we could randomly sample (without replacement) from ct_X
        
        #compute impact on error of increasing weight by tiny amount
        #(numerical derivative)
        impact1 = (error(np.array([W[0]+eps,W[1]]).reshape(2,1),ct_X[i],y[i]) - 
                   error(np.array([W[0]-eps,W[1]]).reshape(2,1),ct_X[i],y[i])) / (2*eps)
        impact2 = (error(np.array([W[0],W[1]+eps]).reshape(2,1),ct_X[i],y[i]) - 
                   error(np.array([W[0],W[1]-eps]).reshape(2,1),ct_X[i],y[i])) / (2*eps)
    
        #update weights
        W[0] = W[0] - 0.01*impact1
        W[1] = W[1] - 0.01*impact2

#display result
W
#we find exactly the same parameters


#plot the observed data:
import matplotlib.pyplot as plt
plt.scatter(X,y)
plt.show()

#plot the predicted values
Xplot = np.arange(min(X),max(X),step = 0.1)
yplot = []

for x in Xplot:
    yplot.append(ct[0]*W[0] + x*W[1])

plt.plot(Xplot,yplot)
plt.show()



##################################################
#####Now that we understand the incremental learning component, let's
#####make it a neural network.

#Create some nonlinear toy data.
import numpy as np
ct = np.ones(20) 
X = np.random.normal(size=20) #one variable, 20 rows
ct_X = np.array(np.column_stack((ct,X))) #combine ct and X
y = ct*2.2222 + X*5.4675 - 3*X**2

plt.scatter(X,y)
plt.show()

####Neural network with one hidden layer containing two sigmoid activated neurons:

#generate initial weights
#weights between input layer and first hidden layer
W_i_h = np.array(np.random.normal(size=4)).reshape(2,2)
#weights between first hidden layer and output layer
W_h_o = np.array(np.random.normal(size=2)).reshape(2,1)

# W_i_h:
    
#     w1 w2
#     w3 w4
    
    
# W_h_o:
    
#     w5
#     w6

def error(W_i_h,W_h_o,X,y): 
    
    #compute weighted sum first layer
    z_1and2 = np.matmul(X, W_i_h)

    #activate the weighted sum
    h_1and2 = 1 / (1 + np.exp(- z_1and2))

    #compute weighted sum output layer
    o = np.matmul(h_1and2, W_h_o)

    #compute error
    error = (y-o)**2

    return error

# Try function for first record:
# i = 0
# error(W_i_h,W_h_o,ct_X[i],y[i])

#incoming weights for hidden node h1
#weight 1
# W_i_h[0,0]
#weight 3
# W_i_h[1,0]

#incoming weights for hidden node h2
#weight 2
# W_i_h[0,1]
#weight 4
# W_i_h[1,1]

#incoming weights for output node
#weight 5
# W_h_o[0,0]
#weight 6
# W_h_o[1,0]

#training loop using the learning rule to update the weights record by record
#we are going to loop 1000 times through all the instances
for j in range(1000): #j for epoch
    for i in range(len(ct_X)): #i for instances
        
        
        #format for w1 until w4
#         impact =  (
#                   error(np.array([]).reshape(2,2),W_h_o,ct_X[i],y[i]) -
#                   error(np.array([]).reshape(2,2),W_h_o,ct_X[i],y[i])
#                   )/ (2*eps)

#         #np array is filled and reshape goes row by row
#         W_i_h[0,0],W_i_h[0,1],W_i_h[1,0],W_i_h[1,1]
        
        
#         #format for w5 and w6
#         impact =  (
#                   error(W_i_h,np.array([]).reshape(2,1),ct_X[i],y[i]) -
#                   error(W_i_h,np.array([]).reshape(2,1),ct_X[i],y[i])
#                   )/ (2*eps)

#         #np array is filled and reshape goes row by row
#         W_h_o[0,0],W_h_o[1,0]     
        
        #compute impact on error of increasing weight
        #weight 1
        
        impact1 = (error(np.array([W_i_h[0,0]+eps,W_i_h[0,1],W_i_h[1,0],W_i_h[1,1]]).reshape(2,2),W_h_o,ct_X[i],y[i]) - 
                   error(np.array([W_i_h[0,0]-eps,W_i_h[0,1],W_i_h[1,0],W_i_h[1,1]]).reshape(2,2),W_h_o,ct_X[i],y[i])) / (2*eps)
        #weight 3
        
        impact3 = (error(np.array([W_i_h[0,0],W_i_h[0,1],W_i_h[1,0]+eps,W_i_h[1,1]]).reshape(2,2),W_h_o,ct_X[i],y[i]) - 
                   error(np.array([W_i_h[0,0],W_i_h[0,1],W_i_h[1,0]-eps,W_i_h[1,1]]).reshape(2,2),W_h_o,ct_X[i],y[i])) / (2*eps)
        #weight 2
        
        impact2 = (error(np.array([W_i_h[0,0],W_i_h[0,1]+eps,W_i_h[1,0],W_i_h[1,1]]).reshape(2,2),W_h_o,ct_X[i],y[i]) - 
                   error(np.array([W_i_h[0,0],W_i_h[0,1]-eps,W_i_h[1,0],W_i_h[1,1]]).reshape(2,2),W_h_o,ct_X[i],y[i])) / (2*eps)
        
        #weight 4
        
        impact4 = (error(np.array([W_i_h[0,0],W_i_h[0,1],W_i_h[1,0],W_i_h[1,1]+eps]).reshape(2,2),W_h_o,ct_X[i],y[i]) - 
                   error(np.array([W_i_h[0,0],W_i_h[0,1],W_i_h[1,0],W_i_h[1,1]-eps]).reshape(2,2),W_h_o,ct_X[i],y[i])) / (2*eps)
        
        #weight 5
        
        impact5 = (error(W_i_h,np.array([W_h_o[0,0]+eps,W_h_o[1,0]]).reshape(2,1),ct_X[i],y[i]) - 
                   error(W_i_h,np.array([W_h_o[0,0]-eps,W_h_o[1,0]]).reshape(2,1),ct_X[i],y[i])) / (2*eps)
        
        #weight 6
        
        impact6 = (error(W_i_h,np.array([W_h_o[0,0],W_h_o[1,0]+eps]).reshape(2,1),ct_X[i],y[i]) - 
                   error(W_i_h,np.array([W_h_o[0,0],W_h_o[1,0]-eps]).reshape(2,1),ct_X[i],y[i])) / (2*eps)        
        
        #update weights
        #1
        W_i_h[0,0] = W_i_h[0,0] - 0.01*impact1
        #3
        W_i_h[1,0] = W_i_h[1,0] - 0.01*impact3
        #2
        W_i_h[0,1] = W_i_h[0,1] - 0.01*impact2
        #4
        W_i_h[1,1] = W_i_h[1,1] - 0.01*impact4
        #5
        W_h_o[0,0] = W_h_o[0,0] - 0.01*impact5
        #6
        W_h_o[1,0] = W_h_o[1,0] - 0.01*impact6

W_i_h
W_h_o



#plot the observed data
import matplotlib.pyplot as plt
plt.scatter(X,y)
plt.show()

#plot the predictd values
ct_X = np.array(np.column_stack((ct,np.sort(X))))
z_1and2 = np.matmul(ct_X, W_i_h)
h_1and2 = 1 / (1 + np.exp(- z_1and2))
o = np.matmul(h_1and2, W_h_o)
plt.plot(np.sort(X),o)
plt.show()


#Automatically finds the relationship. We did not specify the quadratic term!

#This code 
#-constitutes the neural network algorithm coded from scratch
#-shows how to implement icremental learning in a linear model and neural network
#-shows that neural networks can automatically represent the function without manually specifying it