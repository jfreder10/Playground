#GRADIENT DESCENT

#################################################
#Create some toy data. Set b=2.2222 and X=5.4675
import numpy as np
ct = np.ones(20) 
X = np.random.normal(size=20) #one variable, 20 rows
ct_X = np.matrix(np.column_stack((ct,X))) #combine ct and X
y = ct*2.2222 + X*5.4675

#################################################
#Which parameter values does vanilla linear regression find?
from sklearn import linear_model
# specify model
# we set fit_intercept to false because we manually added the 1 vector
reg = linear_model.LinearRegression(fit_intercept=False) 
#fit regression and extract coefficients
reg.fit(ct_X,y).coef_  
#array([2.2222, 5.4675]) #finds the right parameters


#################################################
#Manual implementation of gradient descent

#without loss of generality, below we 
#will set one weight (the bias) to the 
#true value, so we can easily display what is
#going on. 

#Feel free to redo the analysis by
#randomly initializing b. It will find the true
#value of b

#Before we start to learn the parameters
#let's do an exhaustive search. An exhaustive search
#is what we want to do when the search domain is small.
#We would compute the loss for all possible values,
#and the optimal value would then be the one that 
#results in the minimum loss. Obviously this is very expensive,
#and intractable when there are a lot of possible values.
#The reason why we are doing it here, is because
#we want to use the resulting plot later, to show 
#how gradient descent traverses the cost function.

import matplotlib.pyplot as plt
b = 2.2222
#create a range of values to try
wvals = np.array(range(-5, 16, 1))

#define the cost function
def mse(b,w):
    return (1/len(X)) * np.sum(np.power(ct*b + X*w - y,2))

#compute the cost for all w values 
#that we created (remember we are setting b to the 
#true value to keep it simple; we need to 
#define a grid of all w and b combinations
#if we want to do an exhaustive search on both).
mse_store = np.zeros(len(wvals))
for idx, w in enumerate(wvals):
    mse_store[idx] = mse(b,w)


#plt.tight_layout() #solve savefig trimming issue  
plt.plot(wvals,mse_store)
plt.title('MSE by w with true value at 5.4675\n \
            while holding b constant at true value. ')
plt.ylabel('MSE')
plt.xlabel('w')
#plt.savefig('cost.pdf')
plt.show()

#Now, suppose it would be too expensive to create
#that plot and then eyeball the minimum (i.e., do an exhaustive search). 
#We can use gradient descent to find the minimum efficiently.

#Which parameter values does gradient descent find?

#create the weight update function
def update_weights(w, b, X, y, learn_rate):
    w_deriv = 0
    b_deriv = 0
    n = len(X)

    # Calculate partial derivatives
    w_deriv = np.sum((2/float(n)) * ((w * X + b) - y) * X)
    b_deriv = np.sum((2/float(n)) * ((w * X + b) - y))

    # We subtract because the derivatives point in
    # direction of steepest ascent
    w = w - learn_rate * w_deriv
    b = b - learn_rate * b_deriv

    return b, w


#randomly initialize w (remember we are setting b to the true value for didactical purposes) 
np.random.seed(seed=17299) #in practice we would not set this, we do it here for reproducibility purposes
w_init = np.random.uniform(low=-5,high=15)
print(w_init)
b_init = 2.2222 #set this to the true value, so we can see how w evolves holding this constant

#define function to call and plot
def call_and_plot_weight_update(learn_rate, b=b_init, w=w_init):
    
    #call weight update function 100 times (epochs) and plot
    w_store = np.zeros(100) #don't need this if we do not want to plot
    mse_store_gd = np.zeros(100) #don't need this if we do not want to plot
    for idx, epochs in enumerate(range(100)):
        b, w = update_weights(w=w, b=b, X=X, y=y, learn_rate=learn_rate)
        w_store[idx] = w #don't need this if we do not want to plot
        mse_store_gd[idx] = mse(b,w) #don't need this if we do not want to plot
    
    #remainder of this function is plotting
    plt.plot(wvals,mse_store)
    plt.title('Learning true w at 5.4675 \n while holding b constant at true value. \
              \n Learning rate = ' + str(learn_rate) + '. \
              \n Random init w =' + str(round(w_init,4)) + ' \
              \n  w found =' + str(round(w,4)))
    plt.ylabel('MSE')
    plt.xlabel('w')
    plt.plot(np.concatenate((np.array([w_init]),w_store)), \
             np.concatenate((np.array([mse(b,w_init)]),mse_store_gd)), \
             color='green',marker="x")
    plt.show()
    #plt.tight_layout()
    #plt.savefig('learn_rate' + str(learn_rate) + '.pdf')




#Use learning rate 0.0001
#the updates will be very small and since we only do 100 iterations,
#we won't even come close to the minimum if the randomly 
#chosen starting value is far from the minimum
call_and_plot_weight_update(learn_rate=0.0001)   


#Use learning rate 0.01
#we are very close, with a little more iterations we
#would have gotten there.
call_and_plot_weight_update(learn_rate=0.01)   


#Use learning rate 0.1
#It finds the the optimal w right on
#this is thus a good learning rate
call_and_plot_weight_update(learn_rate=0.1)   


#Use learning rate 0.5
#We see that it starts to bounce around
#Trouble will start soon
#It can still find the minimum
call_and_plot_weight_update(learn_rate=0.5)   


#Further increasing the learn_rate will eventually diverge
call_and_plot_weight_update(learn_rate=.....)   


#This demonstrates how gradient descent works and 
#how the learning rate impacts learning.


#If the learning rate is too small, learning happens very slowly
#If the learning rate is too high, we may overshoot the minimum and even diverge.
