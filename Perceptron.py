# A SIMPLE PERCEPTRON FOR AND OPERATION
import numpy as np

#GENERATE TRAINING DATASET
x = np.array([[1.,1.,0.,0.],[1.,0.,1.,0.]])
 #INITIALIZE WEIGHTS
w = np.zeros(shape=(1,2))
 #INITIALIZE BIAS
b = 1.0
 #TARGET VALUES - AND OPERATION BETWEEN ROWS OF TRAINING DATESET
t = np.array([0.,1.,0.,0.])

 #LEARNING RATE
alpha = 0.15

 #TRAINING FUNCTION
def train(X,W,B,T,alpha):
     #calculating dot product of X and W and adding bias
    for i in range(500):
        #print (W)
        z = B + np.sum(np.dot(W,X))
        if (z>=2).all():
            y = 1.
        else:
            y = 0.
        W+= alpha * np.dot(T-np.dot(W,X),np.transpose(X))
        B+=alpha * (T-np.dot(W,X))

    return W

weights_final = train(x,w,b,t,alpha)
print(weights_final)
output =  (np.dot(w,x))
print ((output>=0.5)*1)
