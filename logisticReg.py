# Helper functions to do logistic regression
# To use the logRegCost function needs to be minimised, use the logRegGrad method to provide derivative
#

import cv2
import numpy as np
import scipy.io as sio
import csv as csv
from sklearn.preprocessing import normalize


def featureNormalize(data):
    mu = data.mean(0)
    data_norm = data.__sub__(mu)
    sigma = np.std(data_norm, axis=0,ddof=1)
    data_norm = data_norm.__div__(sigma)
    return data_norm;

def addFirstOnes(data):
    return np.concatenate((np.ones((np.size(data,0),1)),data),1)
    
def sigmoid(z):
    return 1/(1+np.exp(-z))

def logRegGrad(theta, data_x, data_y, lamb):   
    m = float(np.size(data_y))

    theta=np.array([theta]).T

    temp = np.array(theta); 
    temp[0] = 0;

    ha = data_x.dot(theta)
    h=sigmoid(ha);
    
    grad = 1/m * ((h-data_y).T.dot(data_x)).T;
    grad = grad + ((lamb/m)*temp);
    
    return grad.T[0]
    

def logRegCost(theta, data_x, data_y, lamb):
    m = float(np.size(data_y))
    theta=np.array([theta]).T
  
    
    ha = data_x.dot(theta)
    h=sigmoid(ha);
    
    J = 1/m *((-data_y.T.dot(np.log(h))-(1-data_y.T).dot(np.log(1-h))));

    temp = np.array(theta);  
    temp[0] = 0;   # because we don't add anything for j = 0
    J = J + (lamb/(2*m))*sum(np.power(temp,2));
    
    return J[0,0]
    
def predict(theta, data_x):
    n = np.size(data_x,1)
    theta=np.array([theta]).T  

    ha = data_x.dot(theta)
    p=sigmoid(ha);

    for i in range(0,np.size(data_x,0)):
       if p[i]>=0.5:
           p[i]=1
       else:
           p[i]=0
    return p

def testError(theta, data_x,data_y):
    m = float(np.size(data_y))
    sum =0
    p=predict(theta, data_x);
    
    for i in range(0,np.size(data_x,0)):
       if p[i,0]==1 and data_y[0,i]==0:
           sum = sum+1;
       elif p[i,0]==0 and data_y[0,i]==1:
           sum = sum+1;
    
    return 1/m * sum
    
