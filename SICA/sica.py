import numpy as np
from numpy import random

def makeDiff(data):
    diffData = np.mat(np.zeros((data.shape[0],data.shape[1])))
    for i in range(data.shape[1]-1):
        diffData[:,i] = data[:,i] - data[:,i+1]
    diffData[:,-1] = data[:,-1] - data[:,0]
    return np.mat(diffData)

def getRandomW(length,height):
    W = random.random(size=(length,height))
    return W
    
def eigOrth(Data):
    data = Data.copy()
    D,E = np.linalg.eig(data.dot(data.T))
    for i in range(len(D)):
        D[i] = D[i]**0.5
    D = np.mat(np.diag(D))
    D = D.I
    data = D*E.T*data
    return data,D*E.T

def Function(data):                                 
    def G(x):
        y = -np.exp(-0.5*(x**2))
        return y
    length,bordth = data.shape
    output = np.zeros((length,bordth))
    for i in range(length):
        for j in range(bordth):
            output[i,j] = G(data[i,j])
    return output
    
def GFunction(data):                                 
    def G(x):
        y = x*np.exp(-0.5*(x**2))
        return y
    length,bordth = data.shape
    output = np.zeros((length,bordth))
    for i in range(length):
        for j in range(bordth):
            output[i,j] = G(data[i,j])
    return output

def gFunction(data):
    def g(x):
        y = -1*(x**2)*np.exp(-0.5*(x**2))+np.exp(-0.5*(x**2))
        return y
    length,bordth = data.shape
    output = np.zeros((length,bordth))
    for i in range(length):
        for j in range(bordth):
            output[i,j] = g(data[i,j])
    return output 

def distance(W,oldW):
    return abs(abs(float(W.T*oldW)) - 1.)    

def caculateR(A,w,data,a):
    firstPart = (w.T.dot(data))*(GFunction(w.T.dot(data)).T)/data.shape[1]
    secondPart = w.T*((A+A.T)*w)/data.shape[1]
    return float(-0.5*(firstPart+a*secondPart))

def firstDerivate(A,w,R,data,a):
    firstPart = data*GFunction(w.T.dot(data)).T/data.shape[1]
    secondPart = (A+A.T)*w/data.shape[1]
    return firstPart+a*secondPart+2*R*w

def secondDerivate(A,w,R,data,a):
    firstPart = np.mean(gFunction(w.T.dot(data)),1)
    secondPart = (A+A.T)/data.shape[1]
    return firstPart*np.eye(A.shape[0]) + a*secondPart + 2*R*np.eye(A.shape[0])

class SICA:
    def __init__(self,conponent = -1,a = 10):
        self._conponent = conponent
        self._W = 0
        self._a = a
        self._data = 0
        self._mean = 0
        self._orth = 0
        
    def fit_transform(self,data):
        self._data = data.copy()
        if self._conponent == -1:
            self._conponent = data.shape[0]
        datamean = np.mean(data,1)
        self._mean = datamean
        for i in range(data.shape[1]):
            data[:,i] -= datamean
        odata,self._orth = eigOrth(data)
        self._data = odata*np.sqrt(self._data.shape[1])

        diffData = makeDiff(self._data)
        A = diffData*diffData.T
        self._W = getRandomW(data.shape[0],data.shape[0])
        self._W,_ = eigOrth(self._W.T)
        self._W = self._W.T

        MAX_T = 200
        for i in range(self._conponent):
            w = self._W[:,i]
            j,t  = 0,1
            while (j < MAX_T) and (t > 1e-6):
                oldw = w.copy()
                R = caculateR(A,w,self._data,self._a)
                firstD = firstDerivate(A,w,R,self._data,self._a)
                secondD = secondDerivate(A,w,R,self._data,self._a)
                Q = secondD.I.dot(firstD)
                w -= Q
                temp = np.zeros((self._W.shape[0],1))
                for k in range(i):
                    temp += float(w.T*self._W[:,k])*self._W[:,k]
                w = w - temp
                w = w/np.sqrt(w.T*w)
                t = distance(w,oldw)
                self._W[:,i] = w
                j += 1    
        SlowPart = np.zeros((1,data.shape[0]))
        for i in range(data.shape[0]):
            w = self._W[:,i]
            secondPart = (w.T*diffData*diffData.T*w)/data.shape[1]
            SlowPart[0,i] = float(secondPart)
            
        from collections import Counter
        SICAres = {}
        for i in range(data.shape[0]):
            SICAres[i] =  SlowPart[0,i]
        z = Counter(SICAres).most_common()
        WW = np.zeros(self._W.shape)
        for i in range(WW.shape[1]):
            WW[:,WW.shape[1]-1-i] = np.ravel(self._W[:,z[i][0]])
        self._W = WW[:,0:self._conponent]
        return (self._W.T*self._data)
    
    def transform(self,data):     
        data -= np.mat(self._mean).T
        data = self._W.T*self._orth*data*np.sqrt(self._data.shape[1])
        return data
    
    def calculateObj(self):
        data = self._data
        ICAPart = np.mean(Function(self._W.T.dot(data)),1)
        
        diffData = makeDiff(data)
        SlowPart = np.zeros((1,self._conponent))
        for i in range(self._conponent):
            w = np.mat(self._W[:,i]).T
            secondPart = ((w.T*diffData).dot(diffData.T*w))/data.shape[1]
            SlowPart[0,i] = float(secondPart)
            
        return ICAPart,np.ravel(SlowPart),self._a*ICAPart+self._a*np.ravel(SlowPart)





