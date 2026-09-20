from keras.src.applications.nasnet import NASNet

from DL.common.functions import sigmoid, softmax, cross_entropy_error
from DL.network.digit_recoginite import x_test
import numpy as np

class Relu:
    def __init__(self):
        self.mask=None
    def forward(self,x):
        self.mask=(x<=0)
        y=x.copy()
        y[self.mask]=0
        return y
    def backward(self,dy):
        dx=dy.copy()
        dx[self.mask]=0
        return dx

class Sigmoid:
    def __init__(self):
        self.y=None
    def forward(self,x):
        y=sigmoid(x)
        self.y=y
        return y
    def backward(self,dy):
        dx=dy.self.y*(1-self.y)
        return dx

class Affine:
    def __init__(self,w,b):
        self.W=w
        self.b=b
        self.dw=None
        self.db=None
        self.original_x_shape=None;
    def forward(self,X):
        self.original_x_shape=X.shape
        self.X=X.reshape(X.shape[0],-1)
        Y=self.X@self.W+self.b
        return Y
    def backward(self,dY):
        dX=dY@self.W.T
        self.dw=self.X.T@dY
        self.db=np.sum(dY,axis=0)
        dX=dX.reshape(*self.original_x_shape)
        return dX

class SoftmaxWithLoss:
    def __init__(self):
        self.loss=None
        self.y=None
        self.t=None
    def forward(self,x,t):
        self.t=t
        self.y=softmax(x)
        self.loss=cross_entropy_error(self.y,self.t)
    def backward(self,dy=1):
        n=self.t.shape[0]
        if self.t.size==self.y.size:
            dx=self.y-self.t
            return dx
        else:
            dx=self.y.copy()
            dx[np.arange(n), self.t] -= 1
        return dx/n




