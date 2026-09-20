from collections import OrderedDict

import numpy as np
from DL.common.functions import *
from DL.common.gradient import numerical_gradient
from DL.common.layers import *
class TwoLayerNet():
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        self.paras={}
        self.paras["W1"]=np.random.randn(input_size,hidden_size)*weight_init_std

        self.paras["b1"]=np.zeros(hidden_size)
        self.paras["W2"]=np.random.randn(hidden_size,output_size)*weight_init_std
        self.paras["b2"]=np.zeros(output_size)
        self.layers=OrderedDict()
        self.layers["Affine1"]=Affine(self.paras['W1'],self.paras['b1'])
        self.layers['ReLu1']=Relu()
        self.layers["Affine2"]=Affine(self.paras['W2'],self.paras['b2'])
        self.lastLayer=SoftmaxWithLoss()


    def forword(self,x):
        for layer in self.layers.values():
            x=layer.forward(x)

        return x

    def loss(self,x,t):
        y=self.forword(x)
        return self.lastLayer.forward(y,t)
    def accuracy(self,x,t):
        y_proba=self.forword(x,t)
        y_pred=np.argmax(y_proba,axis=1)
        acc=np.sum(y_pred==t)/len(y_pred)

        return acc
    def numerical_gradient(self,x,t):
        loss_f=lambda _: self.loss(x,t)
        grads={}
        grads["W1"]=numerical_gradient(loss_f,self.paras["W1"])
        grads["b1"]=numerical_gradient(loss_f,self.paras["b1"])
        grads["W2"]=numerical_gradient(loss_f,self.paras["W2"])
        grads["b2"]=numerical_gradient(loss_f,self.paras["b2"])
    def gradient(self,x,t):
        self.loss(x,t)
        layers=list(self.layers.values())
        layers.reverse()
        dy=self.lastLayer.backward()
        for layer in self.layers:
            dy=layer.backward(dy)

        grad={}
        grad["W1"]=self.layers["Affine1"].dw
        grad["b1"]=self.layers["Affine1"].db
        grad["W2"]=self.layers["Affine2"].dw
        grad["b2"]=self.layers["Affine2"].db
        return grad


