import numpy as np
from DL.common.functions import *
from DL.common.gradient import numerical_gradient
class TwoLayerNet():
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        self.paras={}
        self.paras["W1"]=np.random.randn(input_size,hidden_size)*weight_init_std

        self.paras["b1"]=np.zeros(hidden_size)
        self.paras["W2"]=np.random.randn(hidden_size,output_size)*weight_init_std
        self.paras["b2"]=np.zeros(output_size)


    def forword(self,x,t):
        W1,W2=self.paras["W1"],self.paras["W2"]
        b1,b2=self.paras["b1"],self.paras["b2"]
        a1=np.dot(x,W1)+b1
        z1=sigmoid(a1)
        a2=np.dot(z1,W2)+b2
        y=softmax(a2)

        return y

    def loss(self,x,t):
        y=self.forword(x,t)
        return cross_entropy_error(y,t)
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

