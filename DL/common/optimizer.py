import numpy as np
from anyio.functools import lru_cache_items
from tensorflow.python.ops.gen_math_ops import select_v2_eager_fallback


class SGD:
    def __init__(self,lr=0.01):
        self.lr=lr
    def update(self,params,grads):
        for key in params.keys():
            params[key]-=self.lr*grads[key]


class Momentum:
    def __init__(self,lr=0.01,momentum=0.9):
        self.lr=lr
        self.momentum=momentum
        self.v=None
    def update(self,params,grads):
        if self.v is None:
            self.v={}
            for key,val in params.items():
                self.v[key]=np.zeros_like(val)
        for key in params.keys():
            self.v[key]=self.momentum*self.v[key]-self.lr*grads[key]
            params[key]+=self.v[key]


class  AdaGrad:
    def __intit__(self,lr=0.01):
        self.lr=lr
        self.h=None

    def update(self,params,grads):
        if self.h is None:
            self.h={}
            for key,val in params.items():
                self.h[key]=np.zeros_like(val)
        for key in params.keys():
            self.h[key]+=grads[key]*grads[key]

            params[key]-=self.lr*grads[key]/(np.sqrt(self.h[key])+1e-8)

class RMSProp:
    def __init__(self,lr=0.01,decay=0.9):
        self.lr=lr
        self.decay=decay
        self.h=None
    def update(self,params,grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.decay*self.h[key]+(1-self.decay)*grads[key]*grads[key]
class Adam:
    def __init__(self,lr=0.01,beta1=0.9,beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
    def update(self,params,grads):
        if self.v is None:
            self.v,self.h={},{}
            for key,val in params.items():
                self.m[key]=np.zeros_like(val)
                self.v[key]=np.zeros_like(val)

        self.t+=1
        lr_t=self.lr*np.sqrt(1-self.beta2**self.t)/(1-self.beta1**self.t)
        for key in params.keys():
            self.v[key]=self.beta1*self.v[key]+(1-self.beta1)*grads[key]
            self.h[key]=self.beta2*self.h[key]+(1-self.beta2)*(grads[key]**2)
            params[key]-=lr_t*self.v[key]/(np.sqrt(self.h[key])+1e-8)
