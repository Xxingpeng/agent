def AND0(x1,x2):
    w1,w2,theta=0.5,0.5,0.7
    res=w1*x1+w2*x2
    if res<=theta:
         return 0
    else:
        return 1
import numpy as np
def ADD(x1,x2):
    x1=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.7
    res=w@x1+b


def NADD(x1, x2):
    x1 = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = 0.7
    res = w @ x1 + b
    if res <= 0:
        return 0
    else:
        return 1
