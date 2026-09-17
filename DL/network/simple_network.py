import numpy as np

from DL.common.functions import sigmoid


def init_network():
    network={}
    network['W1']=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['W2']=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['W3']=np.array([[0.1,0.3],[0.2,0.4]])
    network['b1']=np.array([0.1,0.2,0.3])
    network['b2']=np.array([0.1,0.2])
    network['b3']=np.array([0.1,0.2])
    return network
def forword(network,x):
    w1,w2,w3=network['W1'],network['W2'],network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']
    a1=x@w1+b1
    z1=sigmoid(a1)
    a2=a1@w2+b2
    z2=sigmoid(a2)
    a3=a2@w3+b3
    y=sigmoid(a3)
    return y
x=np.array([1.0,0.5])
network=init_network()
y=forword(network,x)
print(y)

