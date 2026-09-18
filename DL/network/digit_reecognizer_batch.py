import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
from torch.nn.functional import softmax

from DL.common.functions import sigmoid
from ML.feature.heart_disease import x_train, x_test


def get_data():
    dataset=pd.read_csv("../../../ML/data/train.csv")
    x=dataset.drop(columns='label',axis=1)
    y=dataset['label']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
    scaler=MinMaxScaler()
    scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)
    return x_test,y_test

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
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = x @ w1 + b1
    z1 = sigmoid(a1)
    a2 = a1 @ w2 + b2
    z2 = sigmoid(a2)
    a3 = a2 @ w3 + b3
    y = softmax(a3)
    return y


x_test,y_test=get_data()
network=init_network()
batch_size=100
n=x_test.shape[0]
acc_cnt=0
for i in range(0,n,batch_size):
    x_batch=x_test[i,i+batch_size]
    Y_batch=y_test[i,i+batch_size]
    #前向传播
    y_proba=forword(network,x_batch)
    y_pred=np.argmax(y_proba,axis=1)