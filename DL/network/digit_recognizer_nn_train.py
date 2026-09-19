import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from two_layer_network import TwoLayerNet
from DL.common.load_data import get_data
from sklearn.model_selection import train_test_split


x_train,x_test,y_train,y_test=get_data()

network=TwoLayerNet(input_size=784,hidden_size=50,output_size=10)
lr=0.1
batch_size=100
num_epochs=10
n=x_train.shape[0]
iter_per_epoch=np.ceil(n/batch_size)
iters_num=int(iter_per_epoch*num_epochs)
train_loss_list=[]
train_acc_list=[]
test_acc_list=[]
for i in range(iters_num):
    batch_mask=np.random.choice(n,batch_size)
    x_batch=x_train[batch_mask]
    t_batch=y_train[batch_mask]
    grad=network.numerical_gradient(x_batch,t_batch)
    for key in ("W1","W2","b1","b2"):
        network.paras[key]-=lr*grad[key]
    this_loss=network.loss(x_batch,t_batch)
    train_loss_list.append(this_loss)
    if i%iter_per_epoch==0:
        train_acc=network.accuracy(x_train,y_train)
        test_acc=network.accuracy(x_test,y_test)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)
        print(f"train acc:{train_acc},test acc:{test_acc}")