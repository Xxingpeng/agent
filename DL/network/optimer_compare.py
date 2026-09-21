from collections import OrderedDict
import numpy as np
from matplotlib import pyplot as plt

from DL.common.optimizer import SGD, Momentum, AdaGrad, Adam


def f(x,y):
    return x**2/20+y**2
def f_grad(x,y):
    return x/10,2*y
init_pos=(-7,2.0)


params,grads={},{}
optimizers=OrderedDict()
optimizers['SGD']=SGD(lr=0.1)
optimizers['Momentum']=Momentum(lr=0.1,momentum=0.9)
optimizers['AdaGrad']=AdaGrad(lr=0.1)
optimizers['Adam']=Adam(lr=0.1)

for key in optimizers:
    optimizer=optimizers[key]
    params['x'],params['y']=init_pos[0],init_pos[1]
    x_history,y_history=[],[]
    for epoch in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        grads['x'],grads['y']=f_grad(params['x'],params['y'])
        optimizer.update(params,grads)



