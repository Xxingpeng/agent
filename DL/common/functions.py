import numpy as np
def step_function0(x):
    if x>0:
        return 1
    else:
        return 0

def step_function(x):
    return np.array(x>0,dtype=int)
def sigmoid(x):
    return 1/(1+np.exp(-x))
def ReLU(x):
    return np.maximum(0,x)

if __name__ ==" __main__":
    x=np.array([0,1,2,3,4,5,-1,-2,-3,-4,-5])
    print(step_function(x))
    print(sigmoid(x))