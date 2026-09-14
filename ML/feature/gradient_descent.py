def f(x):
    return x**2
def gradient(x):
    return 2*x

x_list=[]
y_list=[]

x=1
alpha=0.1
for i in range(100):
    grad = gradient(x)
    x = x - alpha * grad