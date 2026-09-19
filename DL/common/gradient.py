# 数值微分
def numerical_diff0(f, x):
    h = 1e-4
    return (f(x+h) - f(x))/h
# 中心差分实现
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h))/(2 * h)

import numpy as np

# 利用中心差分计算梯度，x为向量
def _numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    # 遍历x向量中的每个自变量xi
    for i in range(x.size):
        tmp = x[i]   # 临时保存xi的值，用于还原
        # 对当前自变量做微小改变，并计算函数值
        x[i] = tmp + h
        fxh1 = f(x)
        x[i] = tmp - h
        fxh2 = f(x)
        # 中心差分
        grad[i] = (fxh1 - fxh2)/(2 * h)
        # 还原x[i]的值
        x[i] = tmp
    return grad

# 扩展到二维情况：X为n×m的矩阵，表示n个输入数据
def numerical_gradient(f, X):
    # 如果是1维，直接调用底层函数
    if X.ndim == 1:
        return _numerical_gradient(f, X)
    else:
        grad = np.zeros_like(X)
        # 遍历X中的每一行，分别传入底层函数
        for i, x in enumerate(X):
            grad[i] = _numerical_gradient(f, x)
        return grad

# 梯度下降法，返回最小值点以及下降路径
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []
    # 循环迭代
    for i in range(step_num):
        x_history.append(x.copy())
        # 计算梯度
        grad = numerical_gradient(f, x)
        # 更新参数
        x -= lr * grad
    return x, np.array(x_history)