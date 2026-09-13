import numpy as np
from sklearn.feature_selection import VarianceThreshold
#1.低方差过滤法
x1=np.random.randn(100)
print(x1.var())
x2=np.random.normal(5,0.1,100)
x=np.array(x1,x2).T
vt=VarianceThreshold(0.01)
x_filtered=vt.fit_transform(x)
