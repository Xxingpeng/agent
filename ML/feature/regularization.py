import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression,Lasso,ridge_regression



plt.rcParams['font.sans-serif']=['KaiTi']
plt.rcParams['axes.unicode_minus']=False
#1.构建数据
X=np.linspace(-3,3,300).reshape(-1,1)
y=np.sin(X)+np.random.uniform(low=-0.5,high=0.5,size=300).reshape(-1,1)
fig,ax=plt.subplots(2,3,figsize=(15,8))
ax[0,0].scatter(X,y,color='y')
ax[0,1].scatter(X,y,color='y')
ax[0,2].scatter(X,y,color='y')
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

poly20=PolynomialFeatures(degree=20)

x_train=poly20.fit_transform(x_train)
x_test=poly20.transform(x_test)


#不加正则化线性回归模型
model=LinearRegression()
model.fit(x_train,y_train)
y_pred1=model.predict(x_test)
test_loss1=mean_squared_error(y_test,y_pred1)
ax[0,0].text(-3,1,f"测试误差{test_loss1:.4f}")
ax[0,0].plot(X,model.predict(poly20.transform(X)),color="r")

ax[1,0].bar(np.arange(21),model.coef_.reshape(-1))
plt.show()





plt.show()