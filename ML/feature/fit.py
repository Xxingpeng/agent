import numpy as np
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
plt.rcParams['font.sans-serif']=['KaiTi']
plt.rcParams['axes.unicode_minus']=False
#1.构建数据
X=np.linspace(-3,3,300).reshape(-1,1)
y=np.sin(X)+np.random.uniform(low=-0.5,high=0.5,size=300).reshape(-1,1)
fig,ax=plt.subplots(1,3,figsize=(15,4))
ax[0].scatter(X,y,color='y')
ax[1].scatter(X,y,color='y')
ax[2].scatter(X,y,color='y')

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LinearRegression()
#欠拟合
model.fit(x_train,y_train)
print("斜率为:",model.coef_)
print("截距为:",model.intercept_)
ax[0].plot(X,model.predict(X),color="r")
#计算误差:训练误差和测试误差
y_pred=model.predict(x_test)
test_loss=mean_squared_error(y_test,y_pred)
train_loss=mean_squared_error(y_train,model.predict(x_train))

ax[0].text(-3,1,f"测试误差：{test_loss:.4f}")
ax[0].text(-3,1.3,f"测试误差：{train_loss:.4f}")

#正好拟合


poly5=PolynomialFeatures(degree=5)
x_train2=poly5.fit_transform(x_train)
x_test2=poly5.transform(x_test)
model.fit(x_train2,y_train)

print("斜率为:",model.coef_)
print("截距为:",model.intercept_)
ax[1].plot(X,model.predict(poly5.fit_transform(X)),color="r")
#计算误差:训练误差和测试误差
y_pred2=model.predict(x_test2)
test_loss1=mean_squared_error(y_test,y_pred2)
train_loss1=mean_squared_error(y_train,model.predict(x_train2))

ax[1].text(-3,1,f"测试误差：{test_loss1:.4f}")
ax[1].text(-3,1.3,f"测试误差：{train_loss1:.4f}")

#过拟合
poly20=PolynomialFeatures(degree=20)
x_train3=poly20.fit_transform(x_train)
x_test3=poly20.transform(x_test)
model.fit(x_train3,y_train)

print("斜率为:",model.coef_)
print("截距为:",model.intercept_)
ax[2].plot(X,model.predict(poly5.fit_transform(X)),color="r")
#计算误差:训练误差和测试误差
y_pred3=model.predict(x_test3)
test_loss2=mean_squared_error(y_test,y_pred3)
train_loss2=mean_squared_error(y_train,model.predict(x_train3))

ax[2].text(-3,1,f"测试误差：{test_loss2:.4f}")
ax[2].text(-3,1.3,f"测试误差：{train_loss2:.4f}")
plt.show()
