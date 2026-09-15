import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.metrics import mean_squared_error
dataset=pd.read_csv("../data/advertising.csv")
X=dataset.drop(dataset.columns[0],axis=1,inplace=True)
dataset.dropna(inplace=True)
X=dataset.drop("Sales",axis=1)
y=dataset["Sales"]
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

model=LinearRegression()
model_sgd=SGDRegressor()
model.fit(x_train,y_train)
print("正规方程法模型系数",model.coef_)
print("正规方程法模型截距",model.intercept_)
model_sgd.fit(x_train,y_train)
print("SGD模型系数",model.coef_)
print("SGD模型截距",model.intercept_)
y_pred_lr=model.predict(x_test)
y_pred_sgd=model_sgd.predict(x_test)
print("正规方程法:MSE",mean_squared_error(y_test, y_pred_lr))

print("SGD：MSE",mean_squared_error(y_test, y_pred_sgd))
print("正规方程决定系数",model.score(x_test,y_test))
print("SGD决定系数",model_sgd.score(x_test,y_test))





