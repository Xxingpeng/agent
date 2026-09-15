import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import joblib
from tornado.gen import multi

from ML.feature.regularization import x_train

dataset=pd.read_csv("../data/heart_disease.csv")
dataset.dropna(inplace=True)

x=dataset.drop(["是否患有心脏病"],axis=1)
y=dataset["是否患有心脏病"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
numerical_features=["年龄","静息血压","胆固醇","最大心率","运动后的ST下降","主血管数量"]
categorical_features=["胸痛类型","静息心电图结果","峰值ST段的斜率","地中海贫血"]
binary_features=["性别","空腹血糖","运动性心绞痛"]
columnsTransformer=ColumnTransformer(
    transformers=[
        ('num',StandardScaler(),numerical_features),
        ('cat',OneHotEncoder(drop="first"),categorical_features),
        ('bin',"passthrough",binary_features),

    ]
)

x_train=columnsTransformer.fit_transform(x_train)
x_test=columnsTransformer.transform(x_test)
model=LogisticRegression(
    solver="saga",
    max_iter=1000,random_state=42,penalty="l1",C=0.5
)
model.fit(x_train,y_train)
print(model.score(x_test, y_test))
