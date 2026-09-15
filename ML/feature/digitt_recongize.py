import pandas as pd
import pandas as ps
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler



dataset=pd.read_csv("../data/train.csv")
X=dataset.drop("label",axis=1)
y=dataset["label"]
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
model=LogisticRegression(max_iter=500)
model.fit(x_train,y_train)
print(model.score(x_test,y_test))
digit=x_test[123]
pred=model.predict(digit.reshape(1,-1))
print("预测结果:",pred)
print("真实标签:",y_test.iloc[123])
plt.imshow(digit.reshape(28,28),cmap="gray")
plt.show()





