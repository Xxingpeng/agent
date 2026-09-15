from sklearn.linear_model import LinearRegression
x=[[5],[8],[10],[12],[15],[3],[7],[9],[14],[6]]
y=[55,65,70,85,50,60,72,80,58]
model=LinearRegression()
model.fit(x,y)
x_test=[[11]]
y_pred=model.predict(x_test)