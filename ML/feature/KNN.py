from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=2)
x=[[2,1],[3,1],[1,4],[2,6]]
y=[0,0,1,1]

knn.fit(x,y)
x1=[[4,9]]

x_class=knn.predict(x1)
print(x_class)