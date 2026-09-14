#归一化
from sklearn.preprocessing import MinMaxScaler,StandardScaler
X=[[2,1],[3,1],[1,4],[2,6]]
X=MinMaxScaler(feature_range=(-1,1)).fit_transform(X)
X1=StandardScaler().fit_transform(X)
print(X1)

print(X)