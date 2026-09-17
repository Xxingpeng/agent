import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score,calinski_harabasz_score



X,y=make_blobs(n_samples=300,n_features=2,centers=3,cluster_std=2,random_state=42)
fig, ax = plt.subplots(2, 1, figsize=(8, 8))
ax[0].scatter(X[:,0],X[:,1],c="gray",s=50,label="原始数据")
ax[0].set_title("原始数据")
ax[0].legend()

kmeans=KMeans(n_clusters=3)
kmeans.fit(X)
centers=kmeans.cluster_centers_
print(centers)
y_pred=kmeans.predict(X)
print("轮廓系数",silhouette_score(X, y_pred))
print("Calinski-Harabasz指数",calinski_harabasz_score(X, y_pred))
print("蔟内平方和",kmeans.inertia_)


ax[1].scatter(X[:,0],X[:,1],c=y_pred,s=50,label="聚类后")
ax[1].scatter(centers[:,0],centers[:,1],c="red",s=200,marker="*",label="中心点")
ax[1].set_title("Kmeans聚类结果")
ax[1].legend()
plt.show()