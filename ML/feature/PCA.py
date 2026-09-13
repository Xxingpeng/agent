import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
x=np.random.randn(1000,3)
pca=PCA(n_components=2)
x_pca=pca.fit_transform(x)
fig=plt.figure(figsize=(12,4))
ax1=fig.add_subplot(121,projection="3d")
ax1.scatter(x[:,0],x[:,1],x[:,2],c="g")
ax1.set_title("Before PCA")
ax1.set_xlabel("Feature 1")
ax1.set_ylabel("Feature 2")
ax1.set_zlabel("Feature 3")
ax2=fig.add_subplot(122)
ax2.scatter(x_pca[:,0],x_pca[:,1])
ax2.set_title("After PCA")
ax2.set_xlabel("PC 1")
ax2.set_ylabel("PC 2")
plt.show()
