import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv("../data/advertising.csv")
data.dropna(inplace=True)
data.drop(data.columns[0],axis=1,inplace=True)
x=data.drop("Sales",axis=1)
y=data["Sales"]
corr_matrix=data.corr(method="pearson")
sns.heatmap(corr_matrix,annot=True,fmt=".2f",cmap="coolwarm")
plt.show()
