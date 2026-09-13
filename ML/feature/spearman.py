import pandas as pd
x=[[5],[8],[10],[12],[15],[3],[7],[9],[14],[6]]
y=[55,65,70,75,85,50,60,72,80,58]
X=pd.DataFrame(x)
Y=pd.DataFrame(y)
print(X.corrwith(Y,method="spearman"))