import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report
label=["猫","狗"]
y_true=["猫","猫","猫","猫","猫","猫","狗","狗","狗","狗|"]
y_pred1=["猫","猫","狗","猫","猫","猫","猫","猫","狗","狗|"]
matrix=confusion_matrix(y_true,y_pred1,labels=label)
print(matrix)
print(pd.DataFrame(matrix,columns=label,index=label))
print(accuracy_score(y_true,y_pred1))
print(precision_score(y_true,y_pred1,pos_label="猫"))
report=classification_report(y_true,y_pred1,labels=label,target_name=["猫"])

