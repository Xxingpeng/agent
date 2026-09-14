from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

X,y=make_classification(n_samples=1000,n_features=20,n_classes=2,random_state=42)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

report=classification_report(y_test,y_pred)
print(report)
