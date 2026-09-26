import torch
from torch import nn,optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from torch.utils.data import TensorDataset, DataLoader


def create_datasset():

    data=pd.read_csv("../data/house_prices.csv")
    data.drop(["Id"],axis=1,inplace=True)
    x=data.drop("SalePrice",axis=1)
    y=data["SalePrice"]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    numerical_features=x.select_dtypes(exclude="object").columns
    categorical_features=x.select_dtypes(include="object").columns
    numerical_transformer=Pipeline(steps=[("fillna",SimpleImputer(strategy="median")),("sta",StandardScaler())])
    categorical_transformer=Pipeline(steps=[("fillna",SimpleImputer(strategy="constant",fill_value="NaN")),("onehot",OneHotEncoder(handle_unknown="ignore"))])

    column_Transformer=ColumnTransformer(transformers=[
        ("num",numerical_transformer,numerical_features),
        ("cat",categorical_transformer,categorical_features)
    ])
    x_train=x_train.toarray()
    x_test=x_test.toarray()
    train_dataset=TensorDataset(torch.Tensor(x_train),torch.Tensor(y_train.values))
    test_dataset=TensorDataset(torch.Tensor(x_test),torch.Tensor(y_test.values))
    return train_dataset,test_dataset,x_train.shape[1]


train_dataset,test_dataset,feature_num=create_datasset()


model=nn.Sequential(
    nn.Linear(feature_num,128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128,1)
)


def init_weights(layer):
    if isinstance(layer,nn.Linear):
        nn.init.kaiming_normal_(layer.weight)
model.apply(init_weights)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=model.to(device)
lr=0.1
batch_size=64
epoch_num=200
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
optimizer=optim.Adam(model.parameters(),lr=lr)
def log_rmse(pred,target):
    pred=torch.clamp(pred,min=1.0,max=float("inf"))
    mse=nn.MSELoss()
    loss_value=mse(torch.log(pred),torch.log(target))
    return torch.sqrt(loss_value)

train_loss_list=[]
test_loss_list=[]
for epoch in range(epoch_num):
    model.train()
    train_loss_total=0
    for x,y in train_loader:
        x,y=x.to(device),y.to(device)
        y_pred=model(x)
        loss_value=log_rmse(y_pred.squeeze(),y)
        loss_value.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss_total+=loss_value.item()*x.shape[0]

    this_train_loss=train_loss_total/len(train_dataset)
    train_loss_list.append(this_train_loss)
    model.eval()
    test_loss_total=0
    with torch.no_grad():
        for x,y in test_loader:
            x,y=x.to(device),y.to(device)
            y_pred=model(x)
            loss_value=log_rmse(y_pred.squeeze(),y)
            test_loss_total+=loss_value.item()*x.shape[0]

    this_test_loss=test_loss_total/len(test_dataset)
    test_loss_list.append(this_test_loss)




