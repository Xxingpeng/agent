import torch
import matplotlib.pyplot as plt
from torch import nn,optim
from torch.utils.data import TensorDataset,DataLoader

x=torch.randn(100,1)
w=torch.tensor([2.5])
b=torch.tensor([5.2])
noise=torch.rand(100,1)*0.1
y=w*x+b+noise
dataset=TensorDataset(x,y)

dataloader=DataLoader(dataset,batch_size=10,shuffle=True)
model=nn.Linear(1,1)
loss=nn.MSELoss()
optimizer=optim.SGD(model.parameters(),lr=0.001)
for epoch in range(100):
    for batch_x,batch_y in dataloader:

        pred=model(batch_x)
        l=loss(pred,batch_y)
        l.backward()
        optimizer.step()
        optimizer.zero_grad()

print("斜率",model.weight)
print("截距",model.bias)


