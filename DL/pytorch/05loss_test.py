import torch
from torch import nn,optim


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1=nn.Linear(5,3)
        self.linear1.weight.data=torch.tensor(
            [
                [0.1,0.2,0.3],
                [0.7,0.8,0.9],
                [0.4,0.5,0.6],
                [0.2,0.3,0.4],
                [0.5,0.6,0.7],
            ]
        ).T
        self.linear1.bias.data=torch.tensor([0.1,0.2,0.3])

    def forward(self,x):
        y=self.linear1(x)
        return y


X=torch.tensor([[1.0,2.0,3.0,4.0,5.0],[1.0,2.0,3.0,4.0,5.0]],dtype=torch.float)
target=torch.tensor([[0,0,0],[0,0,0]],dtype=torch.float)
model=Model()
y_pred=model(X)
loss=nn.MSELoss()
loss_value=loss(y_pred,target)
print(loss_value)


loss_value.backward()
print(model.linear1.weight.grad)
print(model.linear1.bias.grad)

optimizer=optim.SGD(model.parameters(),lr=0.01)

optimizer.step()
optimizer.zero_grad()









