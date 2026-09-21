import torch
import torch.nn as nn
class nn_model(nn.Module):
    def __init__(self):
        super(nn_model,self).__init__()
        self.linear1=nn.Linear(3,4)
        nn.init.xavier_normal_(self.linear1.weight)
        self.linear2=nn.Linear(4,4)
        nn.init.kaiming_normal_(self.linear2.weight)
        self.out=nn.Linear(4,2)

    def forward(self,x):
        x=self.linear1(x)
        x=torch.tanh(x)
        x=self.linear2(x)
        x=torch.relu(x)
        x=self.out(x)
        x=torch.softmax(x,dim=1)

        return x


if __name__ == '__main__':
    x=torch.randn(4,3)
    model=nn_model()
    y_pred=model(x)
    print(y_pred)
