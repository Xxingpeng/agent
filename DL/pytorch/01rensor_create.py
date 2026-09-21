import torch
import numpy as np
tensor1=torch.tensor()
print(tensor1)
print(tensor1.size())

print(tensor1.dtype)


tensor2=torch.tensor([1,2,3])
print(tensor2.size())
print(tensor2.dtype)
tensor3=torch.tensor([[1,2,3],[4,5,6]])
print(tensor3.size())
print(tensor3.dtype)

ndarray3=np.array([[1,2,3],[4,5,6]])
tensor4=torch.tensor(ndarray3)
print(tensor4.size())

tensor5=torch.tensor([[[1,2,3],[4,5,6]]])
print(tensor5.size())
print(tensor5.dtype)







