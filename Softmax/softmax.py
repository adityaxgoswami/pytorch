import torch
import numpy as np 
import torch.nn as nn

def softmax(x):
    return np.exp(x)/ np.sum(np.exp(x),axis=0)

x = np.array([1.0,2.0,3.0])

output = softmax(x)
print(output)


x= torch.tensor([1.0,2.0,3.0])

output= torch.softmax(x, dim=0)
print(output)

