# better prediction ---> low cross entropy
# D(y',y) = - 1/N  * sum(y_i * log(y'_i))

import numpy as np
import torch
import torch.nn as nn

def cross_entropy(y_pred,y):
    loss = -np.sum(y * np.log(y_pred)) / y.shape[0]
    

y=np.array([1,0,0])

y_pred_good = np.array([0.7,0.2,0.1])
y_pred_bad = np.array([0.1,0.3,0.6])

l1 = cross_entropy(y_pred_good,y)
l2 = cross_entropy(y_pred_bad,y)
print(f'loss1 entropy :  {l1:.4f}')
print(f'loss2 entropy :  {l2:.4f}')



# USING TORCH UTILITY

loss = nn.CrossEntropyLoss()

Y = torch.tensor([0])
#nsamples x nclasses = 1x3
Y_pred_good = torch.tensor([[2.0,1.0,0.1]])
Y_pred_bad = torch.tensor([[0.5,2.0,0.3]])

l1=loss(Y_pred_good,Y)
l2=loss(Y_pred_bad,Y)

print(l1.item())
print(l2.item()) 

_,prediction1 = torch.max(Y_pred_good,1)
_,prediction2 = torch.max(Y_pred_bad,1)

#printing the predicted value --> column
print(prediction1)
print(prediction2)
