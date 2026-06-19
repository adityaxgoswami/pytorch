import torch
import torch.nn as nn
import torchvision
from torch.transforms import transforms
import matplotlib.pyplot as plt 

#device config
device = torch.device("cuda" if torch.cuda.isavaliable() else "cpu")

#hyperparameters
input_size = 784  # 28x28 --> size of the image
hidden_size = 100
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

#MNIST
train_dataset = torchvision.datasets.MNIST(root='./data',train=True,transform = transforms,download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',train=False,transform = transforms)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

ex = iter(train_loader)
samples,labels = ex.next()
print(samples.shape,labels.shape)

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(samples[i][0],cmp='gray')
plt.show()


class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNet,self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLu()
        self.l2 = nn.Linear(hidden_size,num_classes)
        
    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out =self.l2(out)
        return out 
    
model = NeuralNet(input_size,hidden_size,num_classes)

#loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

n_total_step = len(train_loader)

for epoch in range(num_epochs):
    for i,(image,label) in enumerate(train_loader):
        
        image = image.reshape(-1,28*28).to(device)
        label = label.to(device)
        
        
        output = model(image)
        
        loss = criterion(output,label)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        if(i+1)%100 ==0:
            print(f'epoch: {epoch+1}/{num_epochs}, step {i+1}/{n_total_step}, loss={loss.item():.4f}')
            

#test : we don't any of the gradient updated or affected so use torch.no_grad
with torch.no_grad():
    n_correct = 0;
    n_samples =0;
    for images,labels in test_loader:
        #reshaping the image to the input size of our nn
        images = images.reshape(-1,28*28).to(device)
        labels = labels.to(device)
        
        outputs = model(image)
        
        #value,index  : we only need the index value as it gives the labels of the item
        _,predictions = torch.max(outputs,1)
        n_samples+= labels.shape[0] 
        
        #we are storing each of the correct prediction made by our model
        n_correct = (predictions==labels).sum().item()

    acc = 100 * n_correct/n_samples
    print(f'accurary :  {acc}')